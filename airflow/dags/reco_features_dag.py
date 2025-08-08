from datetime import datetime, timedelta
import json
from airflow import DAG
from airflow.models import Variable
from airflow.providers.apache.spark.operators.spark_sql import SparkSqlOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.task_group import TaskGroup

# Load pipeline configuration from Airflow Variable
CFG = json.loads(Variable.get("RECO_VARIANTS_JSON"))
TABLES = CFG["tables"]
SOURCES = CFG["sources"]
SHARDS = int(CFG["shards"])
K_GLOBAL_PER_SOURCE = int(CFG["k_global_per_source"])

default_args = {
    "owner": "reco",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="reco_features_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["reco", "features"],
) as dag:

    query_features = SparkSqlOperator(
        task_id="query_features",
        sql=f"""
        SET hive.exec.dynamic.partition.mode=nonstrict;

        INSERT OVERWRITE TABLE {TABLES['features_query']} PARTITION (ds='{{{{ ds }}}}')
        SELECT q.query_id, q.query_text, q.lang,
               pop.daily_search_count, pop.weekly_search_count
        FROM {TABLES['queries']} q
        LEFT JOIN (
            SELECT query_id,
                   SUM(CASE WHEN ds='{{{{ ds }}}}' THEN searches END) AS daily_search_count,
                   SUM(searches) OVER (
                        PARTITION BY query_id ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                   ) AS weekly_search_count
            FROM reco.query_searches
            WHERE ds BETWEEN date_sub('{{{{ ds }}}}', 6) AND '{{{{ ds }}}}'
        ) pop
        ON q.query_id = pop.query_id
        """,
        conn_id="spark_default",
    )

    product_features = SparkSqlOperator(
        task_id="product_features",
        sql=f"""
        SET hive.exec.dynamic.partition.mode=nonstrict;

        INSERT OVERWRITE TABLE {TABLES['features_product']} PARTITION (ds='{{{{ ds }}}}')
        SELECT p.product_id, p.category_id, p.brand_id, p.price, p.in_stock,
               e.clicks_7d, e.addtocarts_7d, e.purchases_7d
        FROM {TABLES['products']} p
        LEFT JOIN (
            SELECT product_id,
                   SUM(IF(event_type='click',1,0)) AS clicks_7d,
                   SUM(IF(event_type='add_to_cart',1,0)) AS addtocarts_7d,
                   SUM(IF(event_type='purchase',1,0)) AS purchases_7d
            FROM {TABLES['events']}
            WHERE ds BETWEEN date_sub('{{{{ ds }}}}',6) AND '{{{{ ds }}}}'
            GROUP BY product_id
        ) e
        ON p.product_id=e.product_id
        """,
        conn_id="spark_default",
    )

    with TaskGroup("candidate_sources") as candidate_sources:
        for source in SOURCES:
            for shard in range(SHARDS):
                SparkSubmitOperator(
                    task_id=f"cand_{source}_{shard}",
                    application="s3://my-bucket/jobs/candidate_gen.py",
                    conn_id="spark_default",
                    application_args=[
                        f"--source={source}",
                        f"--queries_table={TABLES['queries']}",
                        f"--products_table={TABLES['products']}",
                        f"--output_table={TABLES['candidates_by_source']}",
                        f"--ds={{ ds }}",
                        f"--shard={shard}",
                        f"--num_shards={SHARDS}",
                        f"--k={K_GLOBAL_PER_SOURCE}",
                    ],
                    driver_memory="4g",
                    executor_memory="8g",
                    executor_cores=4,
                    num_executors=40,
                )

    union_candidates = SparkSqlOperator(
        task_id="union_candidates",
        sql=f"""
        SET hive.exec.dynamic.partition.mode=nonstrict;

        INSERT OVERWRITE TABLE {TABLES['candidates_global']} PARTITION (ds='{{{{ ds }}}}')
        SELECT query_id, product_id, MAX(retrieval_score) AS retrieval_score
        FROM {TABLES['candidates_by_source']}
        WHERE ds='{{{{ ds }}}}'
        GROUP BY query_id, product_id
        """,
        conn_id="spark_default",
    )

    qp_features = SparkSubmitOperator(
        task_id="qp_features",
        application="s3://my-bucket/jobs/qp_base_features.py",
        conn_id="spark_default",
        application_args=[
            f"--candidates_table={TABLES['candidates_global']}",
            f"--queries_feat_table={TABLES['features_query']}",
            f"--products_feat_table={TABLES['features_product']}",
            f"--events_table={TABLES['events']}",
            f"--output_table={TABLES['features_qp_base']}",
            f"--ds={{ ds }}",
        ],
        driver_memory="6g",
        executor_memory="12g",
        executor_cores=4,
        num_executors=60,
    )

    query_features >> product_features >> candidate_sources >> union_candidates >> qp_features
