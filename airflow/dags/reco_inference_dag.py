from datetime import datetime, timedelta
import json
from airflow import DAG
from airflow.models import Variable
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.providers.apache.spark.operators.spark_sql import SparkSqlOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.task_group import TaskGroup

CFG = json.loads(Variable.get("RECO_VARIANTS_JSON"))
TABLES = CFG["tables"]
VARIANTS = CFG["variants"]

default_args = {
    "owner": "reco",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="reco_inference_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 6 * * *",
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["reco", "inference"],
) as dag:

    wait_features = ExternalTaskSensor(
        task_id="wait_features",
        external_dag_id="reco_features_dag",
        external_task_id="qp_features",
        execution_delta=timedelta(hours=4),
        poke_interval=60,
        timeout=6 * 60 * 60,
        mode="reschedule",
    )

    for variant in VARIANTS:
        name = variant["name"]
        sources = ",".join([f"'{s}'" for s in variant["sources"]])
        k_cand = variant["topk_candidate"]
        k_out = variant["topk_output"]
        model_uri = variant["ctr_model_uri"]

        with TaskGroup(group_id=f"variant__{name}") as tg:
            variant_candidates = SparkSqlOperator(
                task_id=f"build_candidates_{name}",
                sql=f"""
                SET hive.exec.dynamic.partition.mode=nonstrict;

                INSERT OVERWRITE TABLE {TABLES['candidates_global']}
                PARTITION (ds='{{{{ ds }}}}', variant='{name}')
                WITH unioned AS (
                    SELECT query_id, product_id, MAX(retrieval_score) AS retrieval_score
                    FROM {TABLES['candidates_by_source']}
                    WHERE ds='{{{{ ds }}}}' AND source IN ({sources})
                    GROUP BY query_id, product_id
                ),
                ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY retrieval_score DESC) AS rnk
                    FROM unioned
                )
                SELECT query_id, product_id, retrieval_score FROM ranked WHERE rnk <= {k_cand}
                """,
                conn_id="spark_default",
            )

            build_dataset = SparkSqlOperator(
                task_id=f"build_dataset_{name}",
                sql=f"""
                SET hive.exec.dynamic.partition.mode=nonstrict;

                CREATE TABLE IF NOT EXISTS reco.tmp_inference_{name} USING PARQUET PARTITIONED BY (ds) AS
                SELECT c.query_id, c.product_id,
                       qf.* EXCEPT(query_id, ds),
                       pf.* EXCEPT(product_id, ds),
                       qp.* EXCEPT(query_id, product_id, ds)
                FROM {TABLES['candidates_global']} c
                JOIN {TABLES['features_query']} qf  ON qf.query_id=c.query_id AND qf.ds='{{{{ ds }}}}'
                JOIN {TABLES['features_product']} pf ON pf.product_id=c.product_id AND pf.ds='{{{{ ds }}}}'
                JOIN {TABLES['features_qp_base']} qp ON qp.query_id=c.query_id AND qp.product_id=c.product_id AND qp.ds='{{{{ ds }}}}'
                WHERE c.ds='{{{{ ds }}}}' AND c.variant='{name}'
                """,
                conn_id="spark_default",
            )

            score_ctr = SparkSubmitOperator(
                task_id=f"score_ctr_{name}",
                application="s3://my-bucket/jobs/ctr_inference_xgb.py",
                conn_id="spark_default",
                application_args=[
                    f"--in_table=reco.tmp_inference_{name}",
                    f"--model_uri={model_uri}",
                    f"--out_table=reco.tmp_scores_{name}",
                    f"--ds={{ ds }}",
                ],
                driver_memory="6g",
                executor_memory="12g",
                executor_cores=4,
                num_executors=80,
            )

            write_topk = SparkSqlOperator(
                task_id=f"write_topk_{name}",
                sql=f"""
                SET hive.exec.dynamic.partition.mode=nonstrict;

                INSERT OVERWRITE TABLE {TABLES['predictions']}
                PARTITION (ds='{{{{ ds }}}}', variant='{name}')
                WITH ranked AS (
                    SELECT query_id, product_id, ctr_score,
                           ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY ctr_score DESC) AS rnk
                    FROM reco.tmp_scores_{name}
                    WHERE ds='{{{{ ds }}}}'
                )
                SELECT query_id, product_id, ctr_score FROM ranked WHERE rnk <= {k_out}
                """,
                conn_id="spark_default",
            )

            wait_features >> variant_candidates >> build_dataset >> score_ctr >> write_topk
