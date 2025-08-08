from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import argparse

spark = SparkSession.builder.appName("qp_base_features").getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument("--candidates_table", required=True)
parser.add_argument("--queries_feat_table", required=True)
parser.add_argument("--products_feat_table", required=True)
parser.add_argument("--events_table", required=True)
parser.add_argument("--output_table", required=True)
parser.add_argument("--ds", required=True)
args = parser.parse_args()

candidates = spark.table(args.candidates_table).where(f"ds='{args.ds}'")
qf = spark.table(args.queries_feat_table).where(f"ds='{args.ds}'")
pf = spark.table(args.products_feat_table).where(f"ds='{args.ds}'")

# Placeholder for real feature engineering
qp = (
    candidates.join(qf, "query_id").join(pf, "product_id")
    .select("query_id", "product_id", "retrieval_score")
    .withColumn("bm25", lit(0.0))
    .withColumn("title_cos_sim", lit(0.0))
    .withColumn("desc_cos_sim", lit(0.0))
    .withColumn("co_click_rate_30d", lit(0.0))
)

(
    qp.withColumn("ds", lit(args.ds))
      .write.mode("overwrite")
      .insertInto(args.output_table, overwrite=True)
)
