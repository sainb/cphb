from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, row_number, lit
from pyspark.sql.window import Window
import argparse

spark = SparkSession.builder.appName("candidate_gen").getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--queries_table", required=True)
parser.add_argument("--products_table", required=True)
parser.add_argument("--output_table", required=True)
parser.add_argument("--ds", required=True)
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num_shards", type=int, required=True)
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()

queries = (
    spark.table(args.queries_table)
    .where(f"hash(query_id) % {args.num_shards} = {args.shard}")
)
products = spark.table(args.products_table).select("product_id")

# TODO: Replace stub retrieval with real ANN/BM25 logic
cand = (
    queries.crossJoin(products)
    .withColumn("retrieval_score", rand())
    .withColumn(
        "rnk",
        row_number().over(Window.partitionBy("query_id").orderBy("retrieval_score"))
    )
    .where(f"rnk <= {args.k}")
    .select("query_id", "product_id", "retrieval_score")
)

(
    cand.withColumn("source", lit(args.source))
        .withColumn("shard", lit(args.shard))
        .withColumn("ds", lit(args.ds))
        .write.mode("overwrite")
        .insertInto(args.output_table, overwrite=True)
)
