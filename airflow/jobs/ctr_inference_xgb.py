from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import argparse
import xgboost as xgb

spark = SparkSession.builder.appName("ctr_inference_xgb").getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument("--in_table", required=True)
parser.add_argument("--model_uri", required=True)
parser.add_argument("--out_table", required=True)
parser.add_argument("--ds", required=True)
args = parser.parse_args()

df = spark.table(args.in_table).where(f"ds='{args.ds}'")
columns = [c for c in df.columns if c not in ("query_id", "product_id", "ds")]

# Convert to pandas for XGBoost scoring
pdf = df.select(columns).toPandas()
model = xgb.Booster()
model.load_model(args.model_uri)

dmat = xgb.DMatrix(pdf)
scores = model.predict(dmat)

out_pdf = df.select("query_id", "product_id").toPandas()
out_pdf["ctr_score"] = scores

(
    spark.createDataFrame(out_pdf)
         .withColumn("ds", lit(args.ds))
         .write.mode("overwrite")
         .insertInto(args.out_table, overwrite=True)
)
