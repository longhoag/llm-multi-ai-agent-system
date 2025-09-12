import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Get job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH', 'SYMBOL'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print("🚀 Starting feature engineering for {}".format(args['SYMBOL']))
print("📥 Input: {}".format(args['INPUT_PATH']))
print("📤 Output: {}".format(args['OUTPUT_PATH']))

# Read NDJSON data from S3 - no nesting to extract
df = spark.read.json(args['INPUT_PATH'])
print("📊 DataFrame schema:")
df.printSchema()
print("📊 Record count: {}".format(df.count()))

# Show sample data
print("📋 Sample records:")
df.show(5, truncate=False)

# Feature Engineering - use date for ordering
window_spec = Window.partitionBy().orderBy("date")

# Technical Indicators
df_features = df.withColumn(
    "sma_5", F.avg("close").over(window_spec.rowsBetween(-4, 0))
).withColumn(
    "sma_20", F.avg("close").over(window_spec.rowsBetween(-19, 0))
).withColumn(
    "price_change", F.col("close") - F.lag("close", 1).over(window_spec)
).withColumn(
    "price_change_pct", 
    (F.col("close") - F.lag("close", 1).over(window_spec)) / F.lag("close", 1).over(window_spec) * 100
).withColumn(
    "volatility_5d", 
    F.stddev("close").over(window_spec.rowsBetween(-4, 0))
).withColumn(
    "volume_sma_5", F.avg("volume").over(window_spec.rowsBetween(-4, 0))
).withColumn(
    "high_low_ratio", F.col("high") / F.col("low")
).withColumn(
    "processing_timestamp", F.current_timestamp()
).withColumn(
    "symbol_processed", F.lit(args['SYMBOL'])
)

# Add feature metadata
feature_columns = [col for col in df_features.columns if col.startswith(('sma_', 'price_', 'volatility_', 'volume_', 'high_low'))]
feature_count = len(feature_columns)

df_final = df_features.withColumn(
    "feature_engineering_metadata", 
    F.struct(
        F.lit(feature_count).alias("feature_count"),
        F.lit("glue_pyspark").alias("processing_method"),
        F.current_timestamp().alias("created_at")
    )
)

print("📊 Final DataFrame schema:")
df_final.printSchema()
print("📊 Final record count: {}".format(df_final.count()))

# Show sample processed data
print("📋 Sample processed records with features:")
df_final.select("date", "symbol", "close", "sma_5", "sma_20", "price_change", "volatility_5d").show(5)

# Write processed data back to S3
df_final.coalesce(1).write.mode("overwrite").json(args['OUTPUT_PATH'])

print("✅ Feature engineering completed for {}".format(args['SYMBOL']))
print("📊 Created {} features".format(feature_count))
print("📁 Output written to: {}".format(args['OUTPUT_PATH']))

job.commit()
