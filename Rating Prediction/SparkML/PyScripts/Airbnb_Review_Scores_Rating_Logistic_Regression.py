# Databricks notebook source
# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# File location and type
file_location = "/user/smurali2/airbnb_dataset/airbnb_US.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#display(df)

# COMMAND ----------

# Create a view or table
temp_table_name = "airbnb_sample_csv"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

if PYSPARK_CLI:
    csv = spark.read.csv('/user/smurali2/airbnb_dataset/airbnb_US.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM airbnb_sample_csv")
    
#csv.show(5)

# COMMAND ----------


csv = csv.withColumn("Review Scores Rating", when(col("Review Scores Rating") >= 80,1).otherwise(0))
#csv.show()


# COMMAND ----------

csv = csv.filter(col("Minimum Nights")<= 365)
data = csv.select("Host Response Time",
                  "Minimum Nights","Accommodates","Bathrooms","Bedrooms","Beds","Property Type","Extra People",
                  "Security Deposit","Cleaning Fee","Guests Included","Cancellation Policy","Sentiment",col("Review Scores Rating").alias("label"))

#data.show(10)
#data = data.filter(col("Mininum Nights")<= 365)
#display(data.describe())

# COMMAND ----------

data_clean = data.na.fill(value=0).na.fill("")
#data_clean.show(50)


# COMMAND ----------

data_clean = StringIndexer(inputCol='Host Response Time', outputCol='Host_Response_index').fit(data_clean).transform(data_clean)
#data_clean = StringIndexer(inputCol='Neighborhood Cleansed', outputCol='Neighborhood_index').fit(data_clean).transform(data_clean)
data_clean = StringIndexer(inputCol='Property Type', outputCol='PropertyType_index').fit(data_clean).transform(data_clean)
data_clean = StringIndexer(inputCol='Cancellation Policy', outputCol='Cancellation_index').fit(data_clean).transform(data_clean)

# COMMAND ----------

# Split the data
splits = data_clean.randomSplit([0.7, 0.3])

# for logistic  regression
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")


# for gradient boosted tree regression
#gbt_train = splits[0]
#gbt_test = splits[1].withColumnRenamed("label", "trueLabel")

print ("Training Rows:", train.count(), " Testing Rows:", test.count())

# COMMAND ----------

catVect = VectorAssembler(inputCols = ["Host_Response_index","PropertyType_index","Cancellation_index"], outputCol="catFeatures")
#catVect = VectorAssembler(inputCols = [stringIndexer],outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures").setHandleInvalid("skip")
numVect = VectorAssembler(inputCols = ["Minimum Nights","Accommodates","Bathrooms","Bedrooms","Beds","Extra People",
                  "Security Deposit","Cleaning Fee","Guests Included","Sentiment"], outputCol="numFeatures")
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")
featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"],  outputCol="features")
lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[catVect,catIdx,numVect, minMax,featVect, lr])

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01,0.5]).addGrid(lr.maxIter, [10, 5,20]).build()  
val = CrossValidator(estimator=pipeline, evaluator= BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid,
numFolds=5)

#val = TrainValidationSplit(estimator=pipeline, evaluator=BinaryClassificationEvaluator(),  estimatorParamMaps=paramGrid, trainRatio=0.8)


# COMMAND ----------

model=val.fit(train)

# COMMAND ----------

prediction = model.transform(test)
#predicted = prediction.select("features", "prediction", "trueLabel")
predicted = prediction.select("normFeatures", "prediction", "trueLabel")
predicted.show()

# COMMAND ----------

tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()


# COMMAND ----------

# LogisticRegression: rawPredictionCol="prediction", metricName="areaUnderROC"
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print ("AUC = ", auc)
