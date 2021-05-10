# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ##Import Spark SQL and Spark ML Libraries
# MAGIC Import all the Spark SQL and ML libraries as mentioned below. This is neccessary to access the functions available in those libraries.

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## To run the code in PySpark CLI
# MAGIC Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC Generate .py(Python) file from Databricks: File > Export > Source File
# MAGIC ```
# MAGIC Run it at the Hadoop/Spark cluster:
# MAGIC $ spark-submit Random Forest Regression.py
# MAGIC ```

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Read the csv file from DBFS (Databricks File System)
# MAGIC Locate the data file, mention its type and read the file as a pyspark dataframe

# COMMAND ----------

# File location and type
file_location = "/user/smurali2/airbnb_dataset/airbnb_US.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
# Load the csv file as a pyspark dataframe
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#df.show(10)



# COMMAND ----------

# MAGIC %md
# MAGIC ##Create a temporary view of the dataframe 'df'

# COMMAND ----------

# Create a view or table
temp_table_name = "airbnb_sample_csv"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

if PYSPARK_CLI:
    csv = spark.read.csv('/user/smurali2/airbnb_dataset/airbnb_US.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM airbnb_sample_csv")

##csv.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Selecting features
# MAGIC In the following step, we are selecting the features that are useful for Price Prediction.

# COMMAND ----------

# Select features and label
data=csv.select("Host Listings Count",	"Host Total Listings Count",	"Neighbourhood",	"Latitude","Longitude",	"Property Type",	"Room Type","Bed Type",	"Accommodates",	"Bathrooms",	"Bedrooms","Monthly Price","Cleaning Fee",	"Guests Included","Extra People","Minimum Nights","Review Scores Rating",	"Review Scores Accuracy",	"Review Scores Cleanliness",	"Review Scores Checkin","Review Scores Communication",	"Review Scores Location",	"Review Scores Value","Sentiment",col("Price").alias("label"))

#data.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning
# MAGIC **Detecting and Removing Outliers:** We determine the **5th** percentile and **95th** percentile values of each of the features. Then filter the dataframe to contain data between these values.

# COMMAND ----------

# approxQuantile() to determine the 5th and 95th percentile values
outliers = data.stat.approxQuantile(['label',"Host Listings Count","Host Total Listings Count","Accommodates","Bathrooms","Bedrooms","Monthly Price", "Cleaning Fee","Guests Included","Extra People","Minimum Nights"],[0.05,0.95],0.0)
print(outliers)

import pyspark.sql.functions as f

# Filtering the dataframe by removing the outliers
data = data.filter((f.col('label') >= outliers[0][0]) & (f.col('label') <= outliers[0][1]))
data = data.filter((f.col('Host Listings Count') >= outliers[1][0]) & (f.col('Host Listings Count') <= outliers[1][1]))
data = data.filter((f.col('Host Total Listings Count') >= outliers[2][0]) & (f.col('Host Total Listings Count') <= outliers[2][1]))
data = data.filter((f.col('Accommodates') >= outliers[3][0]) & (f.col('Accommodates') <= outliers[3][1]))
data = data.filter((f.col('Bathrooms') >= outliers[4][0]) & (f.col('Bathrooms') <= outliers[4][1]))
data = data.filter((f.col('Bedrooms') >= outliers[5][0]) & (f.col('Bedrooms') <= outliers[5][1]))
data = data.filter((f.col('Monthly Price') >= outliers[6][0]) & (f.col('Monthly Price') <= outliers[6][1]))
data = data.filter((f.col('Cleaning Fee') >= outliers[7][0]) & (f.col('Cleaning Fee') <= outliers[7][1]))
data = data.filter((f.col('Guests Included') >= outliers[8][0]) & (f.col('Guests Included') <= outliers[8][1]))
data = data.filter((f.col('Extra People') >= outliers[9][0]) & (f.col('Extra People') <= outliers[9][1]))
data = data.filter((f.col('Minimum Nights') >= outliers[10][0]) & (f.col('Minimum Nights') <= outliers[10][1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Data Cleaning
# MAGIC **Handling Missing Values:** Filling the missing values of numeric columns with **'0'** and string columns with **'NA'**

# COMMAND ----------

# Replacing missing values with '0' and 'NA' for numeric columns and string columns respectively
data_clean = data.na.fill(value=0).na.fill("NA")
#data_clean.show(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation
# MAGIC Determine the correlation of the label **'price'** with the features of the data indicating the **dependence** between the label and each of the features.

# COMMAND ----------

import six

df_Corr=data_clean.select("Host Listings Count","Host Total Listings Count","Neighbourhood",	"Latitude","Longitude","Property Type","Room Type","Bed Type","Accommodates","Bathrooms","Bedrooms",	"Monthly Price","Cleaning Fee",	"Guests Included","Extra People","Minimum Nights","Review Scores Rating","Review Scores Accuracy","Review Scores Cleanliness","Review Scores Checkin","Review Scores Communication","Review Scores Location","Review Scores Value","Sentiment","label")

# Determining correlation using DataFrameStatFunctions.corr
for i in df_Corr.columns:
   if not( isinstance(df_Corr.select(i).take(1)[0][0], six.string_types)):
      print( "Correlation to PRICE for ", i, df_Corr.stat.corr('label',i))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Convert the string type columns into indices using StringIndexer

# COMMAND ----------

# Converting the String type columns into indices 
data_clean = StringIndexer(inputCol='Property Type', outputCol='PropertyType_index').fit(data_clean).transform(data_clean)
data_clean = StringIndexer(inputCol='Room Type', outputCol='RoomType_index').fit(data_clean).transform(data_clean)
data_clean = StringIndexer(inputCol='Bed Type', outputCol='BedType_index').fit(data_clean).transform(data_clean)

#data_clean.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Split the Data
# MAGIC Split the data into train data and test data. We have split the data in the ratio of **70:30**

# COMMAND ----------

# Split the data for random forest regression
splits = data_clean.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

print ("Training Rows:", train.count(), " Testing Rows:", test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Pipeline
# MAGIC Define a pipeline that creates a feature vector and trains a regression model
# MAGIC 1. A **VectorAssembler** that combines categorical features into a single vector.
# MAGIC 2. A **Vector Indexer** that creates indices for a vector of categorical features.
# MAGIC 3. A **VectorAssembler** that creates a vector of continuous numeric features.
# MAGIC 4. A **MinMaxScaler** to normalize the continuous numeric features.
# MAGIC 5. A **VectorAssembler** that creates a vector of categorical and continuous features.
# MAGIC 6. A **Random Forest Regressor** that trains a Regression model.
# MAGIC 7. **Process pipeline** with the series of transformations above.

# COMMAND ----------

# Combine Categorical features into a single vector
catVect = VectorAssembler(inputCols =['PropertyType_index', 'RoomType_index','BedType_index'], outputCol="catFeatures")

# Create indices for the vector of categorical features
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures").setHandleInvalid("skip")

#Create a vector of the numeric features
numVect = VectorAssembler(inputCols = ["Host Listings Count","Host Total Listings Count","Latitude","Longitude","Accommodates","Bathrooms","Bedrooms","Monthly Price","Cleaning Fee","Guests Included","Extra People","Minimum Nights","Review Scores Rating","Review Scores Accuracy","Review Scores Cleanliness","Review Scores Checkin","Review Scores Communication","Review Scores Location","Review Scores Value","Sentiment"], outputCol="numFeatures")

# Scale the numeric features
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

#Create a vector of categorical and numeric features
featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"],  outputCol="features")

# Random Forest Regression model
rf = RandomForestRegressor(labelCol="label", featuresCol="features",maxBins=40)

# Process the pipeline with the transformations
pipeline = Pipeline(stages=[catVect,catIdx,numVect, minMax,featVect, rf])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Train a Regression model using Parameter Tuning
# MAGIC Use the  **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple folds of the data split into training and validation datasets, in order to find the best performing parameters. It is used to find the best model for the data. Here the number of folds is assigned to **2**.

# COMMAND ----------

# Defining the parameter grid
paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth,[2,3,20])
            .addGrid(rf.minInfoGain,[0.0, 0.7])
            .build())

# Number of folds
K = 5
cv = CrossValidator(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, numFolds=K)
#tvs = TrainValidationSplit(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

# Train the model
model = cv.fit(train)


# COMMAND ----------

# MAGIC %md ### Test the Pipeline Model
# MAGIC The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, we will transform the **test** DataFrame using the pipeline to generate label predictions.

# COMMAND ----------

# Transform the test data and generate predictions by applying the trained model
prediction = model.transform(test)
predicted = prediction.select("normFeatures", "prediction", "trueLabel")
#predicted.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine the Predicted and Actual Values
# MAGIC Plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot forms a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected.The cells below creates a temporary table from the **predicted** DataFrame and then retrieves the predicted and actual label values using SQL. Display the results as a scatter plot.

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

# Data Visualization
#dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
#display(dataPred)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the model
# MAGIC Metrics used for evaluation are **Root Mean Square Error(RMSE)** and **Co-efficient of Determination(r2)**. RMSE is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average dollars between predicted and actual price values. r2 indicates how close the data are to the fitted regression line. **RegressionEvaluator** class is used to determine **RMSE** and **r2**.

# COMMAND ----------

# Evaluator to determine rmse
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)

# Evaluator to determine r2
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(prediction)

print ("Root Mean Square Error (RMSE):", rmse)
print ("Co-efficient of Determination", r2)

