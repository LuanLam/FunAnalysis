#This is a fun project, analysing the property assessment data in San Francisco from 2007 to 2015.

#When this analysis was carried out, there was a link to public data released by the office of the assessor-recorder. The data included aggregated assessment information from 2007 to 2015 [https://data.sfgov.org/api/views/wv5m-vpq2/rows.csv?accessType=DOWNLOAD]. The link was last accessed on 18/10/2016.

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext

import math

from pyspark.sql import functions as F

from pyspark.sql.functions import *
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel


from datetime import datetime
from pyspark.sql.types import DateType 

dataFile = "/user/Luan/Historic_Secured_Property_Tax_Rolls.csv"

 
# Configure the Spark context to give a name to the application
sparkConf = SparkConf().setAppName("propertyTaxRoll")
sc = SparkContext(conf = sparkConf)
 
# Gets the Hive context
sqlContext = HiveContext(sc)

# Loads the CSV file as a Spark DataFrame
df_data = sqlContext.read.load(dataFile,
format='com.databricks.spark.csv',
header='true',
delimiter=',')



df_data_latest_group = df_data.groupBy("Block and Lot Number").agg(F.max("Closed Roll Fiscal Year").alias("MCRFY")).withColumnRenamed("Block and Lot Number", "BALN")


df_data_latest = df_data.join(df_data_latest_group, df_data["Block and Lot Number"] == df_data_latest_group["BALN"])


df_data_latest = df_data_latest.filter(df_data_latest["Closed Roll Fiscal Year"] == df_data_latest["MCRFY"])


df_data_latest = df_data_latest.drop("BALN").drop("MCRFY")

#df_data_latest.show()



#question 1:  What fraction of assessments are for properties of the most common class? 
#consider all the assessments, even though some properties may be listed more than once.

total_number_of_assessments = df_data.count()

df_total_number_of_assessments_of_most_popular_class = df_data.groupBy("Property Class Code").count().agg(F.max("count").alias("max"))

#df_total_number_of_assessments_of_most_popular_class.show()

df_Question_1 = df_total_number_of_assessments_of_most_popular_class.withColumn("Result", F.lit(df_total_number_of_assessments_of_most_popular_class.max / total_number_of_assessments))

df_Question_1.write.save("/user/Luan/Result/Results1")

#question 1

#question 2: What is the median assessed improvement value, considering only non-zero assessments? 
#Consider only the latest assessment value for each property, which is uniquely identified by the "Block and Lot Number" column.
                   
df_data_question_2 = df_data_latest.select("Block and Lot Number", "Closed Roll Assessed Improvement Value", "Closed Roll Fiscal Year")

df_data_question_2 = df_data_question_2.filter(df_data_question_2["Closed Roll Assessed Improvement Value"] != 0)

df_data_question_2 = df_data_question_2.withColumn("Closed Roll Assessed Improvement Value", df_data_question_2["Closed Roll Assessed Improvement Value"].cast("long"))\
                    .withColumn("Closed Roll Fiscal Year", df_data_question_2["Closed Roll Fiscal Year"].cast("int"))

df_data_question_2 = df_data_question_2.withColumnRenamed("Closed Roll Assessed Improvement Value", "ImprovedValue") 

df_data_question_2.registerTempTable("TempTableForQuestion2")

df_data_question_2 = sqlContext.sql("SELECT percentile_approx(ImprovedValue, 0.5) as medianValue FROM TempTableForQuestion2")

df_data_question_2.write.save("/user/Luan/Result/Results2")

#end question 2

#question 3: Calculate the average improvement value (using only non-zero assessments) in each neighborhood. 
#What is the difference between the greatest and least average values?

df_data_question_3 = df_data_latest.select("Neighborhood Code", "Closed Roll Assessed Improvement Value", "Closed Roll Fiscal Year", "Block and Lot Number")

df_data_question_3 = df_data_question_3.filter(df_data_question_3["Closed Roll Assessed Improvement Value"] != 0)

df_data_question_3 = df_data_question_3.withColumn("Closed Roll Assessed Improvement Value", df_data_question_3["Closed Roll Assessed Improvement Value"].cast("long"))\
                    .withColumn("Closed Roll Fiscal Year", df_data_question_3["Closed Roll Fiscal Year"].cast("int"))

df_data_question_3 = df_data_question_3.groupBy("Neighborhood Code").agg(F.mean("Closed Roll Assessed Improvement Value").alias("AvgImprovedValue"))


df_data_question_3 = df_data_question_3.describe()


df_data_question_3.write.save("/user/Luan/Result/Results3")

#end question 3

#question 4:   What is the difference between the average number of units in buildings build in or after 1950, 
#and that for buildings built before 1950? Consider only buildings that have non-zero units reported, 
#and ignore buildings with obviously incorrect years. To try to avoid the effect of improvements to buildings, 
#use the earliest record for each property, not the latest.  
    
df_data_question_4 = df_data_latest.select("Recordation Date", "Number of Units", "Block and Lot Number")

df_data_question_4 = df_data_question_4.withColumnRenamed("Recordation Date", "RecordedDate").withColumnRenamed("Number of Units", "noOfUnit").withColumnRenamed("Block and Lot Number", "ID") 

df_data_question_4 = df_data_question_4.filter((df_data_question_4.RecordedDate != "") & (df_data_question_4.noOfUnit != 0)) 

convertToDateFunc = udf(lambda x: datetime.strptime(x, "%m/%d/%Y"), DateType())

df_data_question_4 = df_data_question_4.withColumn("RecordationDate", convertToDateFunc(df_data_question_4.RecordedDate))

df_data_question_4 = df_data_question_4.drop("RecordedDate")

df_data_question_4_groupby = df_data_question_4.groupBy("ID").agg(F.min("RecordationDate")) # @UndefinedVariable

df_data_question_4_groupby = df_data_question_4_groupby.withColumnRenamed("ID", "IDGroupby")


df_data_question_4 = df_data_question_4.join(df_data_question_4_groupby, df_data_question_4.ID == df_data_question_4_groupby["IDGroupby"])

    
df_data_question_4 = df_data_question_4.filter(df_data_question_4.RecordationDate == df_data_question_4["min(RecordationDate)"]).dropDuplicates()


df_data_question_4.registerTempTable("df_data_question_4_table")

df_data_question_4_from_1950 = sqlContext.sql("SELECT * FROM df_data_question_4_table WHERE YEAR(RecordationDate) >= 1950")

df_data_question_4_before_1950 = sqlContext.sql("SELECT * FROM df_data_question_4_table WHERE YEAR(RecordationDate) < 1950")



df_data_question_4_from_1950 = df_data_question_4_from_1950.agg(F.mean("noOfUnit").alias("averageAfter1950"))



df_data_question_4_before_1950 = df_data_question_4_before_1950.agg(F.mean("noOfUnit").alias("averageBefore1950"))


df_data_question_4_from_1950.write.save("/user/Luan/Result/Results4F1950")
df_data_question_4_before_1950.write.save("/user/Luan/Result/Results4B1950")

#endQuestion 4

#Question 5: Considering only properties with non-zero numbers of bedrooms and units, calculate the average number of bedrooms per unit in each zip code. 
#Use the ratio of the means instead of the mean of the ratio. What is this ratio in the zip code where it achieves its maximum?

df_data_question_5 = df_data_latest.select("Zipcode of Parcel", "Closed Roll Fiscal Year", "Block and Lot Number", "Number of Bedrooms", "Number of Units")

df_data_question_5 = df_data_question_5.filter( (df_data_question_5["Number of Bedrooms"] != 0) & (df_data_question_5["Number of Units"] != 0) & (df_data_question_5["Zipcode of Parcel"] != 0))


df_data_question_5 = df_data_question_5.groupBy("Zipcode of Parcel").agg(F.mean("Number of Bedrooms").alias("ANoOfBed"), F.mean("Number of Units").alias("ANoOfUnit"))

df_data_question_5 = df_data_question_5.withColumn("avgBedUnits", df_data_question_5["ANoOfBed"] / df_data_question_5["ANoOfUnit"])

df_data_question_5 = df_data_question_5.withColumnRenamed("Zipcode of Parcel", "Zip")

df_data_question_5.describe().write.save("/user/Luan/Result/Results5")

#end question 5

#question 6: Estimate how built-up each zip code is by comparing the total property area to the total lot area. 
#What is the largest ratio of property area to surface area of all zip codes?

df_data_question_6 = df_data_latest.select("Zipcode of Parcel", "Closed Roll Fiscal Year", "Block and Lot Number", "Property Area in Square Feet", "Lot Area")

df_data_question_6 = df_data_question_6.filter( (df_data_question_6["Zipcode of Parcel"] != 0) & (df_data_question_6["Property Area in Square Feet"] != 0) & (df_data_question_6["Lot Area"] != 0))


df_data_question_6 = df_data_question_6.groupBy("Zipcode of Parcel").agg(F.sum("Property Area in Square Feet").alias("SumOfPropertyArea"), F.sum("Lot Area").alias("SumOfLotArea")) # @UndefinedVariable


df_data_question_6 = df_data_question_6.withColumn("PropertyToLotRatio", df_data_question_6["SumOfPropertyArea"] / df_data_question_6["SumOfLotArea"])

#df_data_question_6.show()

df_data_question_6.describe().write.save("/user/Luan/Result/Results6")

#end question 6








