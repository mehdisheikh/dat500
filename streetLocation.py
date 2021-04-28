from pyspark.sql.types import (StructType,
                               StructField,
                               DateType,
                               BooleanType,
                               DoubleType,
                               IntegerType,
                               StringType,
                               TimestampType)
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.functions import unix_timestamp
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import (StructType,
                               StructField,
                               DateType,
                               BooleanType,
                               DoubleType,
                               IntegerType,
                               StringType,
                               TimestampType)
import time
# spark imports
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import csv
from pyspark.sql.types import *
from pyspark.sql.functions import format_number, when
import pyspark.sql.functions as F
import pyspark
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession

# use register the app on the spark


spark = SparkSession.builder.appName("Chicago_crime_analysis").getOrCreate()

# create schema to cast columns to the apropriate type
crimes_schema = StructType([StructField("ID", IntegerType(), True),
                            StructField("Case Number", StringType(), True),
                            StructField("Date", StringType(), True),
                            StructField("Block", StringType(), True),
                            StructField("IUCR", IntegerType(), True),
                            StructField("Primary Type", StringType(), True),
                            StructField("Description", StringType(), True),
                            StructField("Location Description",
                                        StringType(), True),
                            StructField("Arrest", BooleanType(), True),
                            StructField("Domestic", BooleanType(), True),
                            StructField("Beat", IntegerType(), True),
                            StructField("District", IntegerType(), True),
                            StructField("Ward", IntegerType(), True),
                            StructField("Community Area", IntegerType(), True),
                            StructField("FBI Code", IntegerType(), True),
                            StructField("X Coordinate", IntegerType(), True),
                            StructField("Y Coordinate", IntegerType(), True),
                            StructField("Year", IntegerType(), True),
                            StructField("Updated On", StringType(), True),
                            StructField("Latitude", DoubleType(), True),
                            StructField("Longitude", DoubleType(), True),
                            StructField("Location", StringType(), True)
                            ])


############
## read data from HDFS
data = spark.read.csv('/chicago_crime/rows.csv',header = True,schema = crimes_schema)
dataset = data.filter((data['Year'] >2000) & (data['Year'] !=2021))


from pyspark.ml.feature import StringIndexer


##create Date elements like (day, day of week, year, and hour)
dataset = dataset.withColumn("Day", F.split(dataset.Date, " ")[0])


dataset = dataset.withColumn("Day", F.split(dataset.Date, " ")[0])
dataset = dataset.withColumn("Day", F.to_date(dataset.Day, "MM/dd/yyyy"))
dataset = dataset.withColumn("Month", F.month(dataset.Day))
dataset = dataset.withColumn("WeekDay", F.dayofweek(dataset.Day))
dataset = dataset.withColumn("Year", F.year(dataset.Day))
dataset=dataset.withColumn("hour", F.from_unixtime(F.unix_timestamp(dataset.Date,'MM/dd/yyyy hh:mm:ss a'),'HH'))



dataset = dataset.na.drop()
dataset = dataset.drop('Date')
dataset = dataset.drop('Day')

print("***************")
for h in range(0,1):
## filter data 
    dataset=dataset\
            .filter(
                    ((dataset["hour"]>=22) |
                    (dataset["hour"]<=6))
                    &(dataset["Location Description"]=="STREET")
                    &(dataset["Latitude"] < 45)
                 & (dataset["Latitude"] > 40)
                 & (dataset["Longitude"] < -85)
                 & (dataset["Longitude"] > -90))

## vectorize the fields that we want to use in KMeans
    from pyspark.ml.feature import VectorAssembler
    df_load=dataset.na.drop()
    assemble=VectorAssembler(inputCols=[
     'Latitude',
     'Longitude'
    ], outputCol='features')
    assembled_data = assemble.transform(df_load)
## scale the data to be between 0 and 1
    from pyspark.ml.feature import StandardScaler
    scale=StandardScaler(inputCol='features',outputCol='standardized')
    data_scale=scale.fit(assembled_data)
    data_scale_output=data_scale.transform(assembled_data)

    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    silhouette_score=[]
    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', \
                                    metricName='silhouette', distanceMeasure='squaredEuclidean')
    maxScore=-2

    for i in range(3,10):
        print("Start clustering with k=",i)

        KMeans_algo=KMeans(featuresCol='standardized', k=i)

## train data using Kmeans
        KMeans_fit=KMeans_algo.fit(data_scale_output)
## calculate the cluster for each data
        output=KMeans_fit.transform(data_scale_output)
## calculate silhouette score
        score=evaluator.evaluate(output)
        silhouette_score.append(score)

        print("Silhouette Score",i,":",score)


        silhouette_score.append(score)

        print("Silhouette Score",i,":",score)




        fig = plt.figure(figsize=(8,6))

        ax = fig.add_subplot(1,1,1)# ,projection='3d')

        my_cmap = plt.get_cmap('hsv')

## plot data
        plt.scatter(df_load.select("Longitude").rdd.map(lambda r: r[0]).collect(),df_load.select("Latitude").rdd.map(lambda r: r[0]).collect(),c= output.select('prediction').rdd.map(lambda r: r[0]).collect(),cmap='rainbow')
        ax.set_xlabel('lang')
        ax.set_ylabel('lat')
        #ax.set_zlabel('class')
        plt.title("Location Description STREET night:K="+str(i)+"\n"+"Silhouette Score:"+str(score))
        plt.savefig('Location Description STREET night K'+str(i)+'.png')


        
## plot silhouettescore
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize =(8,6))
    ax.plot(range(30,10),silhouette_score)
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.savefig('Location Description night k:'+str(i)+'_score.png')

