# Databricks notebook source
#import all libraries here
pip install nltk

# COMMAND ----------

# MAGIC %md
# MAGIC Part 1: Join the reddit Glossier Comments dataset sentiment+ the competitor comments Data set senemtiment with the Google trends data set 

# COMMAND ----------

#import the glossier comment dataset 
glos_comm = spark.read.parquet("/FileStore/glossier/glossier_comments")
#import the competitor comment data set 
competitor_comments = spark.read.parquet("dbfs:/FileStore/glossier/competitor_comments")
#import the google trends data set 
google = spark.read.csv('/FileStore/google_trends_competitors.csv', header='true')
google.show(5)

# COMMAND ----------

#Data cleaning 
#import libraries
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import udf, col, lower, regexp_replace, translate
import re 
#preliminary data cleaning in order to join appropriate colunms 
#Glossier 
#we know all we are interested in is tge body and the date 
glos_comm2 = glos_comm.select("body","created_utc")
#change type 
glos_comm3 = glos_comm2.withColumn("created_utc",glos_comm2.created_utc.cast('timestamp'))
#glos_comm3.show(5)
## converting all words to lowercase
glos_comm4 = glos_comm3 .withColumn("body",lower(translate('body', '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~', ' ')))
glos_comm4.show(5)
#Now we can move on to doing actual sentiment analysis 

# COMMAND ----------

#Now create the nlp pipeline
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])

# COMMAND ----------

# running the pipeline
from pyspark.sql.functions import col

empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)

data = glos_comm4.select(col("body").alias("text"))
result= pipelineModel.transform(data)


# COMMAND ----------

import pyspark.sql.functions as F
result = result.select('text', F.explode('sentiment.result').alias("sentiment"))
result.show(5)

# COMMAND ----------

#Now we need to join the gloss_com4 data set with our result 
#we are going to join on body 
#so reamke column in result 
result.createOrReplaceTempView("result_vw")
glos_comm4.createOrReplaceTempView("glossier_vw")
glossier_final= spark.sql("select glossier_vw.*, result_vw.sentiment \
                    from glossier_vw join result_vw on glossier_vw.body = result_vw.text")
glossier_final.show()


# COMMAND ----------

#Now I am going to make the sentiment into a dummy variables 
#2= positive 
#1= neutral 
#0=negative
glossier_final2=glossier_final.na.replace("negative","0")
glossier_final3=glossier_final2.na.replace("neutral","1")
glossier_final4=glossier_final3.na.replace("positive","2")
#change sentiment type to integer
glossier_final5=glossier_final4.withColumn("sentiment",col("sentiment").cast("int"))
glossier_final5.show()

# COMMAND ----------

#Now the last step before the join would be to aggraget based on the day 
#for the sentiment column I am going to take the avg for the day
from pyspark.sql import functions as F
glossier_final6=glossier_final5.select(F.date_format('created_utc','yyyy-MM-dd').alias('day'),'sentiment').groupby('day').mean('sentiment')
glossier_final6.show()
#now 

# COMMAND ----------

#Now we turn this into a pnadas dataframe 
glossier_pd_final = glossier_final6.toPandas()
glossier_pd_final.head(20)

# COMMAND ----------

#Now we have to do the same process on the competitors comments ! 
#A lot of this code is taken from Q4_final (1)
#As a result I have already checked for null values ect 
competitor_list = ["Makeup", "MakeupAddiction"]
competitor_comments2 = competitor_comments.filter(competitor_comments.subreddit.isin(competitor_list))
competitor_comments3=competitor_comments2.select("body","created_utc")
#now we need to filter based on Sephora, Ulta, Fenty and Glossier
from pyspark.sql.functions import col
 
competitor_comments4=competitor_comments3.withColumn("Sephora",col("body").rlike("Sephora|sephora"))
competitor_comments5=competitor_comments4.withColumn("Ulta",col("body").rlike("Ulta|ulta"))
competitor_comments6=competitor_comments5.withColumn("Fenty",col("body").rlike("Fenty|fenty"))
competitor_comments7=competitor_comments6.withColumn("Glossier",col("body").rlike("Glossier|glossier"))
competitor_comments7.show(40)

# COMMAND ----------

#Now we need to fiter based on whether is says true 
competitor_comments_Sephora=competitor_comments7.filter("Sephora==True")
#Now lets replace the True with Sephora 
competitor_comments_Sephora2=competitor_comments_Sephora.withColumn("Sephora",col("Sephora").cast("string"))
competitor_comments_Sephora3=competitor_comments_Sephora2.na.replace("true","Sephora")
competitor_comments_Sephora4=competitor_comments_Sephora3.select("body","created_utc","Sephora")
competitor_comments_Sephora4.show(20)
competitor_comments_Ulta=competitor_comments7.filter("Ulta==True")
competitor_comments_Ulta2=competitor_comments_Ulta.withColumn("Ulta",col("Ulta").cast("string"))
competitor_comments_Ulta3=competitor_comments_Ulta2.na.replace("true","Ulta")
competitor_comments_Ulta4=competitor_comments_Ulta3.select("body","created_utc","Ulta")
competitor_comments_Ulta4.show(20)
competitor_comments_Fenty=competitor_comments7.filter("Fenty==True")
competitor_comments_Fenty2=competitor_comments_Fenty.withColumn("Fenty",col("Fenty").cast("string"))
competitor_comments_Fenty3=competitor_comments_Fenty2.na.replace("true","Fenty")
competitor_comments_Fenty4=competitor_comments_Fenty3.select("body","created_utc","Fenty")
competitor_comments_Fenty4.show(20)
competitor_comments_Glossier=competitor_comments7.filter("Glossier==True")
competitor_comments_Glossier2=competitor_comments_Glossier.withColumn("Glossier",col("Glossier").cast("string"))
competitor_comments_Glossier3=competitor_comments_Glossier2.na.replace("true","Glossier")
competitor_comments_Glossier4=competitor_comments_Glossier3.select("body","created_utc","Glossier")
competitor_comments_Glossier4.show(20)
#Now lets 

# COMMAND ----------

#Okay now I need to stack these dataframes 
#let me rename the colums so they all match 
competitor_comments_Sephora5=competitor_comments_Sephora4.withColumnRenamed("Sephora","Brand")
competitor_comments_Ulta5=competitor_comments_Ulta4.withColumnRenamed("Ulta","Brand")
competitor_comments_Fenty5=competitor_comments_Fenty4.withColumnRenamed("Fenty","Brand")
competitor_comments_Glossier5=competitor_comments_Glossier4.withColumnRenamed("Glossier","Brand")
#Now I need to stack the dataframes 
competitor_temp= competitor_comments_Sephora5.union(competitor_comments_Ulta5)
competitor_temp2= competitor_temp.union(competitor_comments_Fenty5)
competitor_temp3= competitor_temp2.union(competitor_comments_Glossier5)
#one more thing change timestamp and make eveythign in the body lower 
competitor_temp4= competitor_temp3.withColumn("created_utc",competitor_temp3.created_utc.cast('timestamp'))
competitor_temp5= competitor_temp4.withColumn("body",lower(translate('body', '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~', ' ')))
competitor_temp5.show(5)

# COMMAND ----------

#Now we can run the sentiment model 
#Now create the nlp pipeline
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])

# COMMAND ----------

# running the pipeline
from pyspark.sql.functions import col

empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)

data = competitor_temp5.select(col("body").alias("text"))
result= pipelineModel.transform(data)

# COMMAND ----------

import pyspark.sql.functions as F
result = result.select('text', F.explode('sentiment.result').alias("sentiment"))
result.show(5)

# COMMAND ----------

#Now we need to join the gloss_com4 data set with our result 
#we are going to join on body 
#so reamke column in result 
result.createOrReplaceTempView("result_vw")
competitor_temp5.createOrReplaceTempView("competitor_vw")
competitor_final= spark.sql("select competitor_vw.*, result_vw.sentiment \
                    from competitor_vw join result_vw on competitor_vw.body = result_vw.text")
competitor_final.show()

# COMMAND ----------

#Now I am going to make the sentiment into a dummy variables 
#2= positive 
#1= neutral 
#0=negative
competitor_final2=competitor_final.na.replace("negative","0")
competitor_final3=competitor_final2.na.replace("neutral","1")
competitor_final4=competitor_final3.na.replace("positive","2")
#change sentiment type to integer
competitor_final5=competitor_final4.withColumn("sentiment",col("sentiment").cast("int"))
competitor_final5.show()

# COMMAND ----------

#Okay now I need to aggregate by day but since I also have brand I need to take that into account
#filter by brand first 
c_Sephora=competitor_final5.filter("Brand=='Sephora'")
c_Ulta=competitor_final5.filter("Brand=='Ulta'")
c_Fenty=competitor_final5.filter("Brand=='Fenty'")
c_Glossier=competitor_final5.filter("Brand=='Glossier'")
#Now get score
c_Sephora2=c_Sephora.select(F.date_format('created_utc','yyyy-MM dd').alias('day'),'sentiment').groupby('day').mean('sentiment')
c_Sephora2.show(5)
c_Ulta2=c_Ulta.select(F.date_format('created_utc','yyyy-MM dd').alias('day'),'sentiment').groupby('day').mean('sentiment')
c_Ulta2.show(5)
c_Fenty2=c_Fenty.select(F.date_format('created_utc','yyyy-MM dd').alias('day'),'sentiment').groupby('day').mean('sentiment')
c_Fenty2.show(5)
c_Glossier2=c_Glossier.select(F.date_format('created_utc','yyyy-MM dd').alias('day'),'sentiment').groupby('day').mean('sentiment')
c_Glossier2.show(5)

# COMMAND ----------

#make them all into pandas dataframes and add a column with the brand name 
#source: https://stackoverflow.com/questions/24039023/add-column-with-constant-value-to-pandas-dataframe
Sephora_c_pd = c_Sephora2.toPandas()
Sephora_c_pd['Brand']='Sephora'
Sephora_c_pd.head(5)
Ulta_c_pd = c_Ulta2.toPandas()
Ulta_c_pd['Brand']='Ulta'
Ulta_c_pd.head(5)
Fenty_c_pd = c_Fenty2.toPandas()
Fenty_c_pd['Brand']='Fenty'
Fenty_c_pd.head(5)
Glossier_c_pd = c_Glossier2.toPandas()
Glossier_c_pd['Brand']='Glossier'
Glossier_c_pd.head(5)

# COMMAND ----------

#now stack these dataframes 
import pandas as pd
comp_final_pd=pd.concat([Sephora_c_pd, Ulta_c_pd], ignore_index=True, axis=0)
#comp_final_pd.count()
comp_final_pd2=pd.concat([comp_final_pd, Fenty_c_pd], ignore_index=True, axis=0)
#comp_final_pd2.count()
comp_final_pd3=pd.concat([comp_final_pd2, Glossier_c_pd], ignore_index=True, axis=0)
#comp_final_pd3.count()
comp_final_pd3.head(20)

# COMMAND ----------

#Okay now all we have left to do is join on the day with the external data 
glossier_pd_final.rename(columns={'day': 'Date'}, inplace=True)
#print(glossier_pd.dtypes)
#make to datetime object so its consistent 
glossier_pd_final['Date']=pd.to_datetime(glossier_pd_final['Date'])
#print(glossier_pd_final.head(20))
comp_final_pd3.rename(columns={'day': 'Date'}, inplace=True)
#print(comp_final_pd3.dtypes)
comp_final_pd3['Date']=pd.to_datetime(comp_final_pd3['Date'])
#print(comp_final_pd3.head(20))
#google = google.toPandas()
#print(google_pd.dtypes)
google_pd['Date']=pd.to_datetime(google_pd['Date'])
#print(google_pd.head(20))
#Now we can do our join 
final_glossier=glossier_pd_final.merge(google_pd, on='Date')
#Since this is just for Glossier, delete columns we dont want such as Sephora, Ulta and Fenty 
final_glossier2=final_glossier.drop(columns=['Sephora', 'Ulta','Fenty'])
final_glossier2.rename(columns={'avg(sentiment)': 'average_sentiment'}, inplace=True)
#Now sort by date 
final_glossier3=final_glossier2.sort_values(by='Date')
print(final_glossier3)
#Now do the competitor data 
final_competitor=comp_final_pd3.merge(google_pd, on='Date')
final_competitor.rename(columns={'avg(sentiment)': 'average_sentiment'}, inplace=True)
#Now sort by date 
final_competitor2=final_competitor.sort_values(by='Date')
print(final_competitor2)

# COMMAND ----------

final_glossier3.to_csv("/dbfs/FileStore/google_glossier.csv")
final_competitor2.to_csv("/dbfs/FileStore/google_competitors.csv")

# COMMAND ----------


