# Databricks notebook source
# MAGIC %md
# MAGIC Business Goal 3: We will identify which products are most common amongst competitors to determine where we can capitalize on market share.

# COMMAND ----------

# MAGIC %md
# MAGIC Technical Proposal: For two of Glossier’s competitors, we will use NLP techniques to identify which posts and comments in the competitor subreddits contain the products identified from Glossier’s website. Similar to the business objective above, we will conduct sentiment analysis of each post to assign positive, negative, or neutral values. We will sum the number of posts and comments by product for positive posts. We will identify the top 10 products with the highest activity (positive sentiment). Similarly, we will sum the number of posts and comments by product for negative posts. We will identify the top 10 products with the highest activity (negative sentiment). We will display this information on a faceted chart to depict the worst performing products and highest performing products are of our competitors.

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

# MAGIC %pip install dataframe_image

# COMMAND ----------

# reading in data
competitor_comments = spark.read.parquet("dbfs:/FileStore/glossier/competitor_comments")
competitor_submissions = spark.read.parquet("dbfs:/FileStore/glossier/competitor_submissions")

# COMMAND ----------

# first let's filter out NSFW submissions
competitor_submissions = competitor_submissions.filter(competitor_submissions.over_18 == False)

# now let's filter out moderators/admins
competitor_comments = competitor_comments.filter(competitor_comments.distinguished.isNull())
competitor_submissions = competitor_submissions.filter(competitor_submissions.distinguished.isNull())

# COMMAND ----------

# keeping only necessary columns
tokeep_comments = ["author", "created_utc", "subreddit", "score", "body"]
tokeep_submissions = ["author", "created_utc", "subreddit", "score", "title", "selftext"]

competitor_comments = competitor_comments.select(*tokeep_comments)
competitor_submissions = competitor_submissions.select(*tokeep_submissions)

# COMMAND ----------

competitor_submissions.show(5)

# COMMAND ----------

competitor_comments.show(5)

# COMMAND ----------

# 1031193
competitor_comments.count()

# COMMAND ----------

# combining title and self text to analyze whole submission
from pyspark.sql.functions import concat_ws
competitor_submissions = competitor_submissions.select("author", "created_utc", "subreddit", "score", concat_ws(" ", \
                                                       competitor_submissions.title,competitor_submissions.selftext).alias("body"))
competitor_submissions.show(5)


# COMMAND ----------

# 107510
competitor_submissions.count()

# COMMAND ----------

df_concat = competitor_submissions.union(competitor_comments)
df_concat.show(5)

# COMMAND ----------

# 1138703
# concat was successful!
df_concat.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Doing some data cleaning

# COMMAND ----------

# removing any null values in the body
df_concat = df_concat.filter(df_concat.body.isNotNull())
df_concat = df_concat.filter(df_concat.body != "[removed]")

# COMMAND ----------

## Data Cleaning
## resource: https://stackoverflow.com/questions/53579444/efficient-text-preprocessing-using-pyspark-clean-tokenize-stopwords-stemming

import pyspark.sql.functions as F

## removing non-alphanumeric characters and making lowercase
df_clean = df_concat.select('author', 'created_utc', 'subreddit', 'score', (F.lower(F.regexp_replace('body', "[^a-zA-Z\\s]", "")).alias('body')))
df_clean.show(10)

# COMMAND ----------

# resource: https://medium.com/trustyou-engineering/topic-modelling-with-pyspark-and-spark-nlp-a99d063f1a6e
# resource: https://github.com/maobedkova/TopicModelling_PySpark_SparkNLP/blob/master/Topic_Modelling_with_PySpark_and_Spark_NLP.ipynb
from sparknlp.base import DocumentAssembler

documentAssembler = DocumentAssembler() \
     .setInputCol("body") \
     .setOutputCol('document')

from sparknlp.annotator import Tokenizer

tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

from sparknlp.annotator import Normalizer

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

from sparknlp.annotator import LemmatizerModel

lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

from sparknlp.annotator import StopWordsCleaner
from sparknlp.base import *

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('unigrams') \
     .setStopWords(eng_stopwords)

finisher = Finisher() \
     .setInputCols(['unigrams'])

pipeline = Pipeline() \
     .setStages([documentAssembler,                  
                 tokenizer,
                 normalizer,                  
                 lemmatizer,                  
                 stopwords_cleaner,  
                 finisher])

# COMMAND ----------

processed_review = pipeline.fit(df_clean.select("body")).transform(df_clean.select("body"))
processed_review.show(5)

# COMMAND ----------

# getting text distributions after cleaning

df_temp = processed_review.withColumn('text_dist', F.size('finished_unigrams'))
df_temp.describe("text_dist").show()

# COMMAND ----------

# now we must concat the clean list of words for the sparknlp model
df_clean = processed_review.withColumn("body_clean", F.concat_ws(" ", F.col("finished_unigrams"))).select('body', 'body_clean')
df_clean.show(10)

# COMMAND ----------

# creating NLP pipeline
# using a twitter sentiment pipeline because Reddit and Twitter are similar as two social media platforms

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

data = df_clean.select(col("body_clean").alias("text"))
result = pipelineModel.transform(data)

# COMMAND ----------

result.show(5)

# COMMAND ----------

result.select("sentiment").show(20, truncate=False)

# COMMAND ----------

result.select("sentiment.metadata").show(20, truncate=False)

# COMMAND ----------

temp = result.select('text', F.explode('sentiment.metadata').alias("sent_score"))
temp.show()

# COMMAND ----------

# extracting score
temp = temp.select(col("text"), F.map_keys(col("sent_score")).alias("keys"), F.map_values(col("sent_score")).alias("values"))
temp = temp.select("text", temp.values[1].alias("positive_score"))
temp.show()

# COMMAND ----------

# getting necessary columns
import pyspark.sql.functions as F

result = result.select('text', F.explode('sentiment.result').alias("sentiment"))
result.show(5)

# COMMAND ----------

# rejoining with the rest of the columns
df_temp = df_concat.select("author", "created_utc", "subreddit", "score")

# resource: https://stackoverflow.com/questions/63727512/unable-to-write-pyspark-dataframe-created-from-two-zipped-dataframes
# create rdds with an additional index to join
# as zipWithIndex adds the index as second column

left = result.rdd.zipWithIndex().map(lambda a: (a[1], a[0]))
right= df_temp.rdd.zipWithIndex().map(lambda a: (a[1], a[0]))

# COMMAND ----------

# resource: https://stackoverflow.com/questions/63727512/unable-to-write-pyspark-dataframe-created-from-two-zipped-dataframes
#join both rdds on index
joined = left.fullOuterJoin(right)

# COMMAND ----------

# resource: https://stackoverflow.com/questions/63727512/unable-to-write-pyspark-dataframe-created-from-two-zipped-dataframes
#restore the original columns
df_final = spark.createDataFrame(joined).select("_2._1.*", "_2._2.*")

# COMMAND ----------

df_final.show(5)

# COMMAND ----------

# removing any null values in the sentiment
df_final = df_final.filter(df_final.sentiment.isNotNull())


# COMMAND ----------

# performing tf idf to find important words in predicting sentiment
# resource: https://www.analyticsvidhya.com/blog/2022/09/implementing-count-vectorizer-and-tf-idf-in-nlp-using-pyspark/
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

df_tfidf = df_final.select("sentiment", "text")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df_tfidf)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("sentiment", "features").show(5)

# COMMAND ----------

for features_label in rescaledData.select("features", "sentiment").take(3):
    print(features_label)

# COMMAND ----------

# getting important words using tf idf
# resource: https://stackoverflow.com/questions/69218494/pyspark-display-top-10-words-of-document
from pyspark.sql.types import *

ndf = wordsData.select('sentiment',F.explode('words').name('expwords')).withColumn('words', F.array('expwords'))
hashudf = F.udf(lambda vector : vector.indices.tolist()[0], IntegerType())
wordtf = hashingTF.transform(ndf).withColumn('wordhash',hashudf(F.col('rawFeatures')))
wordtf.show(5)

# COMMAND ----------

wordtf = wordtf.select("sentiment", "expwords", "wordhash")
wordtf.show()

# COMMAND ----------

# getting important words using tf idf
# resource: https://stackoverflow.com/questions/69218494/pyspark-display-top-10-words-of-document
from pyspark.sql.types import MapType

udf1 = F.udf(lambda vec : dict(zip(vec.indices.tolist(),vec.values.tolist())), MapType(IntegerType(),FloatType()))
valuedf = rescaledData.select('sentiment',F.explode(udf1(F.col('features'))).name('wordhash','value'))
valuedf.show(5)

# COMMAND ----------

# valuedf = valuedf.withColumn("wordhash",col("wordhash").cast("int")) \
#                 .withColumn("value",col("value").cast("float"))
valuedf.printSchema()

# COMMAND ----------

# wordtf = wordtf.withColumn("wordhash",col("wordhash").cast("int"))
wordtf.printSchema()

# COMMAND ----------

# getting top 10 words for each sentiment
# resource: https://stackoverflow.com/questions/69218494/pyspark-display-top-10-words-of-document
from pyspark.sql import Window

w = Window.partitionBy('sentiment').orderBy(F.desc('value'))
valuedf = valuedf.withColumn('rank',F.rank().over(w)).where(F.col('rank')<=10)
valuedf.show()

#valuedf.join(wordtf,['sentiment','wordhash']).groupby('sentiment').agg(F.sort_array(F.collect_set(F.struct(F.col('value'),F.col('expwords'))),asc=False).name('topn')).show(truncate=False)

# COMMAND ----------

word_list = list(valuedf.select('wordhash').toPandas()['wordhash'])
len(word_list)

# COMMAND ----------

df_temp = wordtf.filter(wordtf.wordhash.isin(word_list))

# COMMAND ----------

# df_temp.show()

# COMMAND ----------

# extracting brand from general makeup channels and combining with subreddit data to get big picture brand sentiment
df_final = df_final.withColumn(
            'brand',
            F.when((F.col("subreddit") == 'Sephora') | F.col("text").rlike("Sephora|sephora"), "sephora")\
            .when((F.col("subreddit") == 'Ulta') | F.col("text").rlike("Ulta|ulta"), "ulta")\
            .when((F.col("subreddit") == 'Fentybeauty') | F.col("text").rlike("Fenty|fenty"), "fenty")\
            .otherwise("general_makeup_channel"))

# COMMAND ----------

# first let's get overall sentiment by brand
from pyspark.sql.functions import desc

sentiment_by_brand = df_final.groupBy('brand', 'sentiment').count().sort(desc("brand")).toPandas()
sentiment_by_brand

# COMMAND ----------

import pandas as pd
sentiment_by_brand = sentiment_by_brand[sentiment_by_brand["brand"] != "general_makeup_channel"]

# getting totals
sentiment_by_brand_totals = sentiment_by_brand.groupby("brand").sum("count").reset_index()

# joining
sentiment_by_brand = pd.merge(sentiment_by_brand, sentiment_by_brand_totals, how ='left', on ='brand')

sentiment_by_brand = sentiment_by_brand.rename(columns={'count_x': 'value', 'count_y': 'total_count'})

sentiment_by_brand["ratio"] = round((sentiment_by_brand["value"] / sentiment_by_brand["total_count"]), 3)

sentiment_by_brand.head()

# COMMAND ----------

import matplotlib.pylab as plt
from pandas.plotting import table

plt.rcParams.update({'font.size': 30})

sentiment_by_brand.style.set_properties(**{'font-size': '20pt'})

# set fig size
fig, ax = plt.subplots(figsize=(12, 20)) 
# no axes
ax.xaxis.set_visible(False)  
ax.yaxis.set_visible(False)  
# no frame
ax.set_frame_on(False)  
# plot table
tab = table(ax, sentiment_by_brand, loc='upper right')  
# set font manually
tab.auto_set_font_size(False)
tab.set_fontsize(8) 

#plt.show()

# save the result
plt.savefig('/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/sentiment_by_brand.png')

# COMMAND ----------

## save the csv file in the csv dir
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/", "sentiment_by_brand.csv")
sentiment_by_brand.to_csv(fpath)

# COMMAND ----------

# now let's create some dummy variables based on what products the posts contain
# including general products

products = ["lipstick", "eyeshadow", "blush", "bronzer", "mascara", "foundation", "concealer", "gloss", "eyeliner", "gel", "powder", "moisturizer", "sunscreen", "shampoo", "conditioner", \
           "hairspray", "palette", "perfume", "brush", "oil", "serum", "cream", "soap", "cleanser", "makeup", "hair", "skincare", "mask", "sponge"]

# COMMAND ----------

# creating some dummy variables for analysis
df_final_dummy = df_final

for product in products:
    df_final_dummy = df_final_dummy.withColumn(product, (F.lower(df_final_dummy.text).rlike(product)))

# COMMAND ----------

df_final_dummy.show(5)

# COMMAND ----------

# 1116779
df_final_dummy.count()

# COMMAND ----------

# saving intermediate dataset
# saving to a parquet
df_final_dummy.write.mode("overwrite").format('parquet').save("/FileStore/CG_intermediate_data/competitor_analysis")

# COMMAND ----------

# loading data
df_viz = spark.read.parquet("dbfs:/FileStore/CG_intermediate_data/competitor_analysis")
#df_viz.show(10)

# COMMAND ----------

from pyspark.sql.functions import desc
import pyspark.sql.functions as F

sephora = df_viz.filter(F.col("brand") == "sephora")

lipstick = sephora.groupBy("lipstick", "sentiment").count().sort(desc("count"))
lipstick = lipstick.filter(F.col("lipstick") == True)
lipstick = lipstick.select("sentiment", "count").withColumnRenamed("count","lipstick").toPandas()

eyeshadow = sephora.groupBy("eyeshadow", "sentiment").count().sort(desc("count"))
eyeshadow = eyeshadow.filter(F.col("eyeshadow") == True)
eyeshadow = eyeshadow.select("sentiment", "count").withColumnRenamed("count","eyeshadow").toPandas()

blush = sephora.groupBy("blush", "sentiment").count().sort(desc("count"))
blush = blush.filter(F.col("blush") == True)
blush = blush.select("sentiment", "count").withColumnRenamed("count","blush").toPandas()

bronzer = sephora.groupBy("bronzer", "sentiment").count().sort(desc("count"))
bronzer = bronzer.filter(F.col("bronzer") == True)
bronzer = bronzer.select("sentiment", "count").withColumnRenamed("count","bronzer").toPandas()

mascara = sephora.groupBy("mascara", "sentiment").count().sort(desc("count"))
mascara = mascara.filter(F.col("mascara") == True)
mascara = mascara.select("sentiment", "count").withColumnRenamed("count","mascara").toPandas()

foundation = sephora.groupBy("foundation", "sentiment").count().sort(desc("count"))
foundation = foundation.filter(F.col("foundation") == True)
foundation = foundation.select("sentiment", "count").withColumnRenamed("count","foundation").toPandas()

concealer = sephora.groupBy("concealer", "sentiment").count().sort(desc("count"))
concealer = concealer.filter(F.col("concealer") == True)
concealer = concealer.select("sentiment", "count").withColumnRenamed("count","concealer").toPandas()

gloss = sephora.groupBy("gloss", "sentiment").count().sort(desc("count"))
gloss = gloss.filter(F.col("gloss") == True)
gloss = gloss.select("sentiment", "count").withColumnRenamed("count","gloss").toPandas()

eyeliner = sephora.groupBy("eyeliner", "sentiment").count().sort(desc("count"))
eyeliner = eyeliner.filter(F.col("eyeliner") == True)
eyeliner = eyeliner.select("sentiment", "count").withColumnRenamed("count","eyeliner").toPandas()

gel = sephora.groupBy("gel", "sentiment").count().sort(desc("count"))
gel = gel.filter(F.col("gel") == True)
gel = gel.select("sentiment", "count").withColumnRenamed("count","gel").toPandas()

powder = sephora.groupBy("powder", "sentiment").count().sort(desc("count"))
powder = powder.filter(F.col("powder") == True)
powder = powder.select("sentiment", "count").withColumnRenamed("count","powder").toPandas()

moisturizer = sephora.groupBy("moisturizer", "sentiment").count().sort(desc("count"))
moisturizer = moisturizer.filter(F.col("moisturizer") == True)
moisturizer = moisturizer.select("sentiment", "count").withColumnRenamed("count","moisturizer").toPandas()

sunscreen = sephora.groupBy("sunscreen", "sentiment").count().sort(desc("count"))
sunscreen = sunscreen.filter(F.col("sunscreen") == True)
sunscreen = sunscreen.select("sentiment", "count").withColumnRenamed("count","sunscreen").toPandas()

shampoo = sephora.groupBy("shampoo", "sentiment").count().sort(desc("count"))
shampoo = shampoo.filter(F.col("shampoo") == True)
shampoo = shampoo.select("sentiment", "count").withColumnRenamed("count","shampoo").toPandas()

conditioner = sephora.groupBy("conditioner", "sentiment").count().sort(desc("count"))
conditioner = conditioner.filter(F.col("conditioner") == True)
conditioner = conditioner.select("sentiment", "count").withColumnRenamed("count","conditioner").toPandas()

hairspray = sephora.groupBy("hairspray", "sentiment").count().sort(desc("count"))
hairspray = hairspray.filter(F.col("hairspray") == True)
hairspray = hairspray.select("sentiment", "count").withColumnRenamed("count","hairspray").toPandas()

palette = sephora.groupBy("palette", "sentiment").count().sort(desc("count"))
palette = palette.filter(F.col("palette") == True)
palette = palette.select("sentiment", "count").withColumnRenamed("count","palette").toPandas()

perfume = sephora.groupBy("perfume", "sentiment").count().sort(desc("count"))
perfume = perfume.filter(F.col("perfume") == True)
perfume = perfume.select("sentiment", "count").withColumnRenamed("count","perfume").toPandas()

brush = sephora.groupBy("brush", "sentiment").count().sort(desc("count"))
brush = brush.filter(F.col("brush") == True)
brush = brush.select("sentiment", "count").withColumnRenamed("count","brush").toPandas()

oil = sephora.groupBy("oil", "sentiment").count().sort(desc("count"))
oil = oil.filter(F.col("oil") == True)
oil = oil.select("sentiment", "count").withColumnRenamed("count","oil").toPandas()

serum = sephora.groupBy("serum", "sentiment").count().sort(desc("count"))
serum = serum.filter(F.col("serum") == True)
serum = serum.select("sentiment", "count").withColumnRenamed("count","serum").toPandas()

cleanser = sephora.groupBy("cleanser", "sentiment").count().sort(desc("count"))
cleanser = cleanser.filter(F.col("cleanser") == True)
cleanser = cleanser.select("sentiment", "count").withColumnRenamed("count","cleanser").toPandas()

soap = sephora.groupBy("soap", "sentiment").count().sort(desc("count"))
soap = soap.filter(F.col("soap") == True)
soap = soap.select("sentiment", "count").withColumnRenamed("count","soap").toPandas()

skincare = sephora.groupBy("skincare", "sentiment").count().sort(desc("count"))
skincare = skincare.filter(F.col("skincare") == True)
skincare = skincare.select("sentiment", "count").withColumnRenamed("count","skincare").toPandas()

sponge = sephora.groupBy("sponge", "sentiment").count().sort(desc("count"))
sponge = sponge.filter(F.col("sponge") == True)
sponge = sponge.select("sentiment", "count").withColumnRenamed("count","sponge").toPandas()


# COMMAND ----------

import pandas as pd

products2 = ["lipstick", "eyeshadow", "blush", "bronzer", "mascara", "foundation", "concealer", "gloss", "eyeliner", "gel", "powder", "moisturizer", "sunscreen", "shampoo", "conditioner", \
           "hairspray", "palette", "perfume", "brush", "oil", "serum", "soap", "cleanser", "skincare", "sponge"]

sephora_viz = pd.merge(lipstick, eyeshadow, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, blush, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, bronzer, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, mascara, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, foundation, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, concealer, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, gloss, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, eyeliner, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, gel, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, powder, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, moisturizer, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, sunscreen, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, shampoo, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, conditioner, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, hairspray, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, palette, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, perfume, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, brush, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, oil, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, serum, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, soap, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, cleanser, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, skincare, how ='outer', on ='sentiment')
sephora_viz = pd.merge(sephora_viz, sponge, how ='outer', on ='sentiment')


sephora_viz = pd.melt(sephora_viz, id_vars =['sentiment'], value_vars = products2)

sephora_viz.head()

# COMMAND ----------

# getting totals
sephora_totals = sephora_viz.groupby("variable").sum("value").reset_index()

# joining
sephora_viz = pd.merge(sephora_viz, sephora_totals, how ='left', on ='variable')
sephora_viz = sephora_viz.rename(columns={'value_x': 'value', 'value_y': 'total_count'})

sephora_viz["ratio"] = sephora_viz["value"] / sephora_viz["total_count"]
sephora_viz.head()

# COMMAND ----------

## doing the same for Ulta
ulta = df_viz.filter(F.col("brand") == "ulta")

lipstick = ulta.groupBy("lipstick", "sentiment").count().sort(desc("count"))
lipstick = lipstick.filter(F.col("lipstick") == True)
lipstick = lipstick.select("sentiment", "count").withColumnRenamed("count","lipstick").toPandas()

eyeshadow = ulta.groupBy("eyeshadow", "sentiment").count().sort(desc("count"))
eyeshadow = eyeshadow.filter(F.col("eyeshadow") == True)
eyeshadow = eyeshadow.select("sentiment", "count").withColumnRenamed("count","eyeshadow").toPandas()

blush = ulta.groupBy("blush", "sentiment").count().sort(desc("count"))
blush = blush.filter(F.col("blush") == True)
blush = blush.select("sentiment", "count").withColumnRenamed("count","blush").toPandas()

bronzer = ulta.groupBy("bronzer", "sentiment").count().sort(desc("count"))
bronzer = bronzer.filter(F.col("bronzer") == True)
bronzer = bronzer.select("sentiment", "count").withColumnRenamed("count","bronzer").toPandas()

mascara = ulta.groupBy("mascara", "sentiment").count().sort(desc("count"))
mascara = mascara.filter(F.col("mascara") == True)
mascara = mascara.select("sentiment", "count").withColumnRenamed("count","mascara").toPandas()

foundation = ulta.groupBy("foundation", "sentiment").count().sort(desc("count"))
foundation = foundation.filter(F.col("foundation") == True)
foundation = foundation.select("sentiment", "count").withColumnRenamed("count","foundation").toPandas()

concealer = ulta.groupBy("concealer", "sentiment").count().sort(desc("count"))
concealer = concealer.filter(F.col("concealer") == True)
concealer = concealer.select("sentiment", "count").withColumnRenamed("count","concealer").toPandas()

gloss = ulta.groupBy("gloss", "sentiment").count().sort(desc("count"))
gloss = gloss.filter(F.col("gloss") == True)
gloss = gloss.select("sentiment", "count").withColumnRenamed("count","gloss").toPandas()

eyeliner = ulta.groupBy("eyeliner", "sentiment").count().sort(desc("count"))
eyeliner = eyeliner.filter(F.col("eyeliner") == True)
eyeliner = eyeliner.select("sentiment", "count").withColumnRenamed("count","eyeliner").toPandas()

gel = ulta.groupBy("gel", "sentiment").count().sort(desc("count"))
gel = gel.filter(F.col("gel") == True)
gel = gel.select("sentiment", "count").withColumnRenamed("count","gel").toPandas()

powder = ulta.groupBy("powder", "sentiment").count().sort(desc("count"))
powder = powder.filter(F.col("powder") == True)
powder = powder.select("sentiment", "count").withColumnRenamed("count","powder").toPandas()

moisturizer = ulta.groupBy("moisturizer", "sentiment").count().sort(desc("count"))
moisturizer = moisturizer.filter(F.col("moisturizer") == True)
moisturizer = moisturizer.select("sentiment", "count").withColumnRenamed("count","moisturizer").toPandas()

sunscreen = ulta.groupBy("sunscreen", "sentiment").count().sort(desc("count"))
sunscreen = sunscreen.filter(F.col("sunscreen") == True)
sunscreen = sunscreen.select("sentiment", "count").withColumnRenamed("count","sunscreen").toPandas()

shampoo = ulta.groupBy("shampoo", "sentiment").count().sort(desc("count"))
shampoo = shampoo.filter(F.col("shampoo") == True)
shampoo = shampoo.select("sentiment", "count").withColumnRenamed("count","shampoo").toPandas()

conditioner = ulta.groupBy("conditioner", "sentiment").count().sort(desc("count"))
conditioner = conditioner.filter(F.col("conditioner") == True)
conditioner = conditioner.select("sentiment", "count").withColumnRenamed("count","conditioner").toPandas()

hairspray = ulta.groupBy("hairspray", "sentiment").count().sort(desc("count"))
hairspray = hairspray.filter(F.col("hairspray") == True)
hairspray = hairspray.select("sentiment", "count").withColumnRenamed("count","hairspray").toPandas()

palette = ulta.groupBy("palette", "sentiment").count().sort(desc("count"))
palette = palette.filter(F.col("palette") == True)
palette = palette.select("sentiment", "count").withColumnRenamed("count","palette").toPandas()

perfume = ulta.groupBy("perfume", "sentiment").count().sort(desc("count"))
perfume = perfume.filter(F.col("perfume") == True)
perfume = perfume.select("sentiment", "count").withColumnRenamed("count","perfume").toPandas()

brush = ulta.groupBy("brush", "sentiment").count().sort(desc("count"))
brush = brush.filter(F.col("brush") == True)
brush = brush.select("sentiment", "count").withColumnRenamed("count","brush").toPandas()

oil = ulta.groupBy("oil", "sentiment").count().sort(desc("count"))
oil = oil.filter(F.col("oil") == True)
oil = oil.select("sentiment", "count").withColumnRenamed("count","oil").toPandas()

serum = ulta.groupBy("serum", "sentiment").count().sort(desc("count"))
serum = serum.filter(F.col("serum") == True)
serum = serum.select("sentiment", "count").withColumnRenamed("count","serum").toPandas()

cleanser = ulta.groupBy("cleanser", "sentiment").count().sort(desc("count"))
cleanser = cleanser.filter(F.col("cleanser") == True)
cleanser = cleanser.select("sentiment", "count").withColumnRenamed("count","cleanser").toPandas()

soap = ulta.groupBy("soap", "sentiment").count().sort(desc("count"))
soap = soap.filter(F.col("soap") == True)
soap = soap.select("sentiment", "count").withColumnRenamed("count","soap").toPandas()

skincare = ulta.groupBy("skincare", "sentiment").count().sort(desc("count"))
skincare = skincare.filter(F.col("skincare") == True)
skincare = skincare.select("sentiment", "count").withColumnRenamed("count","skincare").toPandas()

sponge = ulta.groupBy("sponge", "sentiment").count().sort(desc("count"))
sponge = sponge.filter(F.col("sponge") == True)
sponge = sponge.select("sentiment", "count").withColumnRenamed("count","sponge").toPandas()


# COMMAND ----------

ulta_viz = pd.merge(lipstick, eyeshadow, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, blush, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, bronzer, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, mascara, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, foundation, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, concealer, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, gloss, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, eyeliner, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, gel, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, powder, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, moisturizer, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, sunscreen, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, shampoo, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, conditioner, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, hairspray, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, palette, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, perfume, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, brush, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, oil, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, serum, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, soap, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, cleanser, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, skincare, how ='outer', on ='sentiment')
ulta_viz = pd.merge(ulta_viz, sponge, how ='outer', on ='sentiment')


ulta_viz = pd.melt(ulta_viz, id_vars =['sentiment'], value_vars = products2)

ulta_viz.head()

# COMMAND ----------

# getting totals
ulta_totals = ulta_viz.groupby("variable").sum("value").reset_index()

# joining
ulta_viz = pd.merge(ulta_viz, sephora_totals, how ='left', on ='variable')
ulta_viz = ulta_viz.rename(columns={'value_x': 'value', 'value_y': 'total_count'})

ulta_viz["ratio"] = ulta_viz["value"] / ulta_viz["total_count"]
ulta_viz.head()

# COMMAND ----------

## doing the same for fenty
fenty = df_viz.filter(F.col("brand") == "fenty")

lipstick = fenty.groupBy("lipstick", "sentiment").count().sort(desc("count"))
lipstick = lipstick.filter(F.col("lipstick") == True)
lipstick = lipstick.select("sentiment", "count").withColumnRenamed("count","lipstick").toPandas()

eyeshadow = fenty.groupBy("eyeshadow", "sentiment").count().sort(desc("count"))
eyeshadow = eyeshadow.filter(F.col("eyeshadow") == True)
eyeshadow = eyeshadow.select("sentiment", "count").withColumnRenamed("count","eyeshadow").toPandas()

blush = fenty.groupBy("blush", "sentiment").count().sort(desc("count"))
blush = blush.filter(F.col("blush") == True)
blush = blush.select("sentiment", "count").withColumnRenamed("count","blush").toPandas()

bronzer = fenty.groupBy("bronzer", "sentiment").count().sort(desc("count"))
bronzer = bronzer.filter(F.col("bronzer") == True)
bronzer = bronzer.select("sentiment", "count").withColumnRenamed("count","bronzer").toPandas()

mascara = fenty.groupBy("mascara", "sentiment").count().sort(desc("count"))
mascara = mascara.filter(F.col("mascara") == True)
mascara = mascara.select("sentiment", "count").withColumnRenamed("count","mascara").toPandas()

foundation = fenty.groupBy("foundation", "sentiment").count().sort(desc("count"))
foundation = foundation.filter(F.col("foundation") == True)
foundation = foundation.select("sentiment", "count").withColumnRenamed("count","foundation").toPandas()

concealer = fenty.groupBy("concealer", "sentiment").count().sort(desc("count"))
concealer = concealer.filter(F.col("concealer") == True)
concealer = concealer.select("sentiment", "count").withColumnRenamed("count","concealer").toPandas()

gloss = fenty.groupBy("gloss", "sentiment").count().sort(desc("count"))
gloss = gloss.filter(F.col("gloss") == True)
gloss = gloss.select("sentiment", "count").withColumnRenamed("count","gloss").toPandas()

eyeliner = fenty.groupBy("eyeliner", "sentiment").count().sort(desc("count"))
eyeliner = eyeliner.filter(F.col("eyeliner") == True)
eyeliner = eyeliner.select("sentiment", "count").withColumnRenamed("count","eyeliner").toPandas()

gel = fenty.groupBy("gel", "sentiment").count().sort(desc("count"))
gel = gel.filter(F.col("gel") == True)
gel = gel.select("sentiment", "count").withColumnRenamed("count","gel").toPandas()

powder = fenty.groupBy("powder", "sentiment").count().sort(desc("count"))
powder = powder.filter(F.col("powder") == True)
powder = powder.select("sentiment", "count").withColumnRenamed("count","powder").toPandas()

moisturizer = fenty.groupBy("moisturizer", "sentiment").count().sort(desc("count"))
moisturizer = moisturizer.filter(F.col("moisturizer") == True)
moisturizer = moisturizer.select("sentiment", "count").withColumnRenamed("count","moisturizer").toPandas()

sunscreen = fenty.groupBy("sunscreen", "sentiment").count().sort(desc("count"))
sunscreen = sunscreen.filter(F.col("sunscreen") == True)
sunscreen = sunscreen.select("sentiment", "count").withColumnRenamed("count","sunscreen").toPandas()

shampoo = fenty.groupBy("shampoo", "sentiment").count().sort(desc("count"))
shampoo = shampoo.filter(F.col("shampoo") == True)
shampoo = shampoo.select("sentiment", "count").withColumnRenamed("count","shampoo").toPandas()

conditioner = fenty.groupBy("conditioner", "sentiment").count().sort(desc("count"))
conditioner = conditioner.filter(F.col("conditioner") == True)
conditioner = conditioner.select("sentiment", "count").withColumnRenamed("count","conditioner").toPandas()

hairspray = fenty.groupBy("hairspray", "sentiment").count().sort(desc("count"))
hairspray = hairspray.filter(F.col("hairspray") == True)
hairspray = hairspray.select("sentiment", "count").withColumnRenamed("count","hairspray").toPandas()

palette = fenty.groupBy("palette", "sentiment").count().sort(desc("count"))
palette = palette.filter(F.col("palette") == True)
palette = palette.select("sentiment", "count").withColumnRenamed("count","palette").toPandas()

perfume = fenty.groupBy("perfume", "sentiment").count().sort(desc("count"))
perfume = perfume.filter(F.col("perfume") == True)
perfume = perfume.select("sentiment", "count").withColumnRenamed("count","perfume").toPandas()

brush = fenty.groupBy("brush", "sentiment").count().sort(desc("count"))
brush = brush.filter(F.col("brush") == True)
brush = brush.select("sentiment", "count").withColumnRenamed("count","brush").toPandas()

oil = fenty.groupBy("oil", "sentiment").count().sort(desc("count"))
oil = oil.filter(F.col("oil") == True)
oil = oil.select("sentiment", "count").withColumnRenamed("count","oil").toPandas()

serum = fenty.groupBy("serum", "sentiment").count().sort(desc("count"))
serum = serum.filter(F.col("serum") == True)
serum = serum.select("sentiment", "count").withColumnRenamed("count","serum").toPandas()

cleanser = fenty.groupBy("cleanser", "sentiment").count().sort(desc("count"))
cleanser = cleanser.filter(F.col("cleanser") == True)
cleanser = cleanser.select("sentiment", "count").withColumnRenamed("count","cleanser").toPandas()

soap = fenty.groupBy("soap", "sentiment").count().sort(desc("count"))
soap = soap.filter(F.col("soap") == True)
soap = soap.select("sentiment", "count").withColumnRenamed("count","soap").toPandas()

skincare = fenty.groupBy("skincare", "sentiment").count().sort(desc("count"))
skincare = skincare.filter(F.col("skincare") == True)
skincare = skincare.select("sentiment", "count").withColumnRenamed("count","skincare").toPandas()

sponge = fenty.groupBy("sponge", "sentiment").count().sort(desc("count"))
sponge = sponge.filter(F.col("sponge") == True)
sponge = sponge.select("sentiment", "count").withColumnRenamed("count","sponge").toPandas()


# COMMAND ----------

fenty_viz = pd.merge(lipstick, eyeshadow, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, blush, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, bronzer, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, mascara, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, foundation, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, concealer, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, gloss, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, eyeliner, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, gel, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, powder, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, moisturizer, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, sunscreen, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, shampoo, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, conditioner, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, hairspray, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, palette, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, perfume, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, brush, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, oil, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, serum, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, soap, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, cleanser, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, skincare, how ='outer', on ='sentiment')
fenty_viz = pd.merge(fenty_viz, sponge, how ='outer', on ='sentiment')


fenty_viz = pd.melt(fenty_viz, id_vars =['sentiment'], value_vars = products2)

fenty_viz.head()

# COMMAND ----------

# getting totals
fenty_totals = fenty_viz.groupby("variable").sum("value").reset_index()

# joining
fenty_viz = pd.merge(fenty_viz, sephora_totals, how ='left', on ='variable')
fenty_viz = fenty_viz.rename(columns={'value_x': 'value', 'value_y': 'total_count'})

fenty_viz["ratio"] = fenty_viz["value"] / fenty_viz["total_count"]
fenty_viz.head()

# COMMAND ----------

# joining for viz
fenty_viz["brand"] = "fenty"
ulta_viz["brand"] = "ulta"
sephora_viz["brand"] = "sephora"

final_viz = pd.concat([sephora_viz, ulta_viz, fenty_viz], axis=0)
final_viz.head()

# COMMAND ----------

## save the csv file in the csv dir
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/", "competitor_sentiment_viz.csv")
final_viz.to_csv(fpath)

# COMMAND ----------

import plotly.express as px
 
fig = px.bar(final_viz[final_viz["sentiment"] == "positive"], x="variable", y="ratio", 
             color="brand", barmode = 'group',
            title = "Competitor Positive Consumer Sentiment Ratio By Product",
            color_discrete_sequence=["#9EF478", "#EDF197", "#78BCF4"])

fig.update_layout(plot_bgcolor = "white",  xaxis_title="Product", yaxis_title="% Positive Activity Of Total", title_x=0.5,
                 xaxis={'categoryorder':'total descending'})
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/", "competitor_sentiment_viz.html")
fig.write_html(fpath)
fig.show()

# COMMAND ----------

# adding same plot
import matplotlib.pyplot as plt
import seaborn as sns

positive_viz = final_viz[final_viz["sentiment"] == "positive"]

clrs = ["#9EF478", "#EDF197", "#78BCF4"]

# plot data in grouped manner of bar type
sns.set(font_scale=3, rc={'figure.figsize':(20,10)})
sns.set_theme(style='white')
p1 = sns.barplot(data=positive_viz, x='variable', y='ratio', hue='brand', palette=clrs)
p1.set_title('Competitor Positive Consumer Sentiment Ratio By Product')
p1.set_ylabel('% Positive Activity Of Total')
p1 = plt.xticks(rotation=45)
