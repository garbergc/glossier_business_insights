# Databricks notebook source
pip install spark.nlp

# COMMAND ----------

pip install nltk

# COMMAND ----------

pip install proj

# COMMAND ----------

pip install basemap

# COMMAND ----------

glos_comm = spark.read.parquet("/FileStore/glossier/glossier_comments")

cols = ("author_cakeday","author_flair_css_class","author_flair_text","permalink","stickied","gilded","distinguished","can_gild","retrieved_on","edited")
glos_comm = glos_comm.drop(*cols)
glos_comm = glos_comm.withColumn("created_utc",glos_comm.created_utc.cast('timestamp'))

glos_comm.createOrReplaceTempView("glos_comm_vw")
glos_comm_final = glos_comm

glos_sub = spark.read.parquet("/FileStore/glossier/glossier_submissions")


cols = ("whitelist_status","url","thumbnail_width","thumbnail_height","thumbnail","third_party_tracking_2","third_party_tracking","third_party_trackers","suggested_sort",
       "secure_media_embed", "retrieved_on", "promoted_url", "parent_whitelist_status", "link_flair_text", "link_flair_css_class", "imp_pixel", "href_url", "gilded", "embed_url", 
       "author_flair_css_class", "author_cakeday","adserver_imp_pixel", "adserver_click_url", "secure_media_embed", "secure_media", "post_hint", "permalink", "original_link", 
       "mobile_ad_url", "embed_type", "domain_override", "domain", "author", "preview", "author_flair_text", "edited", "crosspost_parent_list", "media", "media_embed")
glos_sub = glos_sub.drop(*cols)

glos_sub_final = glos_sub.withColumn("created_utc",glos_sub.created_utc.cast('timestamp'))


# COMMAND ----------

top_prods = glos_comm_final.select("body")
new = glos_sub_final.select("title").alias("body")


# COMMAND ----------

top_prods = top_prods.union(new)

# COMMAND ----------

import pandas

twitter_loc = pandas.read_csv("/Workspace/Repos/cag199@georgetown.edu/fall-2022-project-eda-adb-project-group-16/data/twitter_data1.csv")
adnl_glossier = pandas.read_csv("/Workspace/Repos/cag199@georgetown.edu/fall-2022-project-eda-adb-project-group-16/data/glossier_storecomments_reddit.csv")

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()

twitterDF=spark.createDataFrame(twitter_loc) 
twitterDF.printSchema()
twitterDF.show()

#twitterrdd = sc.parallelize([twitter_loc])
#twitterdf = spark.createDataFrame(twitterrdd)

# COMMAND ----------

glossDF=spark.createDataFrame(adnl_glossier) 
glossDF.printSchema()
glossDF.show()

# COMMAND ----------

new_test = glossDF.union(twitterDF)

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import udf, col, lower, regexp_replace, translate
new_test.count()
new_test = new_test.withColumn("body",lower(translate('body', '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~', ' ')))
new_test.count()

# COMMAND ----------

import pandas as pd
##cities dataset via GeoLite2 city dataset, provided by Maxmind
cities_df = pd.read_csv("/Workspace/Repos/cag199@georgetown.edu/fall-2022-project-eda-adb-project-group-16/data/worldcities1.csv")
cities_df['city'] = cities_df['city'].str.lower()
cities_df1 = cities_df['city'].str.lower()
#cities_df = cities_df.city_name.unique()
print(cities_df1.head(10))

# COMMAND ----------

cities_df1 = cities_df1.tolist()
print(type(cities_df))

# COMMAND ----------

"oslo" in cities_df1

# COMMAND ----------

import pyspark.sql.functions as psf
new_test.createOrReplaceTempView("new_test_vw")
new_test.withColumn('cities',psf.when(lower(psf.col('body')).rlike('({})\d'.format('|'.join(cities_df1))), '1').otherwise('')).show()

# COMMAND ----------

from pyspark.sql.functions import array, lit
city_newdf = spark.sql(
  """ with t1 ( select body, array('stockholm','san francisco','seattle', 'london', 'brooklyn', 'oslo', 'boston', 'philadelphia','minneapolis', 'wolverhampton', 'kent', 'indianapolis', 'paris', 'miami', 'atlanta', 'new york','houston', 'los angeles','chicago','toronto', 'washington dc','charleston','bordeaux', 'austin','melbourne','madre linda','dublin','dartford','vancouver', 'dallas', 'san juan', 'copenhagen', 'portland','buenos aires','cape may','denver', 'selangor','ipoh','san diego','stirling','leeds', 'phoenix','tucson', 'gothenburg', 'st. louis','greenwich','suzhou', 'montreal', 'nashville','new orleans','savannah','pittsburgh','honolulu','hong kong','albuquerque', 'oxford','edinburgh', 'asbury park','charlotte', 'orlando', 'cincinnati','oakland','san jose','quebec', 'lake worth','midlands', 'canyon lake','salt lake city', 'baltimore') as a1 from new_test_vw),
     t2 (select body,  filter(a1, x -> body like x||'%') a1f from t1)
     select body, a1f as cities from t2
  """)

# COMMAND ----------

city_newdf = city_newdf.select(col("cities"))

# COMMAND ----------

from pyspark.sql.functions import explode
city_newdf = city_newdf.select(explode(city_newdf.cities))

# COMMAND ----------

import pyspark.sql.functions as f
newdf_count = city_newdf.withColumn('city', f.col('col')) \
  .groupBy('city') \
  .count().sort('count', ascending=False)
#newdf_count.show(30)

# COMMAND ----------

from pyspark.sql.window import Window
newdf_count1 = newdf_count\
  .withColumn('total', f.sum('count').over(Window.partitionBy()))\
  .withColumn('percent', (f.col('count')/f.col('total'))*100)

# COMMAND ----------

from pyspark.sql.functions import round, col
data = newdf_count1.select(col("city"),round(col("percent"),2))
#newdf = new_test.filter(new_test.body.isin(cities_df1))
#data.show()

# COMMAND ----------

 df = data.selectExpr("city", "`round(percent, 2)` as percent")
 df.show(5)

# COMMAND ----------

pandas_cities = newdf_count.toPandas()

# COMMAND ----------

cities_df['city']=cities_df['city'].astype(str)
pandas_cities['city']=pandas_cities['city'].astype(str)

# COMMAND ----------

fullnew = pandas_cities.merge(cities_df,how='left', on='city')

# COMMAND ----------

fullnew.head(4)

# COMMAND ----------

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
 
# Set the dimension of the figure
plt.rcParams["figure.figsize"]=100,40;

# Make the background map
m=Basemap(llcrnrlon=-160, llcrnrlat=-50, urcrnrlon=160, urcrnrlat=70)
m.drawmapboundary(fill_color='#edf2f5', linewidth=0)
m.fillcontinents(color='#B86B77', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")

# Add a point per position
scatter = m.scatter(
    x=fullnew['lng'], 
    y=fullnew['lat'], 
    s=fullnew['count']*1500, 
    alpha=0.4, 
    #c = fullnew['city'],
    c=np.random.rand(len(fullnew['city']),3), 
    #label = fullnew['city'],
    cmap="spring"
)
plt.title("Glossier Reddit and Twitter User Location Mentions", fontdict={'fontsize': 70})


for i in range(len(fullnew['city'])):
    plt.text(fullnew['lng'][i], fullnew['lat'][i], fullnew['city'][i],fontsize=10,fontfamily = 'serif',#fontweight='bold', (fullnew['count'][i]*5)
                    ha='left',va='center_baseline',color='k')
