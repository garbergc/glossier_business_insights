# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Cleaning

# COMMAND ----------

pip install nltk

# COMMAND ----------

pip install wordcloud

# COMMAND ----------

## Read in the data 
glos_comm = spark.read.parquet("/FileStore/glossier/glossier_comments")

# COMMAND ----------

## Show the first five records of the glossier comments data frame
glos_comm.show(5)

# COMMAND ----------

## Get the shape of the glossier comments data frame 
(glos_comm.count(), len(glos_comm.columns))

# COMMAND ----------

## Print the names of the glossier comment data frame columns 
## Note: All columns will stay in this version of the data frame and will be filtered via downstream analytics / modeling work 
glos_comm.columns

# COMMAND ----------

## Remove all uneccessary columns from the data frame 
cols = ("author_cakeday","author_flair_css_class","author_flair_text","permalink","stickied","gilded","distinguished","can_gild","retrieved_on","edited")
glos_comm = glos_comm.drop(*cols)
glos_comm.show(5)

# COMMAND ----------

## View the data types 
glos_comm.dtypes

# COMMAND ----------

## Convert the columns to appropriate data type 
glos_comm = glos_comm.withColumn("created_utc",glos_comm.created_utc.cast('timestamp'))

# COMMAND ----------

## View the schema of the cleaned data frame
glos_comm.printSchema()


# COMMAND ----------

## Convert the data frame into a view to run SQL on 
glos_comm.createOrReplaceTempView("glos_comm_vw")

## Get a count of all the missing values 
missing_glos_comm = spark.sql("select \
                    sum(case when author is null then 1 else 0 end) as author_count, \
                    sum(case when body is null then 1 else 0 end) as body_count, \
                    sum(case when controversiality is null then 1 else 0 end) as controversiality_count, \
                    sum(case when created_utc is null then 1 else 0 end) as created_utc_count, \
                    sum(case when id is null then 1 else 0 end) as id_count, \
                    sum(case when is_submitter is null then 1 else 0 end) as is_submitter_count, \
                    sum(case when link_id is null then 1 else 0 end) as link_id_count, \
                    sum(case when parent_id is null then 1 else 0 end) as parent_id_count, \
                    sum(case when score is null then 1 else 0 end) as score_count, \
                    sum(case when subreddit is null then 1 else 0 end) as subreddit_count, \
                    sum(case when subreddit_id is null then 1 else 0 end) as subreddit_id_count \
                    from glos_comm_vw")

## Show and view the results 
missing_glos_comm.show()

# COMMAND ----------

## Save the final data frame 
glos_comm_final = glos_comm
glos_comm_final.show(10)

# COMMAND ----------

## Read in the data 
glos_sub = spark.read.parquet("/FileStore/glossier/glossier_submissions")

# COMMAND ----------

## Show the first five records of the glossier submissions data frame
glos_sub.show(5)

# COMMAND ----------

## Get the shape of the glossier submissions data frame 
(glos_sub.count(), len(glos_sub.columns))

# COMMAND ----------

## Print the names of the glossier submissions data frame columns 
glos_sub.columns

# COMMAND ----------

## Drop all of the uneccessary columns
cols = ("whitelist_status","url","thumbnail_width","thumbnail_height","thumbnail","third_party_tracking_2","third_party_tracking","third_party_trackers","suggested_sort",
       "secure_media_embed", "retrieved_on", "promoted_url", "parent_whitelist_status", "link_flair_text", "link_flair_css_class", "imp_pixel", "href_url", "gilded", "embed_url", 
       "author_flair_css_class", "author_cakeday","adserver_imp_pixel", "adserver_click_url", "secure_media_embed", "secure_media", "post_hint", "permalink", "original_link", 
       "mobile_ad_url", "embed_type", "domain_override", "domain", "author", "preview", "author_flair_text", "edited", "crosspost_parent_list", "media", "media_embed")
glos_sub = glos_sub.drop(*cols)
glos_sub.columns

# COMMAND ----------

## View the data types of each column 
glos_sub.dtypes

# COMMAND ----------

## Convert the columns to appropriate data type 
glos_sub_final = glos_sub.withColumn("created_utc",glos_sub.created_utc.cast('timestamp'))

# COMMAND ----------

top_prods = glos_comm_final.select("body")
new = glos_sub_final.select("title").alias("body")

top_prods = top_prods.union(new)
## Not working with any null values
from pyspark.sql.functions import col,isnan, when, count
top_prods.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in top_prods.columns]
   ).show()

# COMMAND ----------

top_prods.head(2)

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import udf, col, lower, regexp_replace, translate

## converting all words to lowercase
top_prods = top_prods.withColumn("body",lower(translate('body', '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~', ' ')))


## defining tokenizer
tokenizer = Tokenizer(outputCol="words")
tokenizer.setInputCol("body")

## applying tokenizer to dataframe column
df_words_token = tokenizer.transform(top_prods) #.head()

## want to remove common words that tell us nothing about the text
## defining stopwordsremover and applying
remover = StopWordsRemover(inputCol='words', outputCol='words_clean')
df_words_no_stopw = remover.transform(df_words_token)

## creating a custom list of stopwords becuase there are top words that do not tell us anything important
stopwordList = ["•","sure","also", "thing", "glossier", "it", "one", "", "please", "get", "it's", "i'm", "think", "im", "make","much", "20"]
remover1 = StopWordsRemover(inputCol="words_clean", outputCol="words_cleaned" ,stopWords=stopwordList)
df_words_no_stopw = remover1.transform(df_words_no_stopw)


# COMMAND ----------

df_words_no_stopw.head(5)

# COMMAND ----------

#pip install nltk

# COMMAND ----------

from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import StringType, ArrayType

## nltk stemmer creates unity between words. "likes", "liked", "likely", and "liking" would all be counted as their stem, "like"
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df_words_no_stopw.withColumn("words", stemmer_udf("words_cleaned"))

df_stemmed.head(3)


# COMMAND ----------

import pyspark.sql.functions as f
result = df_stemmed.withColumn('indword', f.explode(f.col('words_cleaned'))) \
  .groupBy('indword') \
  .count().sort('count', ascending=False) \

print('############ TOP20 Most used words in Glossier subreddit are:')
result.show()


# COMMAND ----------

result.count()

# COMMAND ----------




# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

pandas_results = result.toPandas()

glossier_mask = np.array(Image.open('/Workspace/Repos/cag199@georgetown.edu/fall-2022-project-eda-adb-project-group-16/glossier.jpg'))

d = {}
for a, x in pandas_results.values:
    d[a] = x
wordcloud = WordCloud(background_color="white", colormap='RdPu', mask=glossier_mask)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plot_fpath = os.path.join(PLOT_DIR, 'wordcloud.png')
plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------

import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

fpath = os.path.join(CSV_DIR, "wordcloud.csv")
pandas_results.to_csv(fpath)

#fig.show()

# COMMAND ----------


import pyspark.sql.functions as F

searchfor = ['bdc','lashslick','wowder','concealer','mascara','highlighter','candle','rollerball','balm dot com','balm dotcom','hoodie','tote','sweatshirt','boybrow','brow gel','generation g','scarf','clip','clips','hair clips','perfume','cloud paint','cloudpaint','blush','eyeliner','no.1 pencil','no. 1 pencil','skin tint','stretch concealer', 'pomade','lipstick','pisces','lash slick','brush','eyeshadow','pro tip','protip','solar paint','bronzer','solarpaint','monochromes', 'milky oil','brow flick','browflick','haloscope','lipgloss','lip gloss','lidstar','ls','swiss miss','le','sticker','comb','kit','body hero','futuredew','milk jelly cleanser','cleanser','jelly cleanser','after baume','after balm','moisturizer','priming moisturizer','retinol','pro-retinol','universal pro-retinol','sunscreen','zit','zit sticker','zitsticker','tin', 'tins', 'cranberry', 'cordial','olivia rodrigo'
]
check_udf = F.udf(lambda x: x if x in searchfor else 'Not_present')

df = top_prods.withColumn('check_presence', check_udf(F.col('body')))
df = df.filter(df.check_presence != 'Not_present').drop('check_presence')

# COMMAND ----------

df.head(5)

# COMMAND ----------

from pyspark.sql.functions import *
newdf = df.withColumn('body', regexp_replace('body', 'bdc', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'balm dot com', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'balm dotcom', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'balm', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'swiss miss', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cloudpaint', 'blush'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cloud paint', 'blush'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'paint', 'blush'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pisces', 'lipstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'ls', 'mascara'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lash slick', 'mascara'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lashslick', 'mascara'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'slick', 'mascara'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'lash', 'mascara'))
newdf = newdf.withColumn('body', regexp_replace('body', 'browflick', 'brow pencil'))
newdf = newdf.withColumn('body', regexp_replace('body', 'brow flick', 'brow pencil'))
newdf = newdf.withColumn('body', regexp_replace('body', 'flick', 'brow pencil'))
newdf = newdf.withColumn('body', regexp_replace('body', 'wowder', 'face powder'))
newdf = newdf.withColumn('body', regexp_replace('body', 'rollerball', 'perfume'))
newdf = newdf.withColumn('body', regexp_replace('body', 'boy brow', 'brow gel'))
newdf = newdf.withColumn('body', regexp_replace('body', 'boybrow', 'brow gel'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'boy', 'brow gel'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pomade', 'brow gel'))
newdf = newdf.withColumn('body', regexp_replace('body', 'generation g', 'lipstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'gen g', 'lipstick'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'g', 'lipstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pisces', 'lipstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'clips', 'merch'))
newdf = newdf.withColumn('body', regexp_replace('body', 'clip', 'merch'))
newdf = newdf.withColumn('body', regexp_replace('body', 'hoodie', 'merch'))
newdf = newdf.withColumn('body', regexp_replace('body', 'sweatshirt', 'merch'))
newdf = newdf.withColumn('body', regexp_replace('body', 'tote', 'merch'))
newdf = newdf.withColumn('body', regexp_replace('body', 'no.1 pencil', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'no. 1 pencil', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pencil', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'skin tint', 'foundation'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'tint', 'foundation'))
newdf = newdf.withColumn('body', regexp_replace('body', 'stretch concealer', 'concealer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'stretchconcealer', 'concealer'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'stretch', 'concealer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'protip', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'solarpaint', 'bronzer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'solar paint', 'bronzer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pro tip', 'eyeliner'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'pro', 'eyeliner'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'tip', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'monochromes', 'eyeshadow'))
newdf = newdf.withColumn('body', regexp_replace('body', 'milky oil', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'oil', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'serum', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'haloscope', 'highlighter'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lip gloss', 'lipgloss'))
#newdf = newdf.withColumn('body', regexp_replace('body', 'gloss', 'lipgloss'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lidstar', 'eyeshadow'))
newdf = newdf.withColumn('body', regexp_replace('body', 'futuredew', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'milk jelly cleanser', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cleanser', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'jelly', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'zit sticker', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'zitsticker', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'zit', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'after baume', 'sunscreen'))
newdf = newdf.withColumn('body', regexp_replace('body', 'after balm', 'sunscreen'))
newdf = newdf.withColumn('body', regexp_replace('body', "r'\ble\b", 'limited edition products'))
newdf = newdf.withColumn('body', regexp_replace('body', "r'\ble", 'limited edition products'))
newdf = newdf.withColumn('body', regexp_replace('body', "le\b", 'limited edition products'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cranberry', 'limited edition products'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cordial', 'limited edition products'))
newdf = newdf.withColumn('body', regexp_replace('body', 'olivia rodrigo', 'limited edition products'))
newdf = newdf.withColumn('body', regexp_replace('body', 'priming moisturize', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'moisturizer', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pro-retinol', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'retinol', 'skincare'))


# COMMAND ----------

import pyspark.sql.functions as f
top5_product_count = newdf.withColumn('product', f.col('body')) \
  .groupBy('product') \
  .count().sort('count', ascending=False) \
  .limit(5) \


print('############ TOP 5 Most used mentioned products in the Glossier subreddit are:')
top5_product_count.show()

# COMMAND ----------

from pyspark.sql.functions import sum

product_count = newdf.withColumn('product', f.col('body')) \
  .groupBy('product') \
  .count().sort('count', ascending=False) \


product_count.groupBy().sum().show()

# COMMAND ----------

from pyspark.sql.types import StructType,StructField, StringType, IntegerType
other_data = [("other", 39)]
schema = StructType([ \
    StructField("product",StringType(),True), \
    StructField("count", IntegerType(), True) \
  ])
other_products = spark.createDataFrame(data=other_data,schema=schema)

# COMMAND ----------

piechart_df = top5_product_count.union(other_products)

# COMMAND ----------

piechart_df.show()

# COMMAND ----------

pandas_product_count = piechart_df.toPandas()
#pandas_product_count.groupby(['product']).sum().plot(kind='pie', y='count')
fpath = os.path.join(CSV_DIR, "top_products.csv")
pandas_product_count.to_csv(fpath)

# COMMAND ----------

#labels = 'blush', 'skincare', 'foundation', 'bronzer', 'tinted chapstick', 'other'
#plot = pandas_product_count.plot.pie(x='product',y='count',label=labels, title="Top Mentioned Glossier Products in #Glossier Subreddit 2021-2022", legend=False, ylabel = ' ', \
#                   autopct='%1.1f%%', \
#                   shadow=True, startangle=0, fontsize =10, colormap = "RdPu" )

# COMMAND ----------

labels = 'blush', 'skincare', 'foundation', 'bronzer', 'tinted chapstick', 'other'
sizes = pandas_product_count["count"]
colors = ['#FBE1E1','#ba97aa','#FFF5FC','#FBD2D7','#E4A199', '#F2F2F2']
          #'#F2F2F2','#FBE1E1','#FFF5FC','#E4A199','#CE897B','#F2F2F2','#FBE1E2','#FBE1E1']
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors=colors,
        shadow=True, startangle=90)
ax1.title.set_text('Top Mentioned Glossier Products in Glossier Subreddit 2021-2022')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#title=
plot_fpath = os.path.join(PLOT_DIR, 'piechart.png')
plt.savefig(plot_fpath)
plt.show()
#plt.imshow(fig1, cmap='RdBu')

# COMMAND ----------

## ALL GARBAGE BELOW

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

#sizes = pd.DataFrame([80,10,5,4,0.1,0.9],index=list("ABCDEF"))

fig1, ax = plt.subplots()

def autopct_more_than_1(pct):
    return ('%1.f%%' % pct) if pct > 1 else ''

p,t,a = ax.pie(pandas_product_count.count, autopct=autopct_more_than_1)
ax.axis('equal') 

# normalize dataframe (not actually needed here, but for general case)
normsizes = pandas_product_count/pandas_product_count.sum()*100
# create handles and labels for legend, take only those where value is > 1
h,l = zip(*[(h,lab) for h,lab,i in zip(p,pandas_product_count.product,normsizes.values) if i > 1])

ax.legend(h, l,loc="best", bbox_to_anchor=(1,1))

plt.show()

# COMMAND ----------

plot = pandas_product_count.plot.pie(x='product', y='count', title="Top Mentioned Glossier Products in Glossier Subreddit 2021-2022", legend=True, \
                   autopct='%1.1f%%', #explode=(0, 0, 0.1), \
                   shadow=True, startangle=0)

# COMMAND ----------

### PLEASE LEAVE ALL CELLS BELOW OUT OF COMBINED NOTEBOOK

# COMMAND ----------

import pyspark.sql.functions as f
searchfor = ['bdc','lashslick','wowder','concealer','mascara','highlighter','candle','rollerball','balm dot com','balm dotcom','hoodie','tote','sweatshirt','boybrow','brow gel','generation g','scarf','clip','clips','hair clips','perfume','cloud paint','cloudpaint','blush','eyeliner','no.1 pencil','no. 1 pencil','skin tint','stretch concealer', 'pomade','lipstick','pisces','lash slick','brush','eyeshadow','pro tip','protip','solar paint','bronzer','solarpaint','monochromes', 'milky oil','brow flick','browflick','haloscope','lipgloss','lip gloss','lidstar','ls','swiss miss','le','sticker','comb','kit','body hero','futuredew','milk jelly cleanser','cleanser','jelly cleanser','after baume','after balm','moisturizer','priming moisturizer','retinol','pro-retinol','universal pro-retinol','sunscreen','zit','zit sticker','zitsticker','tin', 'tins', 'cranberry', 'cordial','olivia rodrigo', 'rodrigo', 'balm', 'pencil', 'swiss' ,'miss', 'hair', 'pro', 'tip', 'milky' , 'flick', 'boy', 'olivia', 'dot', 'com', 'cloud', 'paint', 'lash', 'slick', 'stretch', 'hero', 'solar', 'serum', 'gel', 'tint', 'generation', 'g', 'stretch', 'pro', 'tip', 'gloss'
]
result.filter(result.indword.isin(searchfor)).show()

# COMMAND ----------

only_products = result.filter(result.indword.isin(searchfor))

# COMMAND ----------

#df.head(35)

# COMMAND ----------

from pyspark.sql.functions import *
newdf = only_products.withColumn('indword', regexp_replace('indword', 'bdc', 'tinted chapstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'balm', 'tinted chapstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'swiss', 'tinted chapstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'miss', 'tinted chapstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'cloud', 'blush'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'paint', 'blush'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'cloudpaint', 'blush'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'pisces', 'lipstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'ls', 'mascara'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'lashslick', 'mascara'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'slick', 'mascara'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'lash', 'mascara'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'flick', 'brow pencil'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'browflick', 'brow pencil'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'wowder', 'face powder'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'rollerball', 'perfume'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'boy brow', 'brow gel'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'boy', 'brow gel'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'boybrow', 'brow gel'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'pomade', 'brow gel'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'generation g', 'lipstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'generation', 'lipstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'g', 'lipstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'pisces', 'lipstick'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'clips', 'hair clips'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'clip', 'hair clips'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'no.1 pencil', 'eyeliner'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'no. 1 pencil', 'eyeliner'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'pencil', 'eyeliner'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'skin tint', 'foundation'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'tint', 'foundation'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'stretch concealer', 'concealer'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'stretch', 'concealer'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'protip', 'eyeliner'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'solar', 'bronzer'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'solarpaint', 'bronzer'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'pro tip', 'eyeliner'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'pro', 'eyeliner'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'tip', 'eyeliner'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'monochromes', 'eyeshadow'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'milky', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'oil', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'serum', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'haloscope', 'highlighter'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'lip gloss', 'lipgloss'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'gloss', 'lipgloss'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'lidstar', 'eyeshadow'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'futuredew', 'skincare'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'milk jelly cleanser', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'cleanser', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'jelly', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'zit', 'skincare'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'zit sticker', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'zitsticker', 'skincare'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'after baume', 'sunscreen'))
#newdf = newdf.withColumn('indword', regexp_replace('indword', 'after balm', 'sunscreen'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'le', 'limited edition products'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'cranberry', 'limited edition products'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'cordial', 'limited edition products'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'olivia', 'limited edition products'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'rodrigo', 'limited edition products'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'moisturizer', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'priming moisturize', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'retinol', 'skincare'))
newdf = newdf.withColumn('indword', regexp_replace('indword', 'pro-retinol', 'skincare'))


# COMMAND ----------

newdf.head(3)

# COMMAND ----------

import pyspark.sql.functions as f
product_count = newdf.withColumn('product', f.col('indword')) \
  .groupBy('product') \
  .count().sort('count', ascending=False) \

print('############ TOP20 Most used mentioned products in the Glossier subreddit are:')
product_count.show()

# COMMAND ----------

pandas_product_count = product_count.toPandas()

#fig1, ax1 = plt.subplots()
#ax1.pie(pandas_product_count.count, explode=explode, labels=labels, autopct='%1.1f%%',#
 #       shadow=True, startangle=90)
#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#plt.show()

# COMMAND ----------

#pandas_product_count = product_count.toPandas()
#plot = pandas_product_count.plot.pie(y='count', figsize=(5, 5))

# COMMAND ----------

pandas_product_count = product_count.toPandas()
pandas_product_count.groupby(['product']).sum().plot(kind='pie', y='count')
