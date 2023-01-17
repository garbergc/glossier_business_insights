# Databricks notebook source
# MAGIC %md
# MAGIC Business Goal 2: We will identify which products should be in the newest Glossier kit.

# COMMAND ----------

# MAGIC %md
# MAGIC Technical Proposal: We will scrape the Glossier website to identify its complete set of products. We will then use NLP techniques to identify which posts and comments in the Glossier subreddit contain these products. We will conduct sentiment analysis of each post to assign positive, negative, or neutral values. We will sum the number of posts and comments by product for positive posts. We will identify the top 10 products with the highest activity (positive sentiment). Similarly, we will sum the number of posts and comments by product for negative posts. We will identify the top 10 products with the highest activity (negative sentiment). We will display this information on a faceted chart to show executive audiences which products should be promoted, and which should potentially be discontinued.

# COMMAND ----------

pip install nltk

# COMMAND ----------

## Read in the data 
glos_comm = spark.read.parquet("/FileStore/glossier/glossier_comments")

# COMMAND ----------

## Remove all uneccessary columns from the data frame 
cols = ("author_cakeday","author_flair_css_class","author_flair_text","permalink","stickied","gilded","distinguished","can_gild","retrieved_on","edited")
glos_comm = glos_comm.drop(*cols)

# COMMAND ----------

## Convert the columns to appropriate data type 
glos_comm = glos_comm.withColumn("created_utc",glos_comm.created_utc.cast('timestamp'))

# COMMAND ----------

## Save the final data frame 
glos_comm_final = glos_comm

# COMMAND ----------

## Read in the data 
glos_sub = spark.read.parquet("/FileStore/glossier/glossier_submissions")

# COMMAND ----------

## Drop all of the uneccessary columns
cols = ("whitelist_status","url","thumbnail_width","thumbnail_height","thumbnail","third_party_tracking_2","third_party_tracking","third_party_trackers","suggested_sort",
       "secure_media_embed", "retrieved_on", "promoted_url", "parent_whitelist_status", "link_flair_text", "link_flair_css_class", "imp_pixel", "href_url", "gilded", "embed_url", 
       "author_flair_css_class", "author_cakeday","adserver_imp_pixel", "adserver_click_url", "secure_media_embed", "secure_media", "post_hint", "permalink", "original_link", 
       "mobile_ad_url", "embed_type", "domain_override", "domain", "author", "preview", "author_flair_text", "edited", "crosspost_parent_list", "media", "media_embed")
glos_sub = glos_sub.drop(*cols)

# COMMAND ----------

## Convert the columns to appropriate data type 
glos_sub_final = glos_sub.withColumn("created_utc",glos_sub.created_utc.cast('timestamp'))

# COMMAND ----------

## Get a combination of the submissions and comments data frame 
top_prods = glos_comm_final.select("body")
new = glos_sub_final.select("title").alias("body")
top_prods = top_prods.union(new)

# COMMAND ----------

## Website Used: https://github.com/maobedkova/TopicModelling_PySpark_SparkNLP/blob/master/Topic_Modelling_with_PySpark_and_Spark_NLP.ipynb
## Define a pipeline for text data cleaning 

## Assemble the document 
from sparknlp.base import DocumentAssembler
documentAssembler = DocumentAssembler() \
     .setInputCol("body") \
     .setOutputCol('document')

## 1. Tokenize the data 
from sparknlp.annotator import Tokenizer
tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

## 2. Normalize (or clean the data / remove characters)
from sparknlp.annotator import Normalizer
normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

## 3. Lemmatize the data 
from sparknlp.annotator import LemmatizerModel
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

## 4. Remove stopwords 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
eng_stopwords = stopwords.words('english')
from sparknlp.annotator import StopWordsCleaner
stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('unigrams') \
     .setStopWords(eng_stopwords)

## 5. Transform the data further with finisher
from sparknlp.base import Finisher
finisher = Finisher() \
     .setInputCols(['unigrams'])

## Develop the pipeline 
from pyspark.ml import Pipeline
pipeline = Pipeline() \
     .setStages([documentAssembler,                  
                 tokenizer,
                 normalizer,                  
                 lemmatizer,                  
                 stopwords_cleaner,  
                 finisher])

## Transform the text 
processed_review = pipeline.fit(top_prods).transform(top_prods)

## 6. Then, concatenate the data in order to run sentiment analysis 
from pyspark.sql.functions import col, concat_ws
df2 = processed_review.withColumn("finished_unigrams", concat_ws(" ",col("finished_unigrams")))
df2.show(5)

# COMMAND ----------

## Get the text distributions after cleaning
import pyspark.sql.functions as F
df_temp = processed_review.withColumn('text_dist', F.size('finished_unigrams'))
df_temp.describe("text_dist").show()

# COMMAND ----------

## Convert the data frame into a view to run SQL on 
df2.createOrReplaceTempView("df2_vw")

## Determine if a post or comment contains the respective product or not 
product_df = spark.sql("select body, finished_unigrams, \
                    case when lower(finished_unigrams) like '%bdc%' then 1 else 0 end as bdc, \
                    case when lower(finished_unigrams) like '%lashslick%' then 1 else 0 end as lashslick, \
                    case when lower(finished_unigrams) like '%powder%' then 1 else 0 end as powder, \
                    case when lower(finished_unigrams) like '%concealer%' then 1 else 0 end as concealer, \
                    case when lower(finished_unigrams) like '%mascara%' then 1 else 0 end as mascara,\
                    case when lower(finished_unigrams) like '%highlighter%' then 1 else 0 end as highlighter, \
                    case when lower(finished_unigrams) like '%candle%' then 1 else 0 end as candle, \
                    case when lower(finished_unigrams) like '%rollerball%' then 1 else 0 end as rollerball,\
                    case when lower(finished_unigrams) like '%balm_dot_com%' or lower(finished_unigrams) like '%balm_dotcom%' then 1 else 0 end as balm_dot_com, \
                    case when lower(finished_unigrams) like '%hoodie%' then 1 else 0 end as hoodie, \
                    case when lower(finished_unigrams) like '%tote%' then 1 else 0 end as tote, \
                    case when lower(finished_unigrams) like '%boybrow%' then 1 else 0 end as boybrow, \
                    case when lower(finished_unigrams) like '%brow gel%' then 1 else 0 end as brow_gel, \
                    case when lower(finished_unigrams) like '%generation g%' then 1 else 0 end as generation_g, \
                    case when lower(finished_unigrams) like '%scarf%' then 1 else 0 end as scarf, \
                    case when lower(finished_unigrams) like '%clip%' or lower(finished_unigrams) like '%clips%' then 1 else 0 end as clip, \
                    case when lower(finished_unigrams) like '%hair clips%' then 1 else 0 end as hair_clips, \
                    case when lower(finished_unigrams) like '%perfume%' then 1 else 0 end as perfume, \
                    case when lower(finished_unigrams) like '%cloud paint%' or lower(finished_unigrams) like '%cloudpaint%' then 1 else 0 end as cloud_paint, \
                    case when lower(finished_unigrams) like '%perfume%'  then 1 else 0 end as blush, \
                    case when lower(finished_unigrams) like '%eyeliner%' then 1 else 0 end as eyeliner, \
                    case when lower(finished_unigrams) like '%no. 1 pencil%' or lower(finished_unigrams) like '%no.1 pencil%' then 1 else 0 end as pencil, \
                    case when lower(finished_unigrams) like '%skin tint%' then 1 else 0 end as skin_tint, \
                    case when lower(finished_unigrams) like '%stretch concealer%' then 1 else 0 end as stretch_concealer, \
                    case when lower(finished_unigrams) like '%pomade%' then 1 else 0 end as pomade, \
                    case when lower(finished_unigrams) like '%lipstick%' then 1 else 0 end as lipstick, \
                    case when lower(finished_unigrams) like '%lash slick%' then 1 else 0 end as lash_slick, \
                    case when lower(finished_unigrams) like '%brush%' then 1 else 0 end as brush, \
                    case when lower(finished_unigrams) like '%eyeshadow%' then 1 else 0 end as eyeshadow, \
                    case when lower(finished_unigrams) like '%pro tip%' or lower(finished_unigrams) like '%protip%' then 1 else 0 end as protip, \
                    case when lower(finished_unigrams) like '%solar paint%' or lower(finished_unigrams) like '%solarpaint%' then 1 else 0 end as solar_paint, \
                    case when lower(finished_unigrams) like '%bronzer%' then 1 else 0 end as bronzer, \
                    case when lower(finished_unigrams) like '%monochromes%' then 1 else 0 end as monochromes, \
                    case when lower(finished_unigrams) like '%milky oil%' then 1 else 0 end as milky_oil, \
                    case when lower(finished_unigrams) like '%brow flick%' or lower(finished_unigrams) like '%browflick%' then 1 else 0 end as brow_flick, \
                    case when lower(finished_unigrams) like '%haloscope%' then 1 else 0 end as haloscope, \
                    case when lower(finished_unigrams) like '%lidstar%' or lower(finished_unigrams) like '%ls%' then 1 else 0 end as lidstar, \
                    case when lower(finished_unigrams) like '%swiss miss%' then 1 else 0 end as swiss_miss, \
                    case when lower(finished_unigrams) like '%le%' then 1 else 0 end as le, \
                    case when lower(finished_unigrams) like '%sticker%' then 1 else 0 end as sticker, \
                    case when lower(finished_unigrams) like '%comb%' then 1 else 0 end as comb, \
                    case when lower(finished_unigrams) like '%kit%' then 1 else 0 end as kit, \
                    case when lower(finished_unigrams) like '%body hero%' then 1 else 0 end as body_hero, \
                    case when lower(finished_unigrams) like '%futuredew%' then 1 else 0 end as futuredew, \
                    case when lower(finished_unigrams) like '%cleanser%' or lower(finished_unigrams) like '%milk jelly cleanser%' or lower(finished_unigrams) like '%jelly cleanser%' then 1 else 0 end as cleanser, \
                    case when lower(finished_unigrams) like '%after baume%' or lower(finished_unigrams) like '%after balm%' then 1 else 0 end as after_balm, \
                    case when lower(finished_unigrams) like '%moisturizer%' or lower(finished_unigrams) like '%priming moisturizer%' then 1 else 0  end as moisturizer, \
                    case when lower(finished_unigrams) like '%retinol%' or lower(finished_unigrams) like '%pro-retinol%' or lower(finished_unigrams) like '%universal pro-retinol%' then 1 else 0 end as retinol, \
                    case when lower(finished_unigrams) like '%sunscreen%' then 1 else 0 end as sunscreen, \
                    case when lower(finished_unigrams) like '%zit%' or lower(finished_unigrams) like '%zit sticker%' or lower(finished_unigrams) like '%zitsticker%' then 1 else 0 end as zit, \
                    case when lower(finished_unigrams) like '%tin%' or lower(finished_unigrams) like '%tins%' then 1 else 0 end as tin, \
                    case when lower(finished_unigrams) like '%cranberry%' then 1 else 0 end as cranberry, \
                    case when lower(finished_unigrams) like '%cordial%' then 1 else 0 end as cordial, \
                    case when lower(finished_unigrams) like '%olivia rodrigo%' then 1 else 0 end as olivia_rodrigo from df2_vw")

# COMMAND ----------

# Create an NLP pipeline
# Using the twitter sentiment pipeline because Reddit and Twitter are similar as two social media platforms
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

## Assemble the document 
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
## Use the pretrained encoder model 
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

## Use the pretraind twitter model 
sentimentdl = SentimentDLModel.pretrained(name='sentimentdl_use_twitter', lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

## Assemble and develop the pipeline 
nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl])

# COMMAND ----------

## Run the sentiment analysis pipeline 
from pyspark.sql.functions import col

## Create the dataframe 
empty_df = spark.createDataFrame([['']]).toDF("text")
pipelineModel = nlpPipeline.fit(empty_df)

## Run the pipeline on the data and select the clean text column 
data = product_df.select(col("finished_unigrams").alias("text"))
result = pipelineModel.transform(data)

# COMMAND ----------

## Only select the necessary columns using explode function 
result = result.select('text', F.explode('sentiment.result').alias("sentiment"))

# COMMAND ----------

## Convert the two dataframes into views to run SQL 
result.createOrReplaceTempView("result_vw")
product_df.createOrReplaceTempView("product_df_vw")

## Combine the two dataframes in order to get the product, sentiment, and text values 
combined_df = spark.sql("select product_df_vw.*, result_vw.sentiment \
                    from product_df_vw join result_vw on product_df_vw.finished_unigrams = result_vw.text")

## Show the first few records of the data frame 
combined_df.show(5)

# COMMAND ----------

## Get the columns as a list of products to convert to long format
combined_df.columns

# COMMAND ----------

## Import the necessary libraries 
from pyspark.sql.functions import array, col, explode, lit, struct
from pyspark.sql import DataFrame
from typing import Iterable

## Define a function to melt the dataframe 
## Website Used: https://stackoverflow.com/questions/41670103/how-to-melt-spark-dataframe
def melt(df: DataFrame, 
        id_vars: Iterable[str], value_vars: Iterable[str], 
        var_name: str="variable", value_name: str="value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name)) 
        for c in value_vars))
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))
    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)

## Call the function to convert the dataframe into long format using the column names above
long_df = melt(combined_df, id_vars=['body', 'finished_unigrams', 'sentiment'], value_vars=['bdc',
 'lashslick',
 'powder',
 'concealer',
 'mascara',
 'highlighter',
 'candle',
 'rollerball',
 'balm_dot_com',
 'hoodie',
 'tote',
 'boybrow',
 'brow_gel',
 'generation_g',
 'scarf',
 'clip',
 'hair_clips',
 'perfume',
 'cloud_paint',
 'blush',
 'eyeliner',
 'pencil',
 'skin_tint',
 'stretch_concealer',
 'pomade',
 'lipstick',
 'lash_slick',
 'brush',
 'eyeshadow',
 'protip',
 'solar_paint',
 'bronzer',
 'monochromes',
 'milky_oil',
 'brow_flick',
 'haloscope',
 'lidstar',
 'swiss_miss',
 'le',
 'sticker',
 'comb',
 'kit',
 'body_hero',
 'futuredew',
 'cleanser',
 'after_balm',
 'moisturizer',
 'retinol',
 'sunscreen',
 'zit',
 'tin',
 'cranberry',
 'cordial',
 'olivia_rodrigo'])

# COMMAND ----------

## Filter the dataframe to where there is a value of 1 (product name is mentioned)
long_df = long_df.filter(col("value") == 1)
long_df.show()

# COMMAND ----------

## Convert the view into a SQL 
long_df.createOrReplaceTempView("long_df_vw")

## Get the count of each product 
agg_df1 = spark.sql("select variable,count(1) as product_count \
                    from long_df_vw group by 1")

## Get the count of each sentiment by product 
agg_df2 = spark.sql("select variable, sentiment, count(1) as total_count \
                    from long_df_vw group by 1,2 order by variable, count(1)")

## Rename columns to not have duplicates
agg_df2 = agg_df2.withColumnRenamed("variable","var")

# COMMAND ----------

## Create views of the data frames 
agg_df1.createOrReplaceTempView("agg_df1_vw")
agg_df2.createOrReplaceTempView("agg_df2_vw")

## Combine the two data frames 
agg_df3 = spark.sql("select * \
                    from agg_df2_vw a join agg_df1_vw b on a.var = b.variable")

## Show the dataframe 
agg_df3.show(5)

# COMMAND ----------

## Create view of the data frame 
agg_df3.createOrReplaceTempView("agg_df3_vw")

## Calculate the ratio 
agg_df4 = spark.sql("select var, sentiment, total_count, product_count, total_count / product_count * 100 as ratio \
                    from agg_df3_vw")

## Rename columns to not have duplicates
agg_df4 = agg_df4.withColumnRenamed("product_count","total_mention")
agg_df4 = agg_df4.withColumnRenamed("total_count","sentiment_count")

## Display the data frame 
agg_df4.show(5)

# COMMAND ----------

## Filter the dataframes for positve and negative sentiment respectively
pos_df = agg_df4.filter(col("sentiment") == "positive")
neg_df = agg_df4.filter(col("sentiment") == "negative")
neg_df.createOrReplaceTempView("neg_df_vw")
pos_df.createOrReplaceTempView("pos_df_vw")

## Get the top 10 positive products by ratio 
pos_df= spark.sql("select * \
                    from pos_df_vw order by ratio desc limit 10")

## Get the top 10 negative products by ratio 
neg_df= spark.sql("select * \
                    from neg_df_vw order by ratio desc limit 10")

# COMMAND ----------

## Display summary table 
cols = ("sentiment")
summary_df = pos_df.drop(cols)
summary_df = summary_df.withColumnRenamed("var","product")
summary_df = summary_df.withColumnRenamed("ratio","percent_pos_mentions (%)")
summary_df = summary_df.withColumnRenamed("sentiment_count","num_positive_mentions")
summary_df = summary_df.withColumnRenamed("total_mention","num_total_mentions")
summary_df.show(5)

# COMMAND ----------

## Save the summary table to a location in repo 
import os
summary_df = summary_df.toPandas()
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/", "top_pos_prod_sum_tbl.csv")
summary_df.to_csv(fpath)

# COMMAND ----------

## Import necessary visualization libraries 
import matplotlib.pyplot as plt
import seaborn as sns

## Convert the data frames into pandas for visualization 
pos_df = pos_df.toPandas()
neg_df = neg_df.toPandas()

## Define color pallete 
clrs = ["#9EF478", "#EDF197"]

## Plot the positive sentiment for each product in the glossier subredit 
sns.set(font_scale=3, rc={'figure.figsize':(20,10)})
sns.set_theme(style='white')
p1 = sns.barplot(data=pos_df, x='var', y='ratio', hue='sentiment', palette=clrs)
p1.set_title('Positive Sentiment Ratio By Glossier Product')
p1.set_ylabel('% Positive Activity Of Total')
p1.set_xlabel('Product')
p1 = plt.xticks(rotation=45)

# COMMAND ----------

## Create the same visualization in plotly for interactiveness in html file 
import plotly.express as px
fig = px.bar(pos_df, x="var", y="ratio", 
             color="sentiment", 
            title = "Positive Sentiment Ratio By Glossier Product",
            color_discrete_sequence=["#9EF478"])

fig.update_layout(plot_bgcolor = "white",  xaxis_title="Product", yaxis_title="% Positive Activity Of Total", title_x=0.5,
                 xaxis={'categoryorder':'total descending'})
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/", "glossier_pos_sent_viz.html")
newpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/website/images/", "glossier_pos_sent_viz.html")
fig.write_html(newpath)
fig.write_html(fpath)
fig.show()

# COMMAND ----------

## Define color pallete 
clrs = ["#E3242B"]

## Plot the negative sentiment for each product in the glossier subredit 
sns.set(font_scale=3, rc={'figure.figsize':(20,10)})
sns.set_theme(style='white')
p1 = sns.barplot(data=neg_df, x='var', y='ratio', hue='sentiment', palette=clrs)
p1.set_title('Neagtive Sentiment Ratio By Glossier Product')
p1.set_ylabel('% Negative Activity Of Total')
p1.set_xlabel('Product')
p1 = plt.xticks(rotation=45)

# COMMAND ----------

## Create the same visualization in plotly for interactiveness in html file 
import plotly.express as px
fig = px.bar(neg_df, x="var", y="ratio", 
             color="sentiment", 
            title = "Negative Sentiment Ratio By Glossier Product",
            color_discrete_sequence=["#E3242B"])

fig.update_layout(plot_bgcolor = "white",  xaxis_title="Product", yaxis_title="% Negative Activity Of Total", title_x=0.5,
                 xaxis={'categoryorder':'total descending'})
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/", "glossier_neg_sent_viz.html")
newpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/website/images/", "glossier_neg_sent_viz.html")
fig.write_html(newpath)
fig.write_html(fpath)
fig.show()

# COMMAND ----------

## Filter the dataframes for neutral sentiment respectively
neu_df = agg_df4.filter(col("sentiment") == "neutral")
neu_df.createOrReplaceTempView("neu_df_vw")
neu_df= spark.sql("select * \
                    from neu_df_vw order by ratio desc limit 10")

## Convert the data frames into pandas for visualization 
neu_df = neu_df.toPandas()

## Define color pallete 
clrs = ["#FAE29C"]

## Plot the negative sentiment for each product in the glossier subredit 
sns.set(font_scale=3, rc={'figure.figsize':(20,10)})
sns.set_theme(style='white')
p1 = sns.barplot(data=neu_df, x='var', y='ratio', hue='sentiment', palette=clrs)
p1.set_title('Neutral Sentiment Ratio By Glossier Product')
p1.set_ylabel('% Neutral Activity Of Total')
p1.set_xlabel('Product')
p1 = plt.xticks(rotation=45)

# COMMAND ----------

## Create the same visualization in plotly for interactiveness in html file 
import plotly.express as px
fig = px.bar(neu_df, x="var", y="ratio", 
             color="sentiment", 
            title = "Neutral Sentiment Ratio By Glosier Product",
            color_discrete_sequence=["#FAE29C"])

fig.update_layout(plot_bgcolor = "white",  xaxis_title="Product", yaxis_title="% Neutral Activity Of Total", title_x=0.5,
                 xaxis={'categoryorder':'total descending'})
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/", "glossier_neut_sent_viz.html")
fig.write_html(fpath)
newpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/website/images/", "glossier_neut_sent_viz.html")
fig.write_html(newpath)
fig.show()

# COMMAND ----------


