# Databricks notebook source
# MAGIC %md
# MAGIC # Glossier Business Insights
# MAGIC ## Exploratory Data Analysis
# MAGIC 
# MAGIC ### ANLY 502
# MAGIC ### Clare Garberg, Stephanie Plaza, Amalia Stahl, Kajal Tiwary
# MAGIC 
# MAGIC Project Objective: The goal of this project is to determine the most optimal launch and marketing strategy for Glossier as they work to expand their physical presence from a predominantly online business. For context, Glossier is a beauty brand headquartered in New York. Furthermore, in contrast to some of its competitors, Glossier sells all of its products almost exclusively through its own website and has a very limited number of retail locations. We will assume the personas of Glossier business owners and executives who will leverage insights gleaned from a variety of data to make key business decisions with regards to direction of the business.
# MAGIC 
# MAGIC Specifically, we have the following goals:

# COMMAND ----------

# MAGIC %md
# MAGIC Business Goal 1: We will determine where Glossier should open their next store.
# MAGIC 
# MAGIC Technical Proposal: We will use NLP to identify posts that mention geospatial terms and the term “Glossier” to ensure there is an association between the organization and the location. We will then conduct sentiment analysis of each post to assign a positive, negative, or neutral value. Finally, we will count the number of posts by sentiment for each geospatial area in the United States. We will represent this “Glossier overall sentiment” for each location on a choropleth map to inform executives of regional brand sentiment.  Areas with positive sentiment will indicate optimal store presence.
# MAGIC 
# MAGIC Business Goal 2: We will identify which products should be in the newest Glossier kit.
# MAGIC 
# MAGIC Technical Proposal: We will scrape the Glossier website to identify its complete set of products. We will then use NLP techniques to identify which posts and comments in the Glossier subreddit contain these products. We will conduct sentiment analysis of each post to assign positive, negative, or neutral values. We will sum the number of posts and comments by product for positive posts. We will identify the top 10 products with the highest activity (positive sentiment). Similarly, we will sum the number of posts and comments by product for negative posts. We will identify the top 10 products with the highest activity (negative sentiment). We will display this information on a faceted chart to show executive audiences which products should be promoted, and which should potentially be discontinued.
# MAGIC  
# MAGIC  
# MAGIC Business Goal 3: We will identify which products are most common amongst competitors to determine where we can capitalize on market share.
# MAGIC 
# MAGIC Technical Proposal: For two of Glossier’s competitors, we will use NLP techniques to identify which posts and comments in the competitor subreddits contain the products identified from Glossier’s website. Similar to the business objective above, we will conduct sentiment analysis of each post to assign positive, negative, or neutral values. We will sum the number of posts and comments by product for positive posts. We will identify the top 10 products with the highest activity (positive sentiment). Similarly, we will sum the number of posts and comments by product for negative posts. We will identify the top 10 products with the highest activity (negative sentiment). We will display this information on a faceted chart to depict the worst performing products and highest performing products are of our competitors.
# MAGIC 
# MAGIC Business Goal 4: We will predict when Glossier demand will be highest over the next year and contrast that with competitor forecasts.
# MAGIC 
# MAGIC Technical Proposal: Leveraging regular expressions and searching techniques, we will identify the number of posts and comments that Glossier is mentioned in by day.  Using the total activity by day as demand, we will perform exploratory data analysis to identify if seasonality is present and what type of ML model should be leveraged (e.g., ARMA, SARIMA, etc.) for univariate time series forecasting. This information will be depicted on line charts to easily see demand fluctuations, seasonality, and forecasts over time. The output of this analysis will give us insight into how Glossier should manage their inventory. This same analysis will be repeated for Ulta Beauty (competitor) to have a benchmark for comparison.
# MAGIC 
# MAGIC Business Goal 5: We will identify how Glossier demand is affected by COVID-19 rates and forecast the relationship for the next year.
# MAGIC 
# MAGIC Technical Proposal: To accomplish this, we will join external COVID-19 rate data to the Glossier subreddit activity data by day. Like above, we will identify the number of posts and comments that Glossier is mentioned in by day. We will also aggregate the total COVID-19 cases by day. To measure the effect of the COVID-19 rates on the demand, we will develop a multivariate time series ML model to forecast disease rates in conjunction with the demand. As mentioned above, this information will also be depicted on line charts to easily see the relationships and patterns between the two variables and the forecasts over time.
# MAGIC 
# MAGIC Business Goal 6: We will identify the average persona of a Glossier customer to have a comprehensive understanding of our customer base.
# MAGIC 
# MAGIC Technical Proposal: For users who post on the Glossier subreddit, we will identify their total activity by joining other subreddits and counting the number of other subreddits these users post to. In addition to measuring their level of engagement on the platform, we will use NLP techniques to identify the gender, age, and sentiment of each post. We will bin the age into various groups. We will then count the number of times (posts) each gender, age group, and sentiment value appear for a given user and assign the majority value to that user. We will then count the number of users that are in each gender, age, and sentiment category and depict the high-level information via bar and pie charts to contrast demographic information.
# MAGIC 
# MAGIC Business Goal 7: We will determine if user demographics are predictive of sentiment to target and scale marketing efforts.
# MAGIC 
# MAGIC Technical Proposal: Leveraging the gender, age, total activity, and sentiment variables calculated in the previous objective, we will use ML and develop a classification model for sentiment prediction. To attain the best accuracy, different models - SVM, Naive Bayes, and logistic regression - will be developed. The features will be gender, age, and total activity. The model will be trained on 80% of the dataset and predictions will be run on a test set. Each model will be tuned to obtain optimal hyperparameters. The top feature will be extracted, and the accuracies of each model will be visualized to depict the efficacy of prediction to technical executives. 
# MAGIC 
# MAGIC Business Goal 8: We will determine if user demographics can predict the exact sentiment of a user to inform experimentation and rollout strategy.
# MAGIC 
# MAGIC Technical Proposal: Leveraging the gender, age, total activity, and sentiment variables calculated in the previous objective, we will leverage ML to develop a regression model for sentiment prediction. In this case, NLP (Vader) will be used to calculate the compound sentiment score of each post. The average score will then be determined and attributed to each user. Leveraging machine learning, a linear regression model will be built to predict the sentiment score of a user based on their gender, age, and overall activity. We will run predictions on a test set and tune the hyperparameters to achieve highest accuracy. If we can accurately predict sentiment score, we can experiment with more targeted marketing strategies and gauge incremental impact.
# MAGIC 
# MAGIC Business Goal 9: We will identify how the popularity of Glossier compares to the popularity of other competitor brands across multiple sources.
# MAGIC 
# MAGIC Technical Proposal: T0 accomplish this, we will join a customized google search trends dataset with the Glossier competitor dataset. To assess if a correlation exists between the volume of google searches and the volume of Glossier mentions in the subreddit over time, the volume of google searches and mentions will respectively be aggregated by day. NLP sentiment analysis will again be leveraged to gauge the percentage of positive searches and contrast that to the percentage of positive activity in the subreddit. This will inform us of the popularity and positiveness across platforms to influence advertising strategies. This process will be repeated for at least one other competitor brand to again leverage as a benchmark.
# MAGIC 
# MAGIC Business Goal 10: We will identify how important sustainability is to Glossier customers in order to determine if they should pivot their strategy. 
# MAGIC 
# MAGIC Technical Proposal: As eco-friendly options in the ecommerce space as well as the conversation of  sustainability have grown over the past few years, we hope to gauge how important environmentally conscious practices and products are to the Glossier consumer.  In order to gain a better understanding of this, we will perform topic modeling on posters and commenters data in the Glossier subreddit. Subtasks of this will include vectorizing the text data by using Count and TF-IDF vectorizer and performing LSA and NMF to reduce the dimensionality of input text data. We may discover that sustainability is not a top priority to our customers but might instead gain insights as to what topics are on the customers top of mind (new products, store expansion, etc.). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Gathering
# MAGIC 
# MAGIC **Three separate Reddit datasets were gathered to support the three main streams of analysis:**
# MAGIC 1. Glossier Data: Glossier subreddit activity + any other post mentioning “glossier”.
# MAGIC   - This dataset will be utilized for analysis of Glossier's customer viewpoints. their sentiment regarding products, services, and stores.
# MAGIC 
# MAGIC 2. Competitor Data: Competitor subreddit activity + activity on some general makeup channels.
# MAGIC   - This dataset will be utilized for competitive analysis, to give Glossier a more accurate picture of the competitive market.
# MAGIC 
# MAGIC 3. Glossier Authors Data: All reddit activity of authors of posts/comments on r/glossier.
# MAGIC   - This dataset will be utilized for general analysis of Glossier's customer base, to build a picture of the persona of a Glossier customer.

# COMMAND ----------

pip install nltk

# COMMAND ----------

pip install wordcloud

# COMMAND ----------

pip install pywaffle

# COMMAND ----------

# taking a look
dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/")

# COMMAND ----------

# reading in comments and posts data as spark dataframes
comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

comments.printSchema()

# COMMAND ----------

submissions.printSchema()

# COMMAND ----------

# first gathering all posts and comments in the glossier subreddit as well as any general mentions of glossier
import pyspark.sql.functions as F

glossier_submissions = submissions.filter((submissions.subreddit == "glossier") | (F.lower(submissions.selftext).contains('glossier')) )
glossier_comments = comments.filter((comments.subreddit == "glossier") | (F.lower(comments.body).contains('glossier')) )

# COMMAND ----------

# total number of glossier comments
# 131058
glossier_comments.count()

# COMMAND ----------

# total number of glossier submissions
# 11831
glossier_submissions.count()

# COMMAND ----------

# saving to a parquet
glossier_comments.write.format('parquet').save("/FileStore/glossier/glossier_comments")
glossier_submissions.write.format('parquet').save("/FileStore/glossier/glossier_submissions")

# COMMAND ----------

# gathering competitor data as well for competitive analysis
# https://www.similarweb.com/website/glossier.com/competitors/
# these are just the ones with subreddits
# including some general makeup subreddits to analyze how/how often glossier is mentioned vs other brands

competitor_list = ["Sephora", "Ulta", "Fentybeauty", "Makeup", "MakeupAddiction"]
competitor_comments = comments.filter(comments.subreddit.isin(competitor_list))
competitor_submissions = submissions.filter(submissions.subreddit.isin(competitor_list))

# COMMAND ----------

# 1105784
competitor_comments.count()

# COMMAND ----------

# 108341
competitor_submissions.count()

# COMMAND ----------

# saving to a parquet
competitor_comments.write.format('parquet').save("/FileStore/glossier/competitor_comments")
competitor_submissions.write.format('parquet').save("/FileStore/glossier/competitor_submissions")

# COMMAND ----------

# now let's get glossier posters and commentors information
glossiersubreddit_comments = comments.filter(comments.subreddit == "glossier")
glossiersubreddit_submissions = submissions.filter(submissions.subreddit == "glossier")

# COMMAND ----------

# getting unique users
glossiersubreddit_unique_authors_comments = glossiersubreddit_comments.select("author").distinct()
glossiersubreddit_unique_authors_submissions = glossiersubreddit_submissions.select("author").distinct()

# combining and getting a list of all unique users who interacted with the subreddit
glossiersubreddit_unique_authors = glossiersubreddit_unique_authors_comments.union(glossiersubreddit_unique_authors_submissions)
glossiersubreddit_unique_authors = glossiersubreddit_unique_authors.distinct()

# COMMAND ----------

glossiersubreddit_unique_authors.show(5)

# COMMAND ----------

# removing the automoderator
glossiersubreddit_unique_authors = glossiersubreddit_unique_authors.filter(glossiersubreddit_unique_authors.author != "AutoModerator")

# COMMAND ----------

# 10544 people
glossiersubreddit_unique_authors.count()

# COMMAND ----------

# collecting these values in a list to search
unique_authors_list = glossiersubreddit_unique_authors.rdd.map(lambda x: x[0]).collect()

# COMMAND ----------

# now let's get all reddit activity for glossier authors
glossierauthors_comments = comments.filter(comments.author.isin(unique_authors_list))
glossierauthors_submissions = submissions.filter(submissions.author.isin(unique_authors_list))

# COMMAND ----------

# getting counts
# 536836714
glossierauthors_comments.count()

# COMMAND ----------

# 168454995
glossierauthors_submissions.count()

# COMMAND ----------

# it might make sense to remove their glossier posts but I will leave them in for now
# saving to a parquet
glossierauthors_comments.write.format('parquet').save("/FileStore/glossier/glossierauthors_comments")
glossierauthors_submissions.write.format('parquet').save("/FileStore/glossier/glossierauthors_submissions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Glossier Authors Data

# COMMAND ----------

# MAGIC %md
# MAGIC All NSFW submissions were removed from this dataset as that is not a type of user activity that Glossier would be interested when building customer personas. Additionally all null values were filtered from the author columns because we are only interested in known users for these personas. Additionally, moderators and admins were filtered out as their comments/submissions would not be authentic subreddit content. Finally, only necessary columns and aggregations were kept in the final dataframe for plotting, subreddit, count, and summation aggregations.

# COMMAND ----------

# reading in the data

glossierauthors_comments = spark.read.parquet("dbfs:/FileStore/glossier/glossierauthors_comments")
glossierauthors_submissions = spark.read.parquet("dbfs:/FileStore/glossier/glossierauthors_submissions")

# removing some null values
glossierauthors_submissions = glossierauthors_submissions.filter(glossierauthors_submissions.author.isNotNull())
glossierauthors_comments = glossierauthors_comments.filter(glossierauthors_comments.author.isNotNull())

# then let's filter out NSFW submissions
glossierauthors_submissions = glossierauthors_submissions.filter(glossierauthors_submissions.over_18 == False)

# now let's filter out moderators/admins
glossierauthors_submissions = glossierauthors_submissions.filter(glossierauthors_submissions.distinguished.isNull())
glossierauthors_comments = glossierauthors_comments.filter(glossierauthors_comments.distinguished.isNull())

# COMMAND ----------

# getting subreddits activity for submissions and comments
from pyspark.sql.functions import col, asc,desc

submissions_by_subreddit = glossierauthors_submissions.groupBy("subreddit").count().orderBy(col("count"), ascending=False).collect()

# comments by subreddit
comments_by_subreddit = glossierauthors_comments.groupBy("subreddit").count().orderBy(col("count"), ascending=False).collect()

# COMMAND ----------

# now let's join and get glossier user interests by total interactions
import pandas as pd

comments_by_subreddit_grouped = spark.createDataFrame(comments_by_subreddit).toPandas()
submissions_by_subreddit_grouped = spark.createDataFrame(submissions_by_subreddit).toPandas()
all_by_subreddit = pd.merge(comments_by_subreddit_grouped,submissions_by_subreddit_grouped, on='subreddit', how='outer')

# imputing any nulls as 0
all_by_subreddit_clean = all_by_subreddit.fillna(0)

# getting total activity
all_by_subreddit_clean['total_activity'] = all_by_subreddit_clean['count_x'] + all_by_subreddit_clean['count_y']
all_by_subreddit_clean = all_by_subreddit_clean.rename(columns={'count_x': 'comment_count', 'count_y': 'submission_count'})
all_by_subreddit_clean = all_by_subreddit_clean.sort_values("total_activity", ascending=False)
convert_dict = {'comment_count': int,
                'submission_count': int,
                'total_activity': int }  
all_by_subreddit_clean = all_by_subreddit_clean.astype(convert_dict) 

# COMMAND ----------

## create a directory called data/plots and data/csv to save generated data
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

# getting top 100 by total for plotting and to save to csv
all_by_subreddit_clean_top100 = all_by_subreddit_clean.head(100)
fpath = os.path.join(CSV_DIR, "viz_1.csv")
all_by_subreddit_clean_top100.to_csv(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Glossier Data

# COMMAND ----------

# MAGIC %md
# MAGIC All columns that are not necessary for the EDA analysis and downstream ML and NLP analysis were removed. There were no missing values present in the remaining comments dataframe and the missing values present in the submissions data frame were left in as they were not pertinent to the EDA analysis. Based on the nuances of the downstream modeling, columns with missing values may be removed or missing values will either be removed or imputed. Additionally, all columns were converted to their appropriate data types. 

# COMMAND ----------

## Read in the data 
glos_comm = spark.read.parquet("/FileStore/glossier/glossier_comments")

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

# COMMAND ----------

## Read in the data 
glos_sub = spark.read.parquet("/FileStore/glossier/glossier_submissions")

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

## Convert the data frame into a view to run SQL on 
glos_sub.createOrReplaceTempView("glos_sub_vw")

## Part 1: 
## Get a count of all the missing values 
missing_glos_sub1= spark.sql("select sum(case when author_id is null then 1 else 0 end) as author_id, \
                    sum(case when archived is null then 1 else 0 end) as archived, \
                    sum(case when brand_safe is null then 1 else 0 end) as brand_safe, \
                    sum(case when created_utc is null then 1 else 0 end) as created_utc, \
                    sum(case when crosspost_parent is null then 1 else 0 end) as crosspost_parent, \
                    sum(case when disable_comments is null then 1 else 0 end) as disable_comments, \
                    sum(case when distinguished is null then 1 else 0 end) as distinguished, \
                    sum(case when hidden is null then 1 else 0 end) as hidden, \
                    sum(case when hide_score is null then 1 else 0 end) as hide_score, \
                    sum(case when id is null then 1 else 0 end) as id, \
                    sum(case when is_crosspostable is null then 1 else 0 end) as is_crosspostable, \
                    sum(case when is_reddit_media_domain is null then 1 else 0 end) as is_reddit_media_domain, \
                    sum(case when is_self is null then 1 else 0 end) as is_self, \
                    sum(case when is_video is null then 1 else 0 end) as is_video, \
                    sum(case when locked is null then 1 else 0 end) as locked \
                    from glos_sub_vw")

## Show and view the results 
missing_glos_sub1.show()

# COMMAND ----------

## Part 2: 
## Get a count of all the missing values 
missing_glos_sub2 = spark.sql("select sum(case when num_comments is null then 1 else 0 end) as num_comments,\
                    sum(case when num_crossposts is null then 1 else 0 end) as num_crossposts,\
                    sum(case when over_18 is null then 1 else 0 end) as over_18,\
                    sum(case when pinned is null then 1 else 0 end) as pinned,\
                    sum(case when promoted is null then 1 else 0 end) as promoted,\
                    sum(case when promoted_by is null then 1 else 0 end) as promoted_by,\
                    sum(case when promoted_by is null then 1 else 0 end) as promoted_display_name,\
                    sum(case when score is null then 1 else 0 end) as score,\
                    sum(case when selftext is null then 1 else 0 end) as selftext,\
                    sum(case when spoiler is null then 1 else 0 end) as spoiler,\
                    sum(case when stickied is null then 1 else 0 end) as stickied,\
                    sum(case when subreddit is null then 1 else 0 end) as subreddit,\
                    sum(case when subreddit_id is null then 1 else 0 end) as subreddit_id,\
                    sum(case when title is null then 1 else 0 end) as title\
                    from glos_sub_vw")

## Show and view the results 
missing_glos_sub2.show()

# COMMAND ----------

## View the data types of each column 
glos_sub.dtypes

# COMMAND ----------

## Convert the columns to appropriate data type 
glos_sub_final = glos_sub.withColumn("created_utc",glos_sub.created_utc.cast('timestamp'))

# COMMAND ----------

## Import all necessary libraries
from pyspark.sql.functions import col, asc,desc
from pyspark.sql.functions import *

## Filter to the glossier subreddit only 
df1 = glos_comm_final.filter((col("subreddit") == "glossier"))
## Get the count of records for each day 
df1 = df1.groupBy(to_date("created_utc").alias("dt_day")).agg(count("id").alias("comment_count"))

## Filter to the glossier subreddit only 
df2 = glos_sub_final.filter((col("subreddit") == "glossier"))
## Get the count of records for each day 
df2 = df2.groupBy(to_date("created_utc").alias("dt")).agg(count("id").alias("submission_count"))

## We are using SQL so need to convert them into views 
df2.createOrReplaceTempView("x_vw")
df1.createOrReplaceTempView("y_vw")

## Conduct an outer join to get all of the information from both dataframes 
join_df = df2.join(df1,df2.dt ==  df1.dt_day,"outer")
join_df = join_df.drop("dt")
join_df.show()

# COMMAND ----------

#making a summary table 
#aggregate by month 
from pyspark.sql.functions import year
from pyspark.sql.functions import to_date
import os
table_1=join_df
 
table_2 = table_1.withColumn('dt_month',month(table_1.dt_day))
table_2 = table_2.withColumn('dt_year',year(table_1.dt_day))
table_3 = table_2.drop("dt_day")
table_4 = table_3.withColumn("dt_month_year",concat(col("dt_year"),lit('-'),col("dt_month")))
table_4a=table_4.select("comment_count","dt_month_year")
table_4b=table_4.select("submission_count","dt_month_year")
table_5a = table_4a.groupBy('dt_month_year').agg(sum("comment_count").alias("comment_count"))
table_5b = table_4b.groupBy('dt_month_year').agg(sum("submission_count").alias("submission_count"))

#Join these 2 tables into one table 
table_6 = table_5b.join(table_5a,table_5b.dt_month_year == table_5a.dt_month_year,"inner").drop(table_5b.dt_month_year)
table_6= table_6.select(col("dt_month_year"),to_date(col("dt_month_year"),"yyyy-M").alias("date"),col("comment_count"),col("submission_count")) 

table_7= table_6.sort("date")

table_8=table_7.toPandas()
table_9=table_8[['dt_month_year', 'comment_count', 'submission_count']]
table_9

PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "summary_table_2.csv")
table_9.to_csv(fpath)

join_df = join_df.withColumn('total_activity', join_df.submission_count + join_df.comment_count)
join_df = join_df.drop("submission_count","comment_count")
join_df = join_df.toPandas()
fpath = os.path.join(CSV_DIR, "viz_2.csv")
join_df.to_csv(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Competitor Data

# COMMAND ----------

# Step 1: read in the data 
comments = spark.read.parquet("/FileStore/glossier/competitor_comments")
submissions= spark.read.parquet("/FileStore/glossier/competitor_submissions")
#check that data is read in correctly 
#comments.show(5)         
#submissions.show(5)     
comments_count=comments.count()
submissions_count=submissions.count()
print("comments_count:",comments_count)
print("submissions_count:",submissions_count)

# COMMAND ----------

#Step 2: Data cleaning (specific subredit channels + cleaning )
#a) We are only interested in general makeup channels 
#in this case we are interested in 2 subreddits specifically "Makeup", "MakeupAddiction"
#So lets filter based on those subredits 
competitor_list = ["Makeup", "MakeupAddiction"]
competitor_comments = comments.filter(comments.subreddit.isin(competitor_list))
competitor_submissions = submissions.filter(submissions.subreddit.isin(competitor_list))
#b) Now that we have the correct subreddits we are going to use the body column to get our counts for glossier commpetitors including glossier 
#so keep only body column for comments 
#and title comlumn for submissions
competitor_comments_body=competitor_comments.select("body")
competitor_comments_body.show(1)
competitor_submissions_title=competitor_submissions.select("title")
competitor_submissions_title.show(1)
#Lets get a total count for these variables 
competitor_comments_body_count=competitor_comments_body.count()
print("competitor_comments_body_count:",competitor_comments_body_count)
competitor_submissions_title_count=competitor_submissions_title.count()
print("competitor_submissions_title_count",competitor_submissions_title_count)


# COMMAND ----------

#c) #check shape of the data frame and null values !
print("competitor_comments_body:")
(competitor_comments_body.count(), len(competitor_comments_body.columns))
print("competitor_submissions_title:")
(competitor_submissions_title.count(), len(competitor_submissions_title.columns))
#------
## Convert the data frame into a view to run SQL on 
competitor_comments_body.createOrReplaceTempView("c_c_body")
competitor_submissions_title.createOrReplaceTempView("c_s_title")


missing_competitor_comm = spark.sql("select \
                    sum(case when body is null then 1 else 0 end) as body_count \
                    from c_c_body")


missing_competitor_sub = spark.sql("select \
                    sum(case when title is null then 1 else 0 end) as title_count \
                    from c_s_title")

## Show and view the results 
missing_competitor_comm.show()
missing_competitor_sub.show()
#I have no missing values which is great !

# COMMAND ----------

from pyspark.sql.functions import col
#now we need to find how much their competitors are mentioned 
#we consider main competitors, Sephora, Ulta, Fenty, Milk 
main_competitors = ["Sephora", "Ulta","Fenty"] 
#Now we need to finf the rows that contain Sephora and keep them 
#competitor_comments_body_Sephora=competitor_comments_body.filter(col("body").rlike("Sephora"))
competitor_comments_body_brand=competitor_comments_body.withColumn("Sephora",col("body").rlike("Sephora|sephora"))
#Ulta
competitor_comments_body_brand2=competitor_comments_body_brand.withColumn("Ulta",col("body").rlike("Ulta|ulta"))
#competitor_comments_body_brand2.show()
competitor_comments_body_brand3=competitor_comments_body_brand2.withColumn("Fenty",col("body").rlike("Fenty|fenty"))
#competitor_comments_body_brand3.show()
competitor_comments_body_brand4=competitor_comments_body_brand3.withColumn("Glossier",col("body").rlike("Glossier|glossier"))
#competitor_comments_body_brand4.show()
count_Sephora=competitor_comments_body_brand4.groupBy('Sephora').count()
count_Sephora.show()
count_Ulta=competitor_comments_body_brand4.groupBy('Ulta').count()
count_Ulta.show()
count_Fenty=competitor_comments_body_brand4.groupBy('Fenty').count()
count_Fenty.show()
count_Glossier=competitor_comments_body_brand4.groupBy('Glossier').count()
count_Glossier.show()



# COMMAND ----------

#Now we need to do the same for submissions 
from pyspark.sql.functions import col
competitor_submissions_title2=competitor_submissions_title.withColumn("Sephora",col("title").rlike("Sephora|sephora"))
#Ulta
competitor_submissions_title3=competitor_submissions_title2.withColumn("Ulta",col("title").rlike("Ulta|ulta"))
competitor_submissions_title4=competitor_submissions_title3.withColumn("Fenty",col("title").rlike("Fenty|fenty"))
competitor_submissions_title5=competitor_submissions_title4.withColumn("Glossier",col("title").rlike("Glossier|glossier"))
count_Sephora_title=competitor_submissions_title5.groupBy('Sephora').count()
count_Sephora_title.show()
count_Ulta_title=competitor_submissions_title5.groupBy('Ulta').count()
count_Ulta_title.show()
count_Fenty_title=competitor_submissions_title5.groupBy('Fenty').count()
count_Fenty_title.show()
count_Glossier_title=competitor_submissions_title5.groupBy('Glossier').count()
count_Glossier_title.show()

# COMMAND ----------

#final transformation step: make into one dataframe 
#We can use pandas now 
import pandas as pd
#submissions
competitor_submissions_title6=competitor_submissions_title5.select('Sephora','Ulta','Fenty','Glossier')
#competitor_submissions_title6.show()
#comments
competitor_comments_body_brand5=competitor_comments_body_brand4.select('Sephora','Ulta','Fenty','Glossier')
#competitor_comments_body_brand5.show()
#Now I am going to group by columns 
group_cols = ["Sephora", "Ulta","Fenty","Glossier"]
competitor_submissions_title7=competitor_submissions_title6.groupBy(group_cols).count()
competitor_submissions_title7.show()
competitor_comments_body_brand6=competitor_comments_body_brand5.groupBy(group_cols).count()
competitor_comments_body_brand6.show()
#this is interesting but perhpas a bit complicated for the average consumer 
#however I am keeping these as 2 tables that could be interesting for some further analysis down the road 
#Save these dataframes 
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "summary_table_3.csv")
summary_3 =competitor_submissions_title7.toPandas()
summary_3.to_csv(fpath)
summary_4 =competitor_comments_body_brand6.toPandas()
fpath = os.path.join(CSV_DIR, "summary_table_4.csv")
summary_4.to_csv(fpath)

# COMMAND ----------

##Step 3: Prepping data for viz 
count_Sephora_True=count_Sephora.filter("Sephora==True")
count_Sephora_True2=count_Sephora_True.select("count")
#count_Sephora_True.show()
count_Ulta_True=count_Ulta.filter("Ulta==True")
count_Ulta_True2=count_Ulta_True.select("count")
#count_Ulta_True.show()
count_Fenty_True=count_Fenty.filter("Fenty==True")
count_Fenty_True2=count_Fenty_True.select("count")
#count_Fenty_True.show()
count_Glossier_True=count_Glossier.filter("Glossier==True")
count_Glossier_True2=count_Glossier_True.select("count")
#count_Glossier_True.show()

result1 = count_Sephora_True2.union(count_Ulta_True2)
result2 = result1.union(count_Fenty_True2)
result3 = result2.union(count_Glossier_True2)
result3 = result3.toPandas()

brand_names=["Sephora","Ulta","Fenty","Glossier"]
result3['brands']= brand_names
result3

# COMMAND ----------

#now for submissions 
count_Sephora_title_True=count_Sephora_title.filter("Sephora==True")
count_Sephora_title_True2=count_Sephora_title_True.select("count")
#count_Sephora_title_True.show()
count_Ulta_title_True=count_Ulta_title.filter("Ulta==True")
count_Ulta_title_True2=count_Ulta_title_True.select("count")
#count_Ulta_title_True.show()
count_Fenty_title_True=count_Fenty_title.filter("Fenty==True")
count_Fenty_title_True2=count_Fenty_title_True.select("count")
#count_Fenty_title_True.show()
count_Glossier_title_True=count_Glossier_title.filter("Glossier==True")
count_Glossier_title_True2=count_Glossier_title_True.select("count")
#count_Glossier_title_True.show()

result1_title = count_Sephora_title_True2.union(count_Ulta_title_True2)
result2_title = result1_title.union(count_Fenty_title_True2)
result3_title = result2_title.union(count_Glossier_title_True2)
result3_title= result3_title.toPandas()

brand_names=["Sephora","Ulta","Fenty","Glossier"]
result3_title['brands_title']= brand_names
result3_title['count_title']= result3_title['count']
result3_title=result3_title.drop(['count'], axis=1)
result3_title

# COMMAND ----------

#Now make these 2 data frames into one 
#result3
#result3_title
final_df_temp=pd.concat([result3, result3_title], axis=1)
#final_df_temp
final_df_temp['Total_Count'] = final_df_temp['count'] + final_df_temp['count_title']
#final_df_temp
final_df=final_df_temp.drop(['count','brands_title','count_title'], axis=1)
final_df
#save dataframe to cvs
fpath = os.path.join(CSV_DIR, "viz_3.csv")
final_df.to_csv(fpath)

# COMMAND ----------

#creating new variable in order to get percentage of competitor counts 

df=final_df
#make the valus from the dataframe into a list 
value_list = df["Total_Count"].tolist()
#print(value_list)

#get the total count of  values, meaning comment that contain, sephora, fenty, ulta or glossier 
sum_value_list=sum(value_list)
value_list2 = [round(x / sum_value_list*100) for x in value_list]
print(value_list2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC This section includes both charts as well as summary tables for additional insight. It is imperative to point out that the Python code used to derive the summary table statistics is included in the data cleaning portion above; in order to make the tables more aesthetically pleasing, we exported those tables and leveraged R (ANLY502_EDATables.Rmd) to produce the visuals included below. The code to produce the visualizations in Python is included below.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. *What subreddits do Glossier users interact with the most?*

# COMMAND ----------

# plotting the top 20 most popular subreddits to glossier authors
import plotly.express as px

fig = px.bar(all_by_subreddit_clean_top100.head(20), y='total_activity', x='subreddit', text_auto='.2s',
            color_discrete_sequence =['purple']*len(all_by_subreddit_clean_top100.head(20)),
            title="20 Most Popular Subreddits To Glossier Users By Total Activity <br><sup>January 2021 - August 2022</sup>")
fig.update_layout(plot_bgcolor = "white",  xaxis_title="Subreddit Name", yaxis_title="Subreddit Activity (Comments + Posts)", title_x=0.5)
plot_fpath = os.path.join(PLOT_DIR, 'viz_1.html')
fig.write_html(plot_fpath)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The figure above shows the top subreddits by total activity (comments plus submissions) for those unique users who have posted in r/Glossier. This successfully shows popular interests among these "Glossier Authors", their market base. Individuals in the Glossier market are also interested in investing which is clear from their interactions with r/CryptoCurrency, r/wallstreetbets, r/Superstonk, and r/amcstock. These individuals also like to keep up to date on current events as r/news, r/politics, and r/worldnews are the sixth, eighth, and twelfth subreddits with the most activity. These individuals are interested in sports, particularly basketball and soccer. One can also conclude that the Glossier markets skews toward younger age groups, with r/teenagers being the subreddit with the seventh most activity. 

# COMMAND ----------

# MAGIC %md
# MAGIC ![glossierauthors_table](files/tables/glossierauthors_table.png)

# COMMAND ----------

# MAGIC %md
# MAGIC The table above shows a further breakdown of activity on these top subreddits by submission and comments. Interestingly, one can see that there are many more comments than submissions by these users. That and the interest in r/FreeKarma4U and r/FreeKarma4You suggest that these individuals are avid Reddit users.

# COMMAND ----------

# MAGIC %md
# MAGIC ### *2. What is the distribution of user activity across the Glossier thread?*

# COMMAND ----------

## plotting the time number of comments and submissions over time
from pyspark.sql.functions import col, asc,desc
from pyspark.sql.functions import *
import plotly.express as px

fig = px.line(join_df, x="dt_day", y="total_activity")
fig.update_layout(
    title={'text':"Total Activity In The Glossier From Jan 21 - Aug 22", 'xanchor': 'center', 'yanchor': 'top','y':0.9,'x':0.5},
    xaxis_title="Day",
    yaxis_title="Subreddit Activity (Comments + Submissions)",
    template="plotly_white",
    font=dict(
        size=12,
        color="Black"))
plot_fpath = os.path.join(PLOT_DIR, 'viz_2.html')
fig.write_html(plot_fpath)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The figure above depicts the overall activity, a combination of comments and submissions, in the Glossier subreddit thread from January 2021 to August 2022. From the figure, it is evident that the holiday months (November - January) have significantly more activity than other months. Furthermore, there appears to be some level of seasonality and cyclic behavior present; the activity tends to increase in the spring months, decrease in the summer months, and increase in the winter months. We can surmise that the increase in activity during the winter holiday months is due to an increase in purchases and retail-related discussions. Additionally, we can surmise that the activity tends to increase in the spring in preparation for the summer when new beauty products are desired. 

# COMMAND ----------

# MAGIC %md
# MAGIC ![glossierauthors_table](files/tables/glossier_subredditactivity_table.png)

# COMMAND ----------

# MAGIC %md
# MAGIC The table above shows the total number of comments and submissions in the Glossier subreddit by month. This summary table adds additional granularity to the chart above and helps put the total activity into context. As expected, there are more comments than submissions for any given month. Whenever a month experiences a change in overall activity, the submission and comment volume follow the same behavior (i.e., increase or decrease). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### *3. What are the top words mentioned across all Glossier comments?*

# COMMAND ----------

## gathering only the titles of posts as well as the comments and joining together
top_prods = glos_comm_final.select("body")
new = glos_sub_final.select("title").alias("body")

top_prods = top_prods.union(new)
## Not working with any null values
from pyspark.sql.functions import col,isnan, when, count
top_prods.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in top_prods.columns]
   ).show()

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

from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import StringType, ArrayType

## nltk stemmer creates unity between words. "likes", "liked", "likely", and "liking" would all be counted as their stem, "like"
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df_words_no_stopw.withColumn("words", stemmer_udf("words_cleaned"))


# COMMAND ----------

import pyspark.sql.functions as f
result = df_stemmed.withColumn('indword', f.explode(f.col('words_cleaned'))) \
  .groupBy('indword') \
  .count().sort('count', ascending=False) \

print('############ TOP20 Most used words in Glossier subreddit are:')
result.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image


## converting grouped by dataframe
pandas_results = result.toPandas()

glossier_mask = np.array(Image.open('/Workspace/Repos/cag199@georgetown.edu/fall-2022-project-eda-adb-project-group-16/glossier.jpg'))

d = {}
for a, x in pandas_results.values:
    d[a] = x
wordcloud = WordCloud(background_color="white", colormap='RdPu', mask=glossier_mask, width=3000, height=3000)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize=(8, 6), dpi=80)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Top Mentioned Words in the Glossier Subreddit January 2021- November 2022")
plot_fpath = os.path.join(PLOT_DIR, 'wordcloud.png')
plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC This wordcloud depicts the top used words within the r/Glossier subreddit from January 2021 to August 2022. We can gather that posters and commenters are mostly writing about a wide range of products. We see the words brow, shades, concealer, lip, skin, etc. suggesting that the conversation around products is diverse. We can also see words including don't and love, which we hope to explore more with NLP to understand if, as these words suggest, there is a wide range in sentiment about Glossier's products and business.

# COMMAND ----------

# MAGIC %md
# MAGIC ### *4. How often are cosmetic providers mentioned in the general makeup channel and how does Glossier compare?*

# COMMAND ----------

import matplotlib.pyplot as plt
from pywaffle import Waffle
data = {'Sephora': 34, 'Ulta': 25, 'Fenty': 28, 'Glossier':13}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=10, 
    values=data, 
    colors=("#00008B", "#0000FF","#5C5CFF","#FFB6C1"),
    title={'label': 'Percentage of Brand Mentions in the Makeup \nand MakeupAddiction subreddits from Jan 21 - Aug 22', 'loc': 'left',
            'fontdict': {
            'fontsize': 14
        }},
    labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1),
           'fontsize': 12},
    icons='person', icon_size=20, 
    icon_legend=True
)
fig.gca().set_facecolor('white')
fig.set_facecolor('white')
plot_fpath = os.path.join(PLOT_DIR, 'viz_3.png')
plt.savefig(plot_fpath)
plt.gcf().set_size_inches(7, 5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The "waffle plot" above depicts the Percentage of Brand Mentions in the Makeup and MakeupAddiction subreddits in the Glossier subreddit thread from January 2021 to August 2022. We can clearly see that Sephora and Ulta are mentioned far more in the Makeup and MakeupAddiction subreddits than Glossier and Fenty. We plan to expand on this analysis by linked external Google Trends data and seing if mentions in the subreddits correlate to the number of google searches of these particular makeup brands. 

# COMMAND ----------

# MAGIC %md
# MAGIC ![glossierauthors_table](files/tables/brand_mentions_table.png)

# COMMAND ----------

# MAGIC %md
# MAGIC We created the summary table above as a way to list out all possible combinations of Sephora, Ulta, Fenty and Glossier mentions in the  Makeup and MakeupAddiction subreddits. While brands are most often mentioned by themselves in a post or comment, there are some interesting combinations such as Sephora and Ulta being mentioned in the same post or comment or Fenty and Glossier being mentioned together. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### *5. How do the top products vary by Glossier subreddit versus the competitor subreddits?*

# COMMAND ----------


import pyspark.sql.functions as F

## gathering a list of their products to search for amongst the posts and comments on the subreddit
searchfor = ['bdc','lashslick','wowder','concealer','mascara','highlighter','candle','rollerball','balm dot com','balm dotcom','hoodie','tote','sweatshirt','boybrow','brow gel','generation g','scarf','clip','clips','hair clips','perfume','cloud paint','cloudpaint','blush','eyeliner','no.1 pencil','no. 1 pencil','skin tint','stretch concealer', 'pomade','lipstick','pisces','lash slick','brush','eyeshadow','pro tip','protip','solar paint','bronzer','solarpaint','monochromes', 'milky oil','brow flick','browflick','haloscope','lipgloss','lip gloss','lidstar','ls','swiss miss','le','sticker','comb','kit','body hero','futuredew','milk jelly cleanser','cleanser','jelly cleanser','after baume','after balm','moisturizer','priming moisturizer','retinol','pro-retinol','universal pro-retinol','sunscreen','zit','zit sticker','zitsticker','tin', 'tins', 'cranberry', 'cordial','olivia rodrigo'
]
check_udf = F.udf(lambda x: x if x in searchfor else 'Not_present')
## if one of these terms is present, keep the instance in a dataframe
df = top_prods.withColumn('check_presence', check_udf(F.col('body')))
df = df.filter(df.check_presence != 'Not_present').drop('check_presence')

# COMMAND ----------

from pyspark.sql.functions import *

## replacing abbreviated/ commonly mistaken product names with the correct names
## and translating the products from glossier's language to standard makeup product language so that comparison between brands will be easier
newdf = df.withColumn('body', regexp_replace('body', 'bdc', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'balm dot com', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'balm dotcom', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'balm', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'swiss miss', 'tinted chapstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cloudpaint', 'blush'))
newdf = newdf.withColumn('body', regexp_replace('body', 'cloud paint', 'blush'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pisces', 'lipstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'ls', 'mascara'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lash slick', 'mascara'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lashslick', 'mascara'))
newdf = newdf.withColumn('body', regexp_replace('body', 'browflick', 'brow pencil'))
newdf = newdf.withColumn('body', regexp_replace('body', 'brow flick', 'brow pencil'))
newdf = newdf.withColumn('body', regexp_replace('body', 'flick', 'brow pencil'))
newdf = newdf.withColumn('body', regexp_replace('body', 'wowder', 'face powder'))
newdf = newdf.withColumn('body', regexp_replace('body', 'rollerball', 'perfume'))
newdf = newdf.withColumn('body', regexp_replace('body', 'boy brow', 'brow gel'))
newdf = newdf.withColumn('body', regexp_replace('body', 'boybrow', 'brow gel'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pomade', 'brow gel'))
newdf = newdf.withColumn('body', regexp_replace('body', 'generation g', 'lipstick'))
newdf = newdf.withColumn('body', regexp_replace('body', 'gen g', 'lipstick'))
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
newdf = newdf.withColumn('body', regexp_replace('body', 'stretch concealer', 'concealer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'stretchconcealer', 'concealer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'protip', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'solarpaint', 'bronzer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'solar paint', 'bronzer'))
newdf = newdf.withColumn('body', regexp_replace('body', 'pro tip', 'eyeliner'))
newdf = newdf.withColumn('body', regexp_replace('body', 'monochromes', 'eyeshadow'))
newdf = newdf.withColumn('body', regexp_replace('body', 'milky oil', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'oil', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'serum', 'skincare'))
newdf = newdf.withColumn('body', regexp_replace('body', 'haloscope', 'highlighter'))
newdf = newdf.withColumn('body', regexp_replace('body', 'lip gloss', 'lipgloss'))
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

## creating a dataframe of the top 5 products

import pyspark.sql.functions as f
top5_product_count = newdf.withColumn('product', f.col('body')) \
  .groupBy('product') \
  .count().sort('count', ascending=False) \
  .limit(5) 


# COMMAND ----------

## adding together all of the other products that are mentioned much less frequently and adding them as a row in the dataframe called "other"
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
other_data = [("other", 39)]
schema = StructType([ \
    StructField("product",StringType(),True), \
    StructField("count", IntegerType(), True) \
  ])
other_products = spark.createDataFrame(data=other_data,schema=schema)

piechart_df = top5_product_count.union(other_products)

# COMMAND ----------

##creating the piechart and plotting
pandas_product_count = piechart_df.toPandas()
labels = 'blush', 'skincare', 'foundation', 'bronzer', 'tinted chapstick', 'other'
sizes = pandas_product_count["count"]
colors = ['#FBE1E1','#ba97aa','#FFF5FC','#FBD2D7','#E4A199', '#F2F2F2']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors=colors,
        shadow=True, startangle=90)
ax1.title.set_text('Top Mentioned Glossier Products in Glossier Subreddit 2021-2022')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The pie chart above represents the top 5 mentioned products in the r/Glossier subreddit from January 2021 to November 2022 in comparison to all products. While we can see that there are some standout products, namely their blush, skincare products, foundation, bronzer, and tinted chapstick, the largest area of the pie chart is the products that fall in the "other" category, meaning they were mentioned much less in comparison to the top 5 mentions. Further exploration into this (like adding sentiment analysis) will help us to gather more busines insights including which products should be sold together in a kit, which products may not be worth carrying, and which products we should explore developing more shade options for. 
