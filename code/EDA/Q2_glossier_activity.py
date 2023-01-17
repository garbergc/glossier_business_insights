# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Cleaning

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

## View the final data frame 
glos_sub_final.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA
# MAGIC 
# MAGIC 1. How does the number of comments in the glossier subreddit by day fluctuate over time? 

# COMMAND ----------

## Import all necessary libraries
from pyspark.sql.functions import col, asc,desc
from pyspark.sql.functions import *

## Filter to the glossier subreddit only 
submissions_by_subreddit = glos_comm_final.filter((col("subreddit") == "glossier"))
## Get the count of records for each day 
submissions_by_subreddit = submissions_by_subreddit.groupBy(to_date("created_utc").alias("dt_day")).agg(count("id").alias("comment_count"))

## Filter to the glossier subreddit only 
submissions_by_subreddit2 = glos_sub_final.filter((col("subreddit") == "glossier"))
## Get the count of records for each day 
submissions_by_subreddit2 = submissions_by_subreddit2.groupBy(to_date("created_utc").alias("dt")).agg(count("id").alias("submission_count"))

## We are using SQL so need to convert them into views 
submissions_by_subreddit2.createOrReplaceTempView("x_vw")
submissions_by_subreddit.createOrReplaceTempView("y_vw")

# COMMAND ----------

join_df = submissions_by_subreddit2.join(submissions_by_subreddit,submissions_by_subreddit2.dt ==  submissions_by_subreddit.dt_day,"outer")
join_df = join_df.drop("dt")
join_df.show()



# COMMAND ----------

#making a summary table 
#aggregate by month 
from pyspark.sql.functions import year
from pyspark.sql.functions import to_date
table_1=join_df
#table_1.show()
#table_1.printSchema()
 
table_2 = table_1.withColumn('dt_month',month(table_1.dt_day))
table_2 = table_2.withColumn('dt_year',year(table_1.dt_day))
table_3 = table_2.drop("dt_day")
table_4 = table_3.withColumn("dt_month_year",concat(col("dt_year"),lit('-'),col("dt_month")))
#table_4.show()
table_4a=table_4.select("comment_count","dt_month_year")
table_4b=table_4.select("submission_count","dt_month_year")
table_5a = table_4a.groupBy('dt_month_year').agg(sum("comment_count").alias("comment_count"))
table_5b = table_4b.groupBy('dt_month_year').agg(sum("submission_count").alias("submission_count"))
#table_5a.show()
#table_5b.show()
#Join these 2 tables into one table 
table_6 = table_5b.join(table_5a,table_5b.dt_month_year == table_5a.dt_month_year,"inner").drop(table_5b.dt_month_year)
table_6= table_6.select(col("dt_month_year"),to_date(col("dt_month_year"),"yyyy-M").alias("date"),col("comment_count"),col("submission_count")) 
#table_6.show()
#table_6.printSchema()
table_7= table_6.sort("date")
#table_7.show()
table_8=table_7.toPandas()
table_9=table_8[['dt_month_year', 'comment_count', 'submission_count']]
table_9
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "summary_table_2.csv")
table_9.to_csv(fpath)

# COMMAND ----------

join_df = join_df.withColumn('total_activity', join_df.submission_count + join_df.comment_count)
join_df = join_df.drop("submission_count","comment_count")
join_df = join_df.toPandas()
fpath = os.path.join(CSV_DIR, "viz_2.csv")
join_df.to_csv(fpath)

# COMMAND ----------

from pyspark.sql.functions import col, asc,desc
from pyspark.sql.functions import *
import plotly.express as px

fig = px.line(join_df, x="dt_day", y="total_activity")
fig.update_layout(
    title={'text':"Total Activity In The Glossier From Jan 21 - Aug 22", 'xanchor': 'center', 'yanchor': 'top','y':0.9,'x':0.5},
    xaxis_title="Day",
    yaxis_title="Subreddit Activity (Comments + Posts)",
    template="plotly_white",
    font=dict(
        size=12,
        color="Black"))
plot_fpath = os.path.join(PLOT_DIR, 'viz_2.html')
fig.write_html(plot_fpath)
fig.show()

# COMMAND ----------


