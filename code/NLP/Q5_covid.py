# Databricks notebook source
# MAGIC %md
# MAGIC Business Goal 5: We will identify how Glossier demand is affected by COVID-19 rates and forecast the relationship for the next year.

# COMMAND ----------

# MAGIC %md
# MAGIC Technical Proposal: To accomplish this, we will join external COVID-19 rate data to the Glossier subreddit activity data by day. Like above, we will identify the number of posts and comments that Glossier is mentioned in by day. We will also aggregate the total COVID-19 cases by day. To measure the effect of the COVID-19 rates on the demand, we will develop a multivariate time series ML model to forecast disease rates in conjunction with the demand. As mentioned above, this information will also be depicted on line charts to easily see the relationships and patterns between the two variables and the forecasts over time.

# COMMAND ----------

# loading data
df_covid = spark.read.format("csv").option("header","true").load("dbfs:/FileStore/WHO_covid_data/WHO_COVID_19_global_data.csv")

# COMMAND ----------

df_covid.show()

# COMMAND ----------

## Read in glossier data 
glos_comments = spark.read.parquet("/FileStore/glossier/glossier_comments")
glos_submissions = spark.read.parquet("/FileStore/glossier/glossier_submissions")

# COMMAND ----------

# first let's filter out NSFW submissions
glos_submissions = glos_submissions.filter(glos_submissions.over_18 == False)

# now let's filter out moderators/admins
glos_comments = glos_comments.filter(glos_comments.distinguished.isNull())
glos_submissions = glos_submissions.filter(glos_submissions.distinguished.isNull())

# COMMAND ----------

# keeping only necessary columns
tokeep_comments = ["author", "created_utc", "subreddit", "score", "body"]
tokeep_submissions = ["author", "created_utc", "subreddit", "score", "title", "selftext"]

glos_comments = glos_comments.select(*tokeep_comments)
glos_submissions = glos_submissions.select(*tokeep_submissions)

# COMMAND ----------

# combining title and self text to analyze whole submission
from pyspark.sql.functions import concat_ws
glos_submissions = glos_submissions.select("author", "created_utc", "subreddit", "score", concat_ws(" ", \
                                                       glos_submissions.title,glos_submissions.selftext).alias("body"))
glos_submissions.show(5)

# COMMAND ----------

# combining datasets
df_concat = glos_submissions.union(glos_comments)
df_concat.show(5)

# COMMAND ----------

# removing any null values in the body
df_concat = df_concat.filter(df_concat.body.isNotNull())
df_concat = df_concat.filter(df_concat.body != "[removed]")

# COMMAND ----------

# let's only look at the glossier subreddit for this viz
df_concat = df_concat.filter(df_concat.subreddit == "glossier")

# COMMAND ----------

(df_concat.count(), len(df_concat.columns))

# COMMAND ----------

## Convert the columns to appropriate data type 
df_concat = df_concat.withColumn("created_utc",df_concat.created_utc.cast('timestamp'))

# COMMAND ----------

## Get the count of records for each day 
from pyspark.sql.functions import col, asc,desc
from pyspark.sql.functions import *

df_agg_glossier = df_concat.groupBy(to_date("created_utc").alias("dt")).agg(count("body").alias("activity_count")).sort("dt").toPandas()
df_agg_glossier.head()

# COMMAND ----------

df_covid.printSchema()

# COMMAND ----------

# now let's agg the WHO data by day
df_agg_covid = df_covid.groupBy(to_date("date_reported").alias("dt")).agg(sum("New_cases").cast("int").alias("daily_new_cases"), 
                                                                          sum("Cumulative_cases").cast("int").alias("cumulative_cases"), 
                                                                          sum("New_deaths").cast("int").alias("daily_new_deaths"),
                                                                          sum("Cumulative_deaths").cast("int").alias("cumulative_deaths")).sort("dt").toPandas()
df_agg_covid.head()

# COMMAND ----------

df_agg_covid.dtypes

# COMMAND ----------

df_agg_glossier.dtypes

# COMMAND ----------

# converting to datetime
import pandas as pd

df_agg_covid['dt'] =  pd.to_datetime(df_agg_covid['dt'], format='%Y-%m-%d')
df_agg_glossier['dt'] =  pd.to_datetime(df_agg_glossier['dt'], format='%Y-%m-%d')

df_agg_covid.dtypes

# COMMAND ----------

# now using an inner join to join the two datasets
final_df = pd.merge(df_agg_glossier, df_agg_covid, how ='inner', on ='dt')
final_df.head(20)

# COMMAND ----------

import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/", "glossier_covid.csv")
final_df.to_csv(fpath)

# COMMAND ----------

# visualizing
import plotly.express as px

fig = px.line(final_df, x="dt", y="daily_new_cases")
fig.update_layout(
    title={'text':"COVID New Cases From Jan 21 - Aug 22", 'xanchor': 'center', 'yanchor': 'top','y':0.9,'x':0.5},
    xaxis_title="Day",
    yaxis_title="COVID New Cases",
    template="plotly_white",
    font=dict(
        size=12,
        color="Black"))
fig.show()

# COMMAND ----------

# visualizing
import plotly.express as px

fig = px.line(final_df, x="dt", y="activity_count")
fig.update_layout(
    title={'text':"Glossier Activity From Jan 21 - Aug 22", 'xanchor': 'center', 'yanchor': 'top','y':0.9,'x':0.5},
    xaxis_title="Day",
    yaxis_title="Subreddit Activity (Comments + Posts)",
    template="plotly_white",
    font=dict(
        size=12,
        color="Black"))
fig.show()

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
fig = make_subplots(specs=[[{"secondary_y": True}]])
 
fig.add_trace(
    go.Scatter(x=final_df["dt"], y=final_df["daily_new_cases"], name="COVID Daily New Cases", mode='lines'),
    secondary_y=False)
 
# Use add_trace function and specify secondary_y axes = True.
fig.add_trace(
    go.Scatter(x=final_df["dt"], y=final_df["activity_count"], name="Glossier Activity", mode='lines'),
    secondary_y=True)
 
# Adding title text to the figure
fig.update_layout(
    title_text="Glossier Activity And COVID Cases From Jan 21 - Aug 22", plot_bgcolor = "white"
)

# Naming x-axis
fig.update_xaxes(title_text="Day")
 
# Naming y-axes
fig.update_yaxes(title_text="COVID New Cases", secondary_y=False)
fig.update_yaxes(title_text="Glossier Activity (Comments + Posts)", secondary_y=True)
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/", "covid_viz.html")
fig.write_html(fpath)
fig.show()
