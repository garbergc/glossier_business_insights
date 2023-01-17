# Databricks notebook source
glossierauthors_comments = spark.read.parquet("dbfs:/FileStore/glossier/glossierauthors_comments")
glossierauthors_submissions = spark.read.parquet("dbfs:/FileStore/glossier/glossierauthors_submissions")

# COMMAND ----------

# removing some null values
glossierauthors_submissions = glossierauthors_submissions.filter(glossierauthors_submissions.author.isNotNull())
glossierauthors_comments = glossierauthors_comments.filter(glossierauthors_comments.author.isNotNull())

# COMMAND ----------

# first let's filter out NSFW submissions
glossierauthors_submissions = glossierauthors_submissions.filter(glossierauthors_submissions.over_18 == False)

# COMMAND ----------

# now let's filter out moderators/admins
glossierauthors_submissions = glossierauthors_submissions.filter(glossierauthors_submissions.distinguished.isNull())
glossierauthors_comments = glossierauthors_comments.filter(glossierauthors_comments.distinguished.isNull())

# COMMAND ----------

# getting top subreddits for submissions
from pyspark.sql.functions import col, asc,desc

submissions_by_subreddit = glossierauthors_submissions.groupBy("subreddit").count().orderBy(col("count"), ascending=False).collect()

# COMMAND ----------

submissions_by_subreddit[:20]

# COMMAND ----------

# for plotting
top_n = 20
top_n_subreddits = spark.createDataFrame(submissions_by_subreddit[:top_n]).toPandas()
top_n_subreddits

# COMMAND ----------

## Import data visualization packages
import plotly.express as px

fig = px.bar(top_n_subreddits, y='count', x='subreddit', text_auto='.2s',
            color_discrete_sequence =['purple']*len(top_n_subreddits),
            title="20 Most Popular Subreddits To Glossier Users By Submission Count")
fig.update_layout(plot_bgcolor = "white")
fig.show()

# COMMAND ----------

# comments by subreddit
comments_by_subreddit = glossierauthors_comments.groupBy("subreddit").count().orderBy(col("count"), ascending=False).collect()

# COMMAND ----------

comments_by_subreddit[:20]

# COMMAND ----------

top_n = 20
top_n_subreddits_comments = spark.createDataFrame(comments_by_subreddit[:top_n]).toPandas()
top_n_subreddits_comments

# COMMAND ----------

fig = px.bar(top_n_subreddits_comments, y='count', x='subreddit', text_auto='.2s',
            color_discrete_sequence =['pink']*len(top_n_subreddits_comments),
            title="20 Most Popular Subreddits To Glossier Users By Comment Count")
fig.update_layout(plot_bgcolor = "white")
fig.show()

# COMMAND ----------

# now let's join and get glossier user interests by total interactions
import pandas as pd

comments_by_subreddit_grouped = spark.createDataFrame(comments_by_subreddit).toPandas()
submissions_by_subreddit_grouped = spark.createDataFrame(submissions_by_subreddit).toPandas()
all_by_subreddit = pd.merge(comments_by_subreddit_grouped,submissions_by_subreddit_grouped, on='subreddit', how='outer')
all_by_subreddit.head()

# COMMAND ----------

# imputing any nulls as 0
all_by_subreddit_clean = all_by_subreddit.fillna(0)
all_by_subreddit_clean.head()

# COMMAND ----------

# getting total interactions
all_by_subreddit_clean['total_activity'] = all_by_subreddit_clean['count_x'] + all_by_subreddit_clean['count_y']
all_by_subreddit_clean = all_by_subreddit_clean.rename(columns={'count_x': 'comment_count', 'count_y': 'submission_count'})
all_by_subreddit_clean = all_by_subreddit_clean.sort_values("total_activity", ascending=False)
convert_dict = {'comment_count': int,
                'submission_count': int,
                'total_activity': int }  
all_by_subreddit_clean = all_by_subreddit_clean.astype(convert_dict) 
all_by_subreddit_clean.head(20)

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

# plotting the top 20 most popular subreddits to glossier authors
fig = px.bar(all_by_subreddit_clean_top100.head(20), y='total_activity', x='subreddit', text_auto='.2s',
            color_discrete_sequence =['purple']*len(all_by_subreddit_clean_top100.head(20)),
            title="20 Most Popular Subreddits To Glossier Users By Total Activity <br><sup>January 2021 - August 2022</sup>")
fig.update_layout(plot_bgcolor = "white",  xaxis_title="Subreddit Name", yaxis_title="Subreddit Activity (Comments + Posts)", title_x=0.5)
plot_fpath = os.path.join(PLOT_DIR, 'viz_1.html')
fig.write_html(plot_fpath)
fig.show()
