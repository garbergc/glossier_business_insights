# Databricks notebook source
# MAGIC %md
# MAGIC Business Goal 4: We will predict when Glossier demand will be highest over the next year and contrast that with competitor forecasts.

# COMMAND ----------

# MAGIC %md
# MAGIC Technical Proposal: Leveraging regular expressions and searching techniques, we will identify the number of posts and comments that Glossier is mentioned in by day.  Using the total activity by day as demand, we will perform exploratory data analysis to identify if seasonality is present and what type of ML model should be leveraged (e.g., ARMA, SARIMA, etc.) for univariate time series forecasting. This information will be depicted on line charts to easily see demand fluctuations, seasonality, and forecasts over time. The output of this analysis will give us insight into how Glossier should manage their inventory. This same analysis will be repeated for Ulta Beauty (competitor) to have a benchmark for comparison.

# COMMAND ----------

# MAGIC %md
# MAGIC Glossier Activity

# COMMAND ----------

# loading in the data
import pandas as pd
join_df = pd.read_csv('/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/EDA/viz_2.csv')
join_df.head()

# COMMAND ----------

# creating bool series True for NaN values
bool_series = pd.isnull(join_df["total_activity"])
   
# filtering data
join_df[bool_series]

# COMMAND ----------

# let's imput this as the average of the days near it
row_index_list = [467, 468, 470, 471]

imp_mean = join_df["total_activity"].iloc[row_index_list].mean(axis=0)
imp_mean

# COMMAND ----------

# imputing nulls as zeros
join_df["total_activity"] = join_df["total_activity"].fillna(imp_mean)
join_df["total_activity"].isnull().sum()

# COMMAND ----------

# resource: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

# Drop first column of dataframe
join_df = join_df.iloc[: , 1:]

# datatype parsing
join_df["dt_day"] = pd.to_datetime(join_df["dt_day"], format="%Y-%m-%d")
join_df = join_df.set_index('dt_day')
join_df.head()

# COMMAND ----------

# viewing the rolling mean and rolling std
# resource: https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7
import matplotlib.pyplot as plt

rolling_mean = join_df.rolling(window = 10).mean()
rolling_std = join_df.rolling(window = 10).std()

plt.figure(figsize=(20,8))
plt.plot(join_df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC One can see that the rolling mean and std do not really increase over time so the series is relatively stationary.

# COMMAND ----------

# let's check the Augmented Dickey-Fuller Test to be sure
from statsmodels.tsa.stattools import adfuller

result = adfuller(join_df['total_activity'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))

# COMMAND ----------

# MAGIC %md
# MAGIC Here, one can see that the p value is lower than 0.05 so we can conclude that the series is stationary, meaning not increasing or decreasing steadily over time.

# COMMAND ----------

# converting to series
series = join_df.squeeze()
series.head()

# COMMAND ----------

# checking the autocorrelation plot
autocorrelation_plot(join_df)
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the autocorrelation plot above we can see that there is a positive correlation with the first ~20 lags.

# COMMAND ----------

# let's also check the ACF and PACF plots
# resource: https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
plot_acf(series, ax=ax1)

ax2 = f.add_subplot(122)
plot_pacf(series, ax=ax2)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC However, one can see from the ACF plot above that the data is highly correlated with the observations around it which is not ideal for ARIMA so let's use some differencing.

# COMMAND ----------

# resource: https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
ax1.set_title("First Order Differencing")
ax1.plot(series.diff())

ax2 = f.add_subplot(122)
plot_acf(series.diff().dropna(), ax=ax2)
plt.show()

# COMMAND ----------

# plotting both plots for first order differencing

series_diff = series.diff()

f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
plot_acf(series_diff.dropna(), ax=ax1)

ax2 = f.add_subplot(122)
plot_pacf(series_diff.dropna(), ax=ax2)
plt.show()

# COMMAND ----------

# now let's check for seasonality
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_data = seasonal_decompose(series_diff.dropna(), model="additive")
decompose_data.plot()
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There is clear seasonality as seen in the plot above so an SARIMA model will most likely be the best fit model.

# COMMAND ----------

# inferring the index
# resource: https://stackoverflow.com/questions/49547245/valuewarning-no-frequency-information-was-provided-so-inferred-frequency-ms-wi
series.index = pd.DatetimeIndex(series.index.values,
                               freq=series.index.inferred_freq)

# COMMAND ----------

# let's use grid search to find the optimal SARIMA model
# grid search function adapted from: https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/
import itertools
import statsmodels.api as sm

def sarima_grid_search(y,seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
    
    scores = {}
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                
                mystr = str(param+param_seasonal)
                scores[mystr] = [results.aic, results.bic]
                
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal
            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))
    return param_mini, param_seasonal_mini, mini, scores

# COMMAND ----------

# the lag plot above suggests a correlation with the 20 previous lags so we will use a seasonal period of 20
# selecting best model based on AIC score
param1, param2, AIC, for_table = sarima_grid_search(series,20)

# COMMAND ----------

# getting info for table
for_table_csv = pd.DataFrame(for_table.items(), columns=['Order', 'AIC_BIC'])
for_table_csv[['AIC','BIC']] = pd.DataFrame(for_table_csv.AIC_BIC.tolist(), index= for_table_csv.index)
for_table_csv = for_table_csv.drop("AIC_BIC", axis = 1)
for_table_csv.head()

# COMMAND ----------

# saving as csv
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/ML", "Q4_GlossierSARIMA_scores.csv")
for_table_csv.to_csv(fpath)
for_table_csv.head(10)

# COMMAND ----------

# now let's fit the optimal model
model = sm.tsa.statespace.SARIMAX(series, order=param1, seasonal_order=param2, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())

# COMMAND ----------

# getting plots of residuals
f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
ax1.set_title("Residuals")
residuals = pd.DataFrame(model_fit.resid)
ax1.plot(residuals)

ax2 = f.add_subplot(122)
ax2.set_title("Density Of Residuals")
residuals.plot(kind='kde', ax=ax2)
plt.show()

# COMMAND ----------

# summary stats of residuals
print(residuals.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC The plots of the residuals above show that the residuals are centered around ~0 and suggest that this is be a best fit model for the data.

# COMMAND ----------

# now let's take a look at the predictions
# resource: https://stackoverflow.com/questions/73112516/arimaresults-object-has-no-attribute-plot-predict-error
from statsmodels.graphics.tsaplots import plot_predict

fig, ax = plt.subplots(figsize=(20,8))
ax = series.plot(ax=ax)
ax.set_title("SARIMA Predictions Of Glossier Activity")
plt.xlabel('Day', fontsize=12)
plt.ylabel('Subreddit Activity (Comments + Submissions)', fontsize=12)
plot_predict(model_fit, ax=ax)
plt.ylim(0,1300)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC One can see that the model's forecasts are quite accurate.

# COMMAND ----------

# now let's obtain the forecasts
# resource: https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# let's predict 30 days into the future
pred_uc = model_fit.get_forecast(steps=30)

pred_ci = pred_uc.conf_int()

ax = series.plot(label='Observed', figsize=(20, 8))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Subreddit Activity (Comments + Submissions)')
ax.set_title("SARIMA Forecast Of Glossier Subreddit Activity")

plt.legend()
plt.savefig('/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/ML/Q4_viz1.png')
plt.show()



# COMMAND ----------

# Produce the forcasted tables 
# resource: https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/

pm = pred_uc.predicted_mean.reset_index()
pm.columns = ['Date','Predicted_Mean']
pci = pred_ci.reset_index()
pci.columns = ['Date','Lower Bound','Upper Bound']
final_table = pm.join(pci.set_index('Date'), on='Date')

## save the csv file in the csv dir
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/ML", "Q4_GlossierSARIMA.csv")
final_table.to_csv(fpath)
final_table.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Ulta Activity

# COMMAND ----------

# reading in data
competitor_comments = spark.read.parquet("dbfs:/FileStore/glossier/competitor_comments")
competitor_submissions = spark.read.parquet("dbfs:/FileStore/glossier/competitor_submissions")

# COMMAND ----------

# cleaning and grouping in the same way as the Glossier data for comparison
## Remove all uneccessary columns from the data frame 
cols = ("author_cakeday","author_flair_css_class","author_flair_text","permalink","stickied","gilded","distinguished","can_gild","retrieved_on","edited")
competitor_comments = competitor_comments.drop(*cols)
competitor_comments.show(5)

# COMMAND ----------

## Convert the columns to appropriate data type 
competitor_comments = competitor_comments.withColumn("created_utc",competitor_comments.created_utc.cast('timestamp'))

# COMMAND ----------

## Convert the data frame into a view to run SQL on 
competitor_comments.createOrReplaceTempView("comp_comm_vw")

## Get a count of all the missing values 
missing_comp_comm = spark.sql("select \
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
                    from comp_comm_vw")

## Show and view the results 
missing_comp_comm.show()


# COMMAND ----------

## Drop all of the uneccessary columns
cols = ("whitelist_status","url","thumbnail_width","thumbnail_height","thumbnail","third_party_tracking_2","third_party_tracking","third_party_trackers","suggested_sort",
       "secure_media_embed", "retrieved_on", "promoted_url", "parent_whitelist_status", "link_flair_text", "link_flair_css_class", "imp_pixel", "href_url", "gilded", "embed_url", 
       "author_flair_css_class", "author_cakeday","adserver_imp_pixel", "adserver_click_url", "secure_media_embed", "secure_media", "post_hint", "permalink", "original_link", 
       "mobile_ad_url", "embed_type", "domain_override", "domain", "author", "preview", "author_flair_text", "edited", "crosspost_parent_list", "media", "media_embed")
competitor_submissions = competitor_submissions.drop(*cols)
competitor_submissions.columns

# COMMAND ----------

## Convert the data frame into a view to run SQL on 
competitor_submissions.createOrReplaceTempView("comp_sub_vw")

## Part 1: 
## Get a count of all the missing values 
missing_comp_sub1= spark.sql("select sum(case when author_id is null then 1 else 0 end) as author_id, \
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
                    from comp_sub_vw")

## Show and view the results 
missing_comp_sub1.show()

# COMMAND ----------

## Get a count of all the missing values 
missing_comp_sub2 = spark.sql("select sum(case when num_comments is null then 1 else 0 end) as num_comments,\
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
                    from comp_sub_vw")

## Show and view the results 
missing_comp_sub2.show()

# COMMAND ----------

## Convert the columns to appropriate data type 
competitor_submissions = competitor_submissions.withColumn("created_utc",competitor_submissions.created_utc.cast('timestamp'))

# COMMAND ----------

competitor_submissions.count()

# COMMAND ----------

competitor_comments.count()

# COMMAND ----------

# getting ulta submissions and grouping by day
from pyspark.sql.functions import col, asc,desc
from pyspark.sql.functions import *

## Filter to the glossier subreddit only 
submissions_by_subreddit = competitor_comments.filter((col("subreddit") == "Ulta"))
## Get the count of records for each day 
submissions_by_subreddit = submissions_by_subreddit.groupBy(to_date("created_utc").alias("dt_day")).agg(count("id").alias("comment_count"))

## Filter to the glossier subreddit only 
submissions_by_subreddit2 = competitor_submissions.filter((col("subreddit") == "Ulta"))
## Get the count of records for each day 
submissions_by_subreddit2 = submissions_by_subreddit2.groupBy(to_date("created_utc").alias("dt")).agg(count("id").alias("submission_count"))

# COMMAND ----------

## We are using SQL so need to convert them into views 
submissions_by_subreddit2.createOrReplaceTempView("x_vw")
submissions_by_subreddit.createOrReplaceTempView("y_vw")

# joining
join_df = submissions_by_subreddit2.join(submissions_by_subreddit,submissions_by_subreddit2.dt ==  submissions_by_subreddit.dt_day,"outer")
join_df = join_df.drop("dt")
join_df.show()

# COMMAND ----------

# getting total activity
join_df = join_df.withColumn('total_activity', join_df.submission_count + join_df.comment_count)
join_df = join_df.drop("submission_count","comment_count")
join_df = join_df.toPandas()
join_df.head(10)

# COMMAND ----------

# saving this dataset
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/ML", "Ulta_Activity.csv")
join_df.to_csv(fpath)

# COMMAND ----------

# looking for nulls
bool_series = pd.isnull(join_df["total_activity"])
   
# filtering data
join_df[bool_series]

# COMMAND ----------

# let's imput this as the average of the days near it
row_index_list = [40, 41, 43, 44]

imp_mean = join_df["total_activity"].iloc[row_index_list].mean(axis=0)
imp_mean

# COMMAND ----------

# imputing nulls as zeros
join_df["total_activity"] = join_df["total_activity"].fillna(imp_mean)
join_df["total_activity"].isnull().sum()

# COMMAND ----------

# datatype parsing
join_df["dt_day"] = pd.to_datetime(join_df["dt_day"], format="%Y-%m-%d")
join_df = join_df.set_index('dt_day')
join_df.head()

# COMMAND ----------

# viewing the rolling mean and rolling std
# resource: https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7

rolling_mean = join_df.rolling(window = 10).mean()
rolling_std = join_df.rolling(window = 10).std()

plt.figure(figsize=(20,8))
plt.plot(join_df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Unlike in the Glossier activity data, we see a slight upward trend over time in the Ulta subreddit activity data. The data does not look stationary but let's use an ADF test to be sure.

# COMMAND ----------

result = adfuller(join_df['total_activity'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))

# COMMAND ----------

# MAGIC %md
# MAGIC The p value is greater than 0.05 so we know that the data is not stationary. We must induce stationarity.

# COMMAND ----------

# converting to series
series = join_df.squeeze()
series.head()

# COMMAND ----------

f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
plot_acf(series, ax=ax1)

ax2 = f.add_subplot(122)
plot_pacf(series, ax=ax2)
plt.show()

# COMMAND ----------

# inducing stationarity with differencing 
# resource: https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
ax1.set_title("First Order Differencing")
ax1.plot(series.diff())

ax2 = f.add_subplot(122)
plot_acf(series.diff().dropna(), ax=ax2)
plt.show()

# COMMAND ----------

# plotting both plots for first order differencing

series_diff = series.diff()

f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
plot_acf(series_diff.dropna(), ax=ax1)

ax2 = f.add_subplot(122)
plot_pacf(series_diff.dropna(), ax=ax2)
plt.show()

# COMMAND ----------

decompose_data = seasonal_decompose(series_diff.dropna(), model="additive")
decompose_data.plot()
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Again, we see that there is significant seasonality in this data. Let's fit it using a SARIMA model.

# COMMAND ----------

# inferring the index
# resource: https://stackoverflow.com/questions/49547245/valuewarning-no-frequency-information-was-provided-so-inferred-frequency-ms-wi
series.index = pd.DatetimeIndex(series.index.values,
                               freq=series.index.inferred_freq)

# COMMAND ----------

# similar to glossier we will use a seasonal period of 20
# selecting best model based on AIC score
param1, param2, AIC, for_table2 = sarima_grid_search(series, 20)

# COMMAND ----------

# getting info for table
for_table2_csv = pd.DataFrame(for_table2.items(), columns=['Order', 'AIC_BIC'])
for_table2_csv[['AIC','BIC']] = pd.DataFrame(for_table2_csv.AIC_BIC.tolist(), index= for_table2_csv.index)
for_table2_csv = for_table2_csv.drop("AIC_BIC", axis = 1)
for_table2_csv.head()

# COMMAND ----------

## save the csv file in the csv dir
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/ML", "Q4_UltaSARIMA_scores.csv")
for_table2_csv.to_csv(fpath)
for_table2_csv.head(10)

# COMMAND ----------

# now let's fit the optimal model
model = sm.tsa.statespace.SARIMAX(series, order=param1, seasonal_order=param2, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())

# COMMAND ----------

# getting plots of residuals
f = plt.figure(figsize=(20,8))
ax1 = f.add_subplot(121)
ax1.set_title("Residuals")
residuals = pd.DataFrame(model_fit.resid)
ax1.plot(residuals)

ax2 = f.add_subplot(122)
ax2.set_title("Density Of Residuals")
residuals.plot(kind='kde', ax=ax2)
plt.show()

# COMMAND ----------

# summary stats of residuals
print(residuals.describe())

# COMMAND ----------

# now let's take a look at the predictions
# resource: https://stackoverflow.com/questions/73112516/arimaresults-object-has-no-attribute-plot-predict-error

fig, ax = plt.subplots(figsize=(20,8))
ax = series.plot(ax=ax)
ax.set_title("SARIMA Predictions Of Ulta Activity")
plt.xlabel('Day', fontsize=12)
plt.ylabel('Subreddit Activity (Comments + Submissions)', fontsize=12)
plot_predict(model_fit, ax=ax)
plt.ylim(-100, 500)
plt.show()

# COMMAND ----------

# now let's obtain the forecasts
# resource: https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/

plt.rcParams.update({'font.size': 14})

# let's predict 30 days into the future
pred_uc = model_fit.get_forecast(steps=30)

pred_ci = pred_uc.conf_int()

ax = series.plot(label='Observed', figsize=(20, 8))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Subreddit Activity (Comments + Submissions)')
ax.set_title("SARIMA Forecast Of Ulta Activity")

plt.legend()
plt.savefig('/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/ML/Q4_viz2.png')
plt.show()

# COMMAND ----------

# Produce the forcasted tables 
# resource: https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/

pm = pred_uc.predicted_mean.reset_index()
pm.columns = ['Date','Predicted_Mean']
pci = pred_ci.reset_index()
pci.columns = ['Date','Lower Bound','Upper Bound']
final_table = pm.join(pci.set_index('Date'), on='Date')

## save the csv file in the csv dir
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/ML", "Q4_UltaSARIMA.csv")
final_table.to_csv(fpath)
final_table.head(10)
