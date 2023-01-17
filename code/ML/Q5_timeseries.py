# Databricks notebook source
# MAGIC %md
# MAGIC Business Goal 5: We will identify how Glossier demand is affected by COVID-19 rates and forecast the relationship for the next year.

# COMMAND ----------

# MAGIC %md
# MAGIC Technical Proposal: To accomplish this, we will join external COVID-19 rate data to the Glossier subreddit activity data by day. Like above, we will identify the number of posts and comments that Glossier is mentioned in by day. We will also aggregate the total COVID-19 cases by day. To measure the effect of the COVID-19 rates on the demand, we will develop a multivariate time series ML model to forecast disease rates in conjunction with the demand. As mentioned above, this information will also be depicted on line charts to easily see the relationships and patterns between the two variables and the forecasts over time.

# COMMAND ----------

# MAGIC %md
# MAGIC Much of this code is adapted from lab 11 of Dr Purna and Dr Hickman's ANLY 560 Time Series course.

# COMMAND ----------

# importing the data
import pandas as pd

df = pd.read_csv("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/NLP/glossier_covid.csv")
df.head()

# COMMAND ----------

# checking for null values
df.isnull().sum()

# COMMAND ----------

# resource: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
from matplotlib import pyplot

# Drop first column of dataframe
df = df.iloc[: , 1:]

# datatype parsing
df["dt"] = pd.to_datetime(df["dt"], format="%Y-%m-%d")
df = df.set_index('dt')
df.head()

# COMMAND ----------

# Plotting the data
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=5, ncols=1, dpi=120, figsize=(20,8))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout();

# COMMAND ----------

# let's drop cumulative variables because those are just sums of daily new
df = df.drop(columns=["cumulative_deaths", "cumulative_cases"], axis=1)
df.head()

# COMMAND ----------

# inferring the index
# resource: https://stackoverflow.com/questions/49547245/valuewarning-no-frequency-information-was-provided-so-inferred-frequency-ms-wi
df.index = pd.DatetimeIndex(df.index.values,
                               freq=df.index.inferred_freq)

# COMMAND ----------

from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

maxlag=12

test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_table = grangers_causation_matrix(df, variables = df.columns)  
grangers_table

# COMMAND ----------

## save the csv file in the csv dir
import os
fpath = os.path.join("/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/csv/ML", "Q5_grangers_table.csv")
grangers_table.to_csv(fpath)

# COMMAND ----------

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Perform Johanson's Cointegration Test and Report Summary
def cointegration_test(df, alpha=0.05): 
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)

# COMMAND ----------

# predicting 20 values
nobs = 20
df_train, df_test = df[0:-nobs], df[-nobs:]

# Check size
print(df_train.shape)
print(df_test.shape)

# COMMAND ----------

# Perform ADFuller to test for Stationarity of given series and print report
from statsmodels.tsa.stattools import adfuller

def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")  
        
def adjust(val, length= 6): return str(val).ljust(length)

# COMMAND ----------

# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# 1st difference
df_differenced = df_train.diff().dropna()

# COMMAND ----------

# ADF Test on each column of 1st Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# Second Differencing
df_differenced = df_differenced.diff().dropna()

# COMMAND ----------

# ADF Test on each column of 2nd Differences Dataframe
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# COMMAND ----------

from statsmodels.tsa.api import VAR

model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

# COMMAND ----------

x = model.select_order(maxlags=20)
x.summary()

# COMMAND ----------

model_fitted = model.fit(13)
model_fitted.summary()

# COMMAND ----------

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(adjust(col), ':', round(val, 2))

# COMMAND ----------

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  

# COMMAND ----------

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input

# COMMAND ----------

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast.head()

# COMMAND ----------

len(df_forecast)

# COMMAND ----------

#Revert back the differencing to get the forecast to original scale.
def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

# COMMAND ----------

df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
df_results.loc[:, ['activity_count_forecast', 'daily_new_cases_forecast', 'daily_new_deaths_forecast']]

# COMMAND ----------

fig, axes = plt.subplots(nrows=3, ncols=1, dpi=150, figsize=(18, 12))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('/Workspace/Repos/cag199@georgetown.edu/fall-2022-reddit-big-data-project-project-group-16/data/plots/ML/Q5_viz1.png')

# COMMAND ----------

from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

# COMMAND ----------

print('Forecast Accuracy of: Glossier Activity')
accuracy_prod = forecast_accuracy(df_results['activity_count_forecast'].values, df_test['activity_count'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

# COMMAND ----------

print('\nForecast Accuracy of: Daily New Cases')
accuracy_prod = forecast_accuracy(df_results['daily_new_cases_forecast'].values, df_test['daily_new_cases'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

# COMMAND ----------

print('\nForecast Accuracy of: Daily New Deaths')
accuracy_prod = forecast_accuracy(df_results['daily_new_deaths_forecast'].values, df_test['daily_new_deaths'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))
