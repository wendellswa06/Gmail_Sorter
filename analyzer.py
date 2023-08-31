import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

monthly_raw = pd.read_csv('CPI.csv', parse_dates=True, index_col=0)
# print(monthly_raw)
print(monthly_raw.dtypes)
monthly_raw.DATE = pd.to_datetime(monthly_raw.DATE)

# print(monthly_raw['DATE'].unique())

# Create a copy
monthly_df = monthly_raw.copy()
print(monthly_df)
#Macroeconomic Indicator Trend

monthly_df['cpi_pct_mom'] = round((monthly_df['General Index'].pct_change().fillna(0)) * 100, 2)
monthly_df['cpi_pct_yoy'] = round((monthly_df['General Index'].pct_change(12).fillna(0)) * 100, 2)

print(monthly_df['cpi_pct_mom'], monthly_df['cpi_pct_yoy'])
title_origin = ['Miscellaneous goods and services', 'Insurance', 'Accommodation services', 'Education', 'Recreation', 'Information and communication', 'Transport', 'Health', 'Furnishings', 'Housing', 'Clothing and footwear', 'Tobacco', 'Food and beverages', 'General Index', 'CPI % Change MOM', 'CPI % Change YOY']
# id = 10
# interval = 3
# monthly_df.iloc[:, 1:3].plot(kind = 'line', subplots = True, figsize = (14, 14), 
#                               title = title_origin[0:2],
#                               legend=False,
#                               layout = (1, 2),
#                               sharex=True,
#                               sharey=['midnightblue', 'steelblue', 'dodgerblue', 'slateblue','mediumblue','darkslateblue','red','salmon','brown','maroon','tomato'])
# plt.suptitle('5 year Macroeconomic Indicators for industry in Dubai', fontsize=22)
# plt.show()

# Core CPI trend by Month and Quarter

monthly_df['year'] = monthly_df['DATE'].apply(lambda x : x.year)
monthly_df['quarter'] = monthly_df['DATE'].apply(lambda x : x.quarter)
monthly_df['month'] = monthly_df['DATE'].apply(lambda x : x.month)

######################box###########################
fig = px.box(monthly_df[12:], x = 'month', y = 'cpi_pct_mom', points = 'all', template = 'presentation',)
fig.update_layout(xaxis = dict(tickmode = 'linear'))
# fig.show()

fig = px.box(monthly_df[12:], x='quarter', y ='cpi_pct_yoy', points = 'all', template='presentation')
# fig.show()


####################################################

######################bar###########################
fig = px.bar(
    data_frame = monthly_df.groupby(['month']).std().reset_index(), 
    x = 'month', 
    y = 'cpi_pct_yoy', text = 'cpi_pct_yoy'
).update_traces(texttemplate = '%{text:0.3f}', textposition = 'outside').update_xaxes(nticks = 13)
# fig.show()
fig = px.bar(
    data_frame=monthly_df.groupby(['quarter']).std().reset_index(), 
    x="quarter", 
    y="cpi_pct_yoy", text="cpi_pct_yoy").update_traces(texttemplate='%{text:0.3f}', textposition='outside').update_xaxes(nticks=5)
# fig.show()

fig = px.bar(
    data_frame = monthly_df.groupby(['month']).std().reset_index(), 
    x = 'month', 
    y = 'Food and beverages', text = 'Food and beverages'
).update_traces(texttemplate = '%{text:0.3f}', textposition = 'outside').update_xaxes(nticks = 13)
fig.show()
fig = px.bar(
    data_frame = monthly_df.groupby(['month']).std().reset_index(), 
    x = 'month', 
    y = 'Education', text = 'Education'
).update_traces(texttemplate = '%{text:0.3f}', textposition = 'outside').update_xaxes(nticks = 13)
fig.show()


####################################################

#Forecasting  Inflation

df_cpi = monthly_raw.set_index('DATE')

################################################ARIMA Implementation############################################

#################################################Time Series Decomposition######################################
# df_cpi['General Index'].plot()
# seasonal_decompose(df_cpi['General Index'], model = 'additive').plot()
# plt.show()

#################################################Splitting the Data#############################################
split_point = len(df_cpi) - 12
train, test = df_cpi[0:split_point], df_cpi[split_point:]
print('Training dataset: %d, Test dataset: %d' % (len(train), len(test)))
# plt.plot(train['General Index'])
# plt.plot(test['General Index'])
# plt.show()

#################################################Take first differences##############################################

diff = train['General Index'].diff()
# plt.plot(diff)
# plt.show()



#################################################Augmented Dickey-Fuller test##############################################
diff = diff.dropna()
def adf_test(df):
    result = adfuller(df.values, autolag = 'AIC')
    # print(result)
    if result[1] > 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")

adf_test(diff)
###########################################################################################################################

############################################################Plot ACF and PACF##############################################


# plot_pacf(diff.values).show()
# plot_acf(diff.values).show()
# plt.show()
###########################################################################################################################

########################################################Building the model#################################################
arima_model = ARIMA(np.log(train['General Index']), order = (1,1,1))

arima_fit = arima_model.fit()
print(arima_fit.summary())

############################################################Forecast#####################################################
forecast = arima_fit.forecast(steps=12)
forecast = np.exp(forecast)

# plt.plot(forecast, color = 'red')


###############################EvaLuating the ARIMA model with RMSE and Mean of observed y - predicted y#####################

mse = mean_squared_error(test['General Index'].values, forecast[:12])
print('MSE: ', mse)
mae = mean_absolute_error(test['General Index'].values, forecast[:12])
print('MAE: ', mae)
model_error = test['General Index'] - forecast
print('Mean Model Error: ', model_error.mean())

#############################################Forecasting################################################
forecast = arima_fit.forecast(steps=12)
forecast = np.exp(forecast)
# plt.plot(forecast, color = 'red')
# plt.show()

arima_model = ARIMA(np.log(test['General Index']), order = (1,1,1),freq=test.index.inferred_freq)

arima_fit = arima_model.fit()



pct_chg = ((forecast[-1] - df_cpi.iloc[-12]['General Index'])/df_cpi.iloc[-12]['General Index']) * 100
print('The forecasted Dubai Consumer Price Index (CPI) YoY is ' , round(pct_chg,2))
print('The CPI value for the month January 2023 predicted by ARIMA model is', round(forecast[0],2))