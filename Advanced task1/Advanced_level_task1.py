

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


date_rng = pd.date_range(start='2018-01-01', end='2023-12-01', freq='MS')
np.random.seed(42)
sales = np.random.randint(200, 500, size=len(date_rng)) + np.linspace(0, 100, len(date_rng))
df = pd.DataFrame({'Date': date_rng, 'Sales': sales})
df.set_index('Date', inplace=True)

print("Sample Data:")
print(df.head())


plt.figure(figsize=(8,4))
plt.plot(df['Sales'], label='Sales')
plt.title("Original Time Series")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

decomposition = seasonal_decompose(df['Sales'], model='additive', period=12)
decomposition.plot()
plt.show()


df['Moving_Avg'] = df['Sales'].rolling(window=3).mean()
plt.figure(figsize=(8,4))
plt.plot(df['Sales'], label='Original')
plt.plot(df['Moving_Avg'], label='3-Month Moving Average', color='red')
plt.title("Moving Average Smoothing")
plt.legend()
plt.show()

exp_model = ExponentialSmoothing(df['Sales'], seasonal='add', seasonal_periods=12).fit()
df['Exp_Smooth'] = exp_model.fittedvalues

plt.figure(figsize=(8,4))
plt.plot(df['Sales'], label='Original')
plt.plot(df['Exp_Smooth'], label='Exponential Smoothing', color='green')
plt.legend()
plt.show()


train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]


arima_model = ARIMA(train['Sales'], order=(1,1,1))
arima_fit = arima_model.fit()


forecast = arima_fit.forecast(steps=len(test))
test['Forecast'] = forecast.values


rmse = sqrt(mean_squared_error(test['Sales'], test['Forecast']))
print(f"RMSE: {rmse:.2f}")


plt.figure(figsize=(8,4))
plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(test['Forecast'], label='ARIMA Forecast', color='red')
plt.title("ARIMA Time Series Forecast")
plt.legend()
plt.show()
