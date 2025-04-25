# demand_forecasting.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Generate synthetic demand data
np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=100, freq='D')
demand = np.random.normal(50, 10, 100).cumsum() + 100
df = pd.DataFrame({'Date': dates, 'Demand': demand})

# Train ARIMA model
model = ARIMA(df['Demand'], order=(5, 1, 0))
fit = model.fit()
forecast = fit.forecast(steps=10)

# Calculate RMSE
train_pred = fit.predict(start=1, end=len(df))
rmse = np.sqrt(np.mean((train_pred[1:] - df['Demand'][1:]) ** 2))

# Visualize
plt.plot(df['Date'], df['Demand'], label='Actual Demand')
plt.plot(df['Date'][1:], train_pred[1:], label='Fitted')
plt.plot(pd.date_range('2025-04-11', periods=10, freq='D'), forecast, label='Forecast')
plt.title(f'Demand Forecast (RMSE: {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Demand (Orders)')
plt.legend()
plt.savefig('visualizations/demand_forecast.png')
plt.close()

if __name__ == "__main__":
    print(f"Demand Forecast RMSE: {rmse:.2f}")
    print(f"10-Day Forecast: {forecast.values}")