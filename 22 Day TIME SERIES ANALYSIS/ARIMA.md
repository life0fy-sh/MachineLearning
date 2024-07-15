
# ARIMA Model

---

**Introduction to ARIMA**

The ARIMA (AutoRegressive Integrated Moving Average) model is one of the most widely used approaches for time series forecasting. ARIMA is a combination of three components: Autoregressive (AR), Integrated (I), and Moving Average (MA).

**Key Topics in ARIMA:**

1. **Understanding ARIMA**
2. **Identifying ARIMA Components**
3. **Making the Time Series Stationary**
4. **Determining the Order of ARIMA (p, d, q)**
5. **Fitting the ARIMA Model**
6. **Model Diagnostics**
7. **Forecasting with ARIMA**

---

### 1. Understanding ARIMA

ARIMA models are denoted as ARIMA(p, d, q) where:

- **p:** Number of lag observations included in the model (AR part).
- **d:** Number of times that the raw observations are differenced (I part).
- **q:** Size of the moving average window (MA part).

**Example:**

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Example time series data
data = {
    'Date': pd.date_range(start='1/1/2020', periods=24, freq='M'),
    'Value': [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 132, 150, 158, 160, 172, 181, 195, 201, 210, 222, 230, 240]
}
df = pd.DataFrame(data)

# Plotting the data
plt.plot(df['Date'], df['Value'])
plt.title('Monthly Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```

---

### 2. Identifying ARIMA Components

Before fitting an ARIMA model, we need to understand its components: AR, I, and MA.

- **Autoregressive (AR) part:** Shows that the evolving variable of interest is regressed on its own lagged (prior) values.
- **Integrated (I) part:** Involves differencing of raw observations to make the time series stationary.
- **Moving Average (MA) part:** Shows that the regression error is a linear combination of error terms whose values occurred contemporaneously and at various times in the past.

---

### 3. Making the Time Series Stationary

A stationary time series has a constant mean and variance over time. If the time series is not stationary, it needs to be differenced.

**Example:**

```python
from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
result = adfuller(df['Value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, differencing is required
df['Value_diff'] = df['Value'].diff().dropna()

plt.plot(df['Date'], df['Value_diff'])
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Differenced Value')
plt.show()
```

---

### 4. Determining the Order of ARIMA (p, d, q)

To determine the order (p, d, q) of the ARIMA model, we use the ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots.

**Example:**

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['Value_diff'].dropna())
plot_pacf(df['Value_diff'].dropna())
plt.show()
```

---

### 5. Fitting the ARIMA Model

Once the orders (p, d, q) are determined, we can fit the ARIMA model to the data.

**Example:**

```python
# Fit ARIMA model
model = ARIMA(df['Value'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

---

### 6. Model Diagnostics

After fitting the model, we need to check the residuals (errors) to ensure they are random and normally distributed. This can be done through residual plots and statistical tests.

**Example:**

```python
residuals = model_fit.resid
plt.plot(residuals)
plt.title('Residuals')
plt.show()

from statsmodels.graphics.gofplots import qqplot
qqplot(residuals, line='s')
plt.title('QQ Plot')
plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)
```

---

### 7. Forecasting with ARIMA

Once the model is validated, it can be used for forecasting future values.

**Example:**

```python
# Forecasting future values
forecast = model_fit.forecast(steps=12)
print(forecast)

# Plotting the forecast
plt.plot(df['Date'], df['Value'], label='Original')
plt.plot(pd.date_range(start='1/1/2022', periods=12, freq='M'), forecast, label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
```

