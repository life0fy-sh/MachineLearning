# FB Prophet 
---

**Introduction to FB Prophet**

FB Prophet is an open-source tool for forecasting time series data, especially when the data exhibits strong seasonal patterns. Developed by Facebook, Prophet is designed to handle missing data, outliers, and seasonal variations efficiently.

**Key Topics in FB Prophet:**

1. **Installation**
2. **Preparing the Data**
3. **Fitting the Prophet Model**
4. **Model Components**
5. **Forecasting**
6. **Plotting the Forecast**
7. **Model Diagnostics**
8. **Handling Holidays and Special Events**
9. **Hyperparameter Tuning**
10. **Advanced Features**

---

### 1. Installation

To use FB Prophet, you need to install the `prophet` library. You can install it using pip.

**Installation:**

```bash
pip install prophet
```

---

### 2. Preparing the Data

Prophet expects the data to have two columns: `ds` (date) and `y` (value).

**Example:**

```python
import pandas as pd

# Example time series data
data = {
    'Date': pd.date_range(start='1/1/2020', periods=24, freq='M'),
    'Value': [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 132, 150, 158, 160, 172, 181, 195, 201, 210, 222, 230, 240]
}
df = pd.DataFrame(data)

# Renaming columns to fit Prophet's requirements
prophet_df = df.rename(columns={'Date': 'ds', 'Value': 'y'})
```

---

### 3. Fitting the Prophet Model

Creating and fitting a Prophet model to the data involves initializing the model and then fitting it with the prepared data.

**Example:**

```python
from prophet import Prophet

# Initializing the model
model = Prophet()

# Fitting the model
model.fit(prophet_df)
```

---

### 4. Model Components

Prophet models the time series data using three main components:

- **Trend:** Long-term increase or decrease in the data.
- **Seasonality:** Periodic changes, such as daily, weekly, or yearly patterns.
- **Holidays:** Special events that can impact the data differently from regular seasonal patterns.

You can customize these components as needed.

**Example:**

```python
# Initializing the model with yearly seasonality
model = Prophet(yearly_seasonality=True)
model.fit(prophet_df)
```

---

### 5. Forecasting

Creating a dataframe with future dates and making predictions is straightforward with Prophet. The model will extend the time series data and provide forecasts.

**Example:**

```python
# Creating a dataframe for future dates
future = model.make_future_dataframe(periods=12, freq='M')

# Making predictions
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

---

### 6. Plotting the Forecast

Prophet provides built-in functionality to visualize the forecasted values along with historical data.

**Example:**

```python
# Plotting the forecast
fig = model.plot(forecast)
fig.show()
```

---

### 7. Model Diagnostics

Understanding the model components and checking the performance is crucial. Prophet provides functions to plot the different components of the model, such as trends and seasonality.

**Example:**

```python
# Plotting the components
fig2 = model.plot_components(forecast)
fig2.show()
```

---

### 8. Handling Holidays and Special Events

Prophet allows incorporating holidays and special events that may affect the time series differently from regular seasonal patterns.

**Example:**

```python
# Defining holidays
holidays = pd.DataFrame({
    'holiday': 'special_event',
    'ds': pd.to_datetime(['2020-12-25', '2021-12-25']),
    'lower_window': 0,
    'upper_window': 1,
})

# Adding holidays to the model
model = Prophet(holidays=holidays)
model.fit(prophet_df)

# Creating future dataframe and making predictions
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plotting the forecast with holidays
fig = model.plot(forecast)
fig.show()

# Plotting the components
fig2 = model.plot_components(forecast)
fig2.show()
```

---

### 9. Hyperparameter Tuning

Prophet allows tuning various hyperparameters to improve the model's accuracy. Some key hyperparameters include `changepoint_prior_scale` and `seasonality_prior_scale`.

**Example:**

```python
# Tuning hyperparameters
model = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10.0)
model.fit(prophet_df)

# Forecasting
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plotting the forecast
fig = model.plot(forecast)
fig.show()
```

---

### 10. Advanced Features

Prophet provides several advanced features, such as adding custom seasonalities, handling missing data, and more.

**Example:**

```python
# Adding custom seasonality
model = Prophet()
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(prophet_df)

# Forecasting
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plotting the forecast
fig = model.plot(forecast)
fig.show()
```

