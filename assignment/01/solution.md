### Solution: Laptop Sales Price Prediction

Below is a step-by-step solution to the assignment on predicting laptop sales prices.

#### 1. **Data Preprocessing:**

1. **Loading the Dataset:**
   ```python
   import pandas as pd

   # Load the dataset
   data = pd.read_csv('laptop_prices.csv')
   ```

2. **Inspecting the Dataset:**
   ```python
   # Checking for missing values
   print(data.isnull().sum())

   # Checking for duplicates
   data.drop_duplicates(inplace=True)

   # Display the first few rows
   data.head()
   ```

3. **Handling Missing Values:**
   ```python
   # Example: Impute missing values in 'RAM' with the median
   data['RAM'].fillna(data['RAM'].median(), inplace=True)

   # Example: Drop rows with missing target values (Price)
   data.dropna(subset=['Price'], inplace=True)
   ```

4. **Encoding Categorical Variables:**
   ```python
   # One-Hot Encoding for categorical variables like 'Brand'
   data = pd.get_dummies(data, columns=['Brand', 'Processor', 'Storage'], drop_first=True)
   ```

5. **Normalizing/Standardizing Data (if necessary):**
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   numerical_features = ['RAM', 'Screen_Size', 'Weight']

   data[numerical_features] = scaler.fit_transform(data[numerical_features])
   ```

#### 2. **Exploratory Data Analysis (EDA):**

1. **Descriptive Statistics:**
   ```python
   # Summary statistics
   data.describe()
   ```

2. **Visualizing the Distribution of Laptop Prices:**
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   sns.histplot(data['Price'], kde=True)
   plt.title('Distribution of Laptop Prices')
   plt.show()
   ```

3. **Visualizing Relationships Between Features and Price:**
   ```python
   # Scatter plot for 'RAM' vs 'Price'
   sns.scatterplot(x=data['RAM'], y=data['Price'])
   plt.title('RAM vs Price')
   plt.show()

   # Box plot for 'Brand' vs 'Price'
   sns.boxplot(x=data['Brand'], y=data['Price'])
   plt.title('Brand vs Price')
   plt.show()
   ```

4. **Correlation Analysis:**
   ```python
   # Correlation matrix
   corr_matrix = data.corr()
   sns.heatmap(corr_matrix, annot=True)
   plt.title('Correlation Matrix')
   plt.show()
   ```

#### 3. **Feature Selection:**

1. **Correlation Analysis for Feature Selection:**
   ```python
   # Displaying highly correlated features with the target
   target_corr = corr_matrix['Price'].sort_values(ascending=False)
   print(target_corr)
   ```

2. **Feature Importance using a Random Forest Model:**
   ```python
   from sklearn.ensemble import RandomForestRegressor

   model = RandomForestRegressor()
   model.fit(data.drop(columns=['Price']), data['Price'])

   # Displaying feature importances
   feature_importances = pd.Series(model.feature_importances_, index=data.drop(columns=['Price']).columns)
   print(feature_importances.sort_values(ascending=False))
   ```

#### 4. **Model Building:**

1. **Splitting the Data:**
   ```python
   from sklearn.model_selection import train_test_split

   X = data.drop(columns=['Price'])
   y = data['Price']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

2. **Building and Training Models:**
   - **Linear Regression:**
     ```python
     from sklearn.linear_model import LinearRegression

     lr_model = LinearRegression()
     lr_model.fit(X_train, y_train)
     ```

   - **Random Forest Regression:**
     ```python
     from sklearn.ensemble import RandomForestRegressor

     rf_model = RandomForestRegressor()
     rf_model.fit(X_train, y_train)
     ```

   - **XGBoost:**
     ```python
     from xgboost import XGBRegressor

     xgb_model = XGBRegressor()
     xgb_model.fit(X_train, y_train)
     ```

#### 5. **Model Evaluation:**

1. **Evaluating Models:**
   ```python
   from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

   def evaluate_model(model, X_test, y_test):
       y_pred = model.predict(X_test)
       mae = mean_absolute_error(y_test, y_pred)
       mse = mean_squared_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)
       return mae, mse, r2

   # Evaluate Linear Regression
   lr_mae, lr_mse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
   print(f"Linear Regression - MAE: {lr_mae}, MSE: {lr_mse}, R2: {lr_r2}")

   # Evaluate Random Forest
   rf_mae, rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
   print(f"Random Forest - MAE: {rf_mae}, MSE: {rf_mse}, R2: {rf_r2}")

   # Evaluate XGBoost
   xgb_mae, xgb_mse, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)
   print(f"XGBoost - MAE: {xgb_mae}, MSE: {xgb_mse}, R2: {xgb_r2}")
   ```

#### 6. **Hyperparameter Tuning:**

1. **Tuning Random Forest with GridSearchCV:**
   ```python
   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, 30],
       'min_samples_split': [2, 5, 10]
   }

   grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
   grid_search.fit(X_train, y_train)

   # Best parameters and model
   best_params = grid_search.best_params_
   best_rf_model = grid_search.best_estimator_
   ```

2. **Evaluating the Tuned Model:**
   ```python
   tuned_rf_mae, tuned_rf_mse, tuned_rf_r2 = evaluate_model(best_rf_model, X_test, y_test)
   print(f"Tuned Random Forest - MAE: {tuned_rf_mae}, MSE: {tuned_rf_mse}, R2: {tuned_rf_r2}")
   ```

#### 7. **Model Interpretation:**

1. **Interpreting Feature Importance:**
   ```python
   import matplotlib.pyplot as plt

   # Feature importance from Random Forest
   feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
   feature_importances.sort_values(ascending=False).plot(kind='bar')
   plt.title('Feature Importance from Random Forest')
   plt.show()
   ```

#### 8. **Final Report:**

- **Summary:**
  - This report outlines the process of predicting laptop sales prices using machine learning models.
  - The models used include Linear Regression, Random Forest, and XGBoost.
  - The Random Forest model with tuned hyperparameters performed the best, with an R-squared value of X on the test data.

- **Key Insights:**
  - Features like 'Brand', 'Processor Type', and 'RAM' were found to be the most important in predicting laptop prices.
  - The Random Forest model, after hyperparameter tuning, outperformed the other models in terms of accuracy.

**Optional: Web Application:**
- A simple Flask or Django app can be developed to take user inputs (like Brand, RAM, Processor) and predict the laptop price using the trained Random Forest model.

This solution provides a complete guide to the assignment tasks and shows how to approach each step to achieve the desired outcome.