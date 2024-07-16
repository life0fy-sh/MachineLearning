
Step 1: Import the necessary libraries
First, you need to import the required libraries such as NumPy, pandas, and scikit-learn.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

Step 2: Load your dataset 
Next, load your dataset into a pandas DataFrame. Let's assume we have a CSV file named "data.csv" with two columns - "X" and "y".

```python
data = pd.read_csv("data.csv")
X = data["X"].values.reshape(-1, 1)
y = data["y"].values.reshape(-1, 1)
```

Step 3: Split the dataset into training and testing sets 
Split the dataset into training and testing sets using `train_test_split` from scikit-learn. This allows us to evaluate the model on unseen data.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
```

Step4: Perform Lasso regression 
Instantiate a Lasso regression model from scikit-learn's `Lasso` class and fit it on the training data.

```python    
lasso_reg = Lasso(alpha=.01) # set regularization parameter alpha (hyperparameter tuning may be required)
lasso_reg.fit(X_train,y_train) # fit the model on training data

# Predict values for test set using trained model:
predictions_lasso = lasso_reg.predict(X_test)

# Evaluate the model:
mse_lasso = mean_squared_error(y_true=y_test,y_pred=predictions_lasso)

print("Mean Squared Error (Lasso): ", mse_lasso)
```

Step5: Perform Ridge regression  
Instantiate a Ridge regression model from scikit-learn's `Ridge` class and fit it on the training data.

```python    
ridge_reg = Ridge(alpha=.01) # set regularization parameter alpha (hyperparameter tuning may be required)
ridge_reg.fit(X_train,y_train) # fit the model on training data

# Predict values for test set using trained model:
predictions_ridge = ridge_reg.predict(X_test)

# Evaluate the model:
mse_ridge = mean_squared_error(y_true=y_test,y_pred=predictions_ridge)

print("Mean Squared Error (Ridge): ", mse_ridge)
```

These are the basic steps for performing Lasso and Ridge regression using scikit-learn in Python. Remember that hyperparameter tuning may be required to find the optimal values for the regularization parameter alpha.