# Linear Regression Example Using a Dataset in Python

In this example, we'll use a real-world dataset to demonstrate linear regression in Python. The dataset we'll use is a commonly used one: the **Boston Housing dataset**. This dataset contains information about various features of houses in Boston, along with their prices. We'll use this data to build a linear regression model that predicts house prices based on one feature: the number of rooms in the house.

### Step 1: Import Libraries

First, let's import the necessary Python libraries.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```

Here’s a brief explanation of the libraries:
- **numpy** and **pandas** are used for handling data.
- **scikit-learn** provides the dataset, model, and tools for splitting the data and evaluating the model.
- **matplotlib** is used for visualizing the data and results.

### Step 2: Load the Dataset

We’ll load the Boston Housing dataset directly from the `scikit-learn` library.

```python
# Load the Boston Housing dataset
boston = load_boston()

# Convert it into a DataFrame for easier handling
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target  # Add the target variable (house prices)
```

The Boston dataset contains 13 features, such as crime rate, number of rooms, etc. For simplicity, we’ll use only the `RM` (average number of rooms per house) feature to predict the `PRICE` (house price).

### Step 3: Explore the Data

Let’s take a quick look at the data.

```python
print(df.head())
```

This will display the first few rows of the dataset. Here’s what the data might look like:

```
      CRIM    ZN  INDUS  CHAS    NOX  RM   AGE     DIS  RAD  TAX  PTRATIO   B  LSTAT  PRICE
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   15.3  396.9  4.98  24.0
1  0.02731  0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   17.8  396.9  9.14  21.6
```

We’re interested in the `RM` column (number of rooms) and the `PRICE` column (house prices).

### Step 4: Prepare the Data

We’ll use the `RM` column as our feature and the `PRICE` column as our target variable.

```python
# Define the feature (X) and target (y)
X = df[['RM']]  # Feature: number of rooms
y = df['PRICE']  # Target: house price
```

Next, we’ll split the data into training and testing sets. The training set will be used to train the model, and the testing set will be used to evaluate its performance.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 5: Create and Train the Model

Now, we’ll create a linear regression model and train it on the training data.

```python
# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)
```

The model will learn the relationship between the number of rooms and the house prices during training.

### Step 6: Make Predictions

After training, we can use the model to make predictions on the test data.

```python
# Make predictions on the test data
y_pred = model.predict(X_test)
```

### Step 7: Evaluate the Model

To see how well the model performs, we can calculate the Mean Squared Error (MSE) between the actual and predicted prices. A lower MSE indicates a better fit.

```python
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

### Step 8: Visualize the Results

Finally, let’s plot the regression line along with the test data to visualize how well the model fits.

```python
# Plot the test data points
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')

# Add labels and title
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression: Number of Rooms vs. Price')
plt.legend()

plt.show()
```

### Detailed Explanation of the Example

1. **Loading the Dataset:**  
   The Boston Housing dataset contains information about different houses, such as the number of rooms, age, and crime rate in the area. We convert this data into a DataFrame, making it easier to work with.

2. **Selecting Features and Target:**  
   We focus on predicting the price of a house (`PRICE`) based on one feature, the number of rooms (`RM`). This is a simple linear regression problem with one feature (also called univariate linear regression).

3. **Splitting the Data:**  
   We split the data into training and testing sets. The training set is used to build the model, while the testing set helps evaluate how well the model performs on unseen data.

4. **Training the Model:**  
   The model learns the relationship between the number of rooms and the price by adjusting its parameters (slope and intercept) to fit the training data as closely as possible.

5. **Making Predictions:**  
   Once the model is trained, we use it to predict house prices for the test data. These predictions are based on the number of rooms in the test dataset.

6. **Evaluating the Model:**  
   The Mean Squared Error (MSE) is calculated to measure how close the predicted prices are to the actual prices. A smaller MSE means the model's predictions are more accurate.

7. **Visualizing the Results:**  
   The scatter plot shows the actual test data points (in blue), and the red line represents the predicted prices from the model. If the red line closely follows the blue points, it indicates that the model fits the data well.

### Conclusion

This example demonstrates how to use linear regression to predict house prices based on the number of rooms. By following these steps in Python, you can apply linear regression to other datasets and problems. Remember, linear regression works best when there is a linear relationship between the input feature(s) and the output variable. If the relationship is more complex, you may need more advanced models, but linear regression is a great starting point for understanding machine learning.
