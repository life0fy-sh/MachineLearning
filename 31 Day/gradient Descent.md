# gradient Descent

### **Gradient Descent: Step-by-Step Tutorial**

### **Introduction**

Gradient Descent is an iterative optimization algorithm used to find the minimum of a function. In the context of machine learning, it is used to minimize the loss function of models. The main idea is to update the parameters in the opposite direction of the gradient of the loss function with respect to the parameters.

### **Types of Gradient Descent**

1. **Batch Gradient Descent**: Uses the entire dataset to compute the gradient of the cost function.
2. **Stochastic Gradient Descent (SGD)**: Uses one sample to compute the gradient of the cost function.
3. **Mini-Batch Gradient Descent**: Uses a mini-batch (a subset of the dataset) to compute the gradient of the cost function.

### **Step-by-Step Guide**

#### **Step 1: Importing Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
```

#### **Step 2: Generate Synthetic Data**

Let's create a simple linear dataset with some noise.

```python
# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plot the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Linear Data')
plt.show()
```

#### **Step 3: Define the Model**

Define the linear model.

```python
def linear_model(X, theta):
    return X.dot(theta)
```

#### **Step 4: Define the Cost Function**

Define the mean squared error (MSE) cost function.

```python
def compute_cost(X, y, theta):
    m = len(y)
    predictions = linear_model(X, theta)
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    return cost
```

#### **Step 5: Gradient Descent Function**

Implement the Gradient Descent algorithm.

```python
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradients = X.T.dot(linear_model(X, theta) - y) / m
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history
```

#### **Step 6: Add Bias Term**

Add a column of ones to X to account for the bias term (intercept).

```python
X_b = np.c_[np.ones((X.shape[0], 1)), X]
```

#### **Step 7: Initialize Parameters and Hyperparameters**

```python
theta = np.random.randn(2, 1)
learning_rate = 0.1
iterations = 1000
```

#### **Step 8: Perform Gradient Descent**

Run the Gradient Descent algorithm.

```python
theta_best, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)
```

#### **Step 9: Plot the Cost Function**

Plot the cost function to visualize the convergence.

```python
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
```

#### **Step 10: Evaluate the Model**

Print the final parameters and visualize the fitted line.

```python
print(f"Optimal parameters: {theta_best}")

# Plot the data and the linear model
plt.scatter(X, y, label='Data')
plt.plot(X, linear_model(X_b, theta_best), color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Data and Fitted Line')
plt.show()
```

### **Conclusion**

In this tutorial, we covered:
- The concept of Gradient Descent
- Types of Gradient Descent: Batch, Stochastic, and Mini-Batch
- Step-by-step implementation of Gradient Descent for linear regression
- Visualization of the cost function and the fitted line

This approach can be extended to other machine learning models and more complex datasets. Understanding Gradient Descent is fundamental to optimizing machine learning models effectively.

