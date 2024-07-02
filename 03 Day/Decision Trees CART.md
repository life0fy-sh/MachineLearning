# Decision Trees (CART) 

## Table of Contents

1. Introduction
2. Understanding Decision Trees
   - What is a Decision Tree?
   - Types of Decision Trees
3. The CART Algorithm
   - What is CART?
   - CART Components
   - How CART Works
4. Building a Decision Tree
   - Data Preparation
   - Selecting the Best Splits
   - Pruning the Tree
5. Implementation in Python
   - Using Scikit-Learn
   - Visualizing the Tree
   - Evaluating the Model
6. Practical Applications
7. Advantages and Disadvantages
8. Conclusion
9. References

## 1. Introduction

Decision trees are a popular machine learning algorithm used for both classification and regression tasks. They are easy to understand and interpret, making them a valuable tool for data analysis. In this tutorial, we will explore Decision Trees with a focus on the CART (Classification and Regression Tree) algorithm.

## 2. Understanding Decision Trees

### What is a Decision Tree?

A decision tree is a tree-like model used to make decisions based on a series of questions about the features of the data. Each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or a continuous value.

### Types of Decision Trees

- **Classification Trees**: Used when the target variable is categorical.
- **Regression Trees**: Used when the target variable is continuous.

## 3. The CART Algorithm

### What is CART?

CART stands for Classification and Regression Trees. It is a popular algorithm used to create decision trees that can be used for both classification and regression tasks.

### CART Components

- **Nodes**: Points where the data is split.
- **Branches**: Resulting sub-groups from the split.
- **Leaves**: Final nodes representing a decision or a continuous value.

### How CART Works

1. **Start at the Root**: Begin with the entire dataset.
2. **Split the Data**: Use a feature to split the data into two or more homogeneous sets.
3. **Repeat Splitting**: Continue splitting the subsets until a stopping criterion is met.
4. **Assign Outputs**: Assign a class label or a continuous value to each leaf node.

## 4. Building a Decision Tree

### Data Preparation

1. **Collect and Clean Data**: Ensure the data is complete and clean.
2. **Feature Selection**: Choose relevant features for building the model.
3. **Split Data**: Divide the data into training and testing sets.

### Selecting the Best Splits

1. **Impurity Measures**:
   - **Gini Impurity**: Used for classification trees.
   - **Entropy**: Another measure for classification trees.
   - **Mean Squared Error (MSE)**: Used for regression trees.

2. **Best Split**: Choose the feature and threshold that minimize impurity.

### Pruning the Tree

- **Pre-pruning**: Stop splitting when a certain criterion is met (e.g., maximum depth, minimum samples per leaf).
- **Post-pruning**: Remove branches from a fully grown tree to reduce complexity.

## 5. Implementation in Python

### Using Scikit-Learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('your_dataset.csv')

# Split dataset into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
clf = DecisionTreeClassifier()  # For classification
# clf = DecisionTreeRegressor()  # For regression
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # For classification
# mse = mean_squared_error(y_test, y_pred)  # For regression

print(f'Accuracy: {accuracy}')
# print(f'Mean Squared Error: {mse}')

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
plt.show()
```

### Visualizing the Tree

Using `plot_tree` from Scikit-Learn, we can visualize the structure of our decision tree.

### Evaluating the Model

For classification tasks, we use metrics like accuracy, precision, recall, and F1-score. For regression tasks, we use metrics like mean squared error (MSE) and R-squared.

## 6. Practical Applications

Decision trees are used in various domains such as:
- **Medical Diagnosis**: Classifying diseases based on symptoms.
- **Customer Segmentation**: Identifying different customer segments.
- **Credit Scoring**: Evaluating the creditworthiness of applicants.
- **Predictive Maintenance**: Predicting equipment failures.

## 7. Advantages and Disadvantages

### Advantages

- Easy to understand and interpret.
- Can handle both numerical and categorical data.
- Requires little data preparation.
- Non-parametric and versatile.

### Disadvantages

- Prone to overfitting.
- Can be unstable with small variations in data.
- Greedy algorithms can lead to suboptimal splits.
- Complex trees are hard to interpret.

## 8. Conclusion

Decision trees, especially those built using the CART algorithm, are powerful tools for both classification and regression tasks. They are easy to understand and can be visualized, making them highly interpretable. However, they require careful tuning to avoid overfitting and ensure stability.

## 9. References

- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth International Group.
- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81-106.
- Scikit-Learn documentation: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

