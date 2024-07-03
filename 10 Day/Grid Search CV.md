# Supervised Learning: Grid Search CV

### Table of Contents
1. Introduction to Supervised Learning
2. Understanding Grid Search CV
3. Advantages and Disadvantages of Grid Search CV
4. Key Concepts in Grid Search CV
    - Hyperparameter Tuning
    - Cross-Validation
    - Parameter Grid
5. Data Preprocessing for Grid Search CV
6. Implementing Grid Search CV
7. Evaluating Model Performance with Grid Search CV
8. Handling Imbalanced Data
9. Advanced Techniques and Best Practices
10. Case Study: Predicting Wine Quality using Grid Search CV
11. Conclusion

---

### 1. Introduction to Supervised Learning

**Supervised Learning** is a type of machine learning where the model is trained on a labeled dataset. Each training example is paired with an output label, and the goal is for the model to learn the mapping from inputs to outputs. Common supervised learning tasks include classification and regression.

---

### 2. Understanding Grid Search CV

**Grid Search Cross-Validation (Grid Search CV)** is a method for hyperparameter tuning that is used to determine the optimal hyperparameters for a given model. It involves:
- Defining a grid of possible hyperparameters.
- Training the model for each combination of hyperparameters.
- Using cross-validation to evaluate each combination.
- Selecting the combination that results in the best performance.

---

### 3. Advantages and Disadvantages of Grid Search CV

**Advantages**:
- Systematic and thorough search for the best hyperparameters.
- Can be used with any model and evaluation metric.
- Ensures the best possible performance for a given model and dataset.

**Disadvantages**:
- Computationally expensive and time-consuming, especially with large datasets and complex models.
- May not always find the global optimum if the grid is not comprehensive.

---

### 4. Key Concepts in Grid Search CV

**Hyperparameter Tuning**:
- Hyperparameters are parameters that are set before the learning process begins. Examples include the learning rate, number of trees in a random forest, and the regularization parameter in logistic regression.
- Hyperparameter tuning involves finding the set of hyperparameters that provides the best performance for the model.

**Cross-Validation**:
- Cross-validation is a technique for assessing how a model will generalize to an independent dataset. It involves partitioning the data into subsets, training the model on some subsets, and validating it on others.
- Common methods include k-fold cross-validation, stratified k-fold cross-validation, and leave-one-out cross-validation.

**Parameter Grid**:
- The parameter grid defines the hyperparameters and their possible values to be tested during the grid search.
- Example for a Support Vector Machine (SVM):
  ```python
  param_grid = {
      'C': [0.1, 1, 10, 100],
      'gamma': [1, 0.1, 0.01, 0.001],
      'kernel': ['rbf', 'poly', 'sigmoid']
  }
  ```

---

### 5. Data Preprocessing for Grid Search CV

**Data Cleaning**: Handle missing values and outliers.

**Feature Engineering**: Create new features or modify existing ones to improve model performance.

**Encoding Categorical Variables**: Convert categorical variables to numerical values using techniques like one-hot encoding.

**Splitting Data**: Divide data into training and testing sets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example data preprocessing
df = pd.read_csv('wine_dataset.csv')
df.fillna(df.mean(), inplace=True)

# Splitting data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 6. Implementing Grid Search CV

**Step-by-Step Guide**:

1. **Import Libraries**:
    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    ```

2. **Define the Parameter Grid**:
    ```python
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    ```

3. **Initialize the Model**:
    ```python
    model = SVC()
    ```

4. **Perform Grid Search CV**:
    ```python
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    ```

5. **Evaluate the Best Model**:
    ```python
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    ```

---

### 7. Evaluating Model Performance with Grid Search CV

**Common Evaluation Metrics**:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of positive identifications that are actually correct.
- **Recall**: Proportion of actual positives that are correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visual representation of the performance of the classification algorithm.

Example of calculating evaluation metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

### 8. Handling Imbalanced Data

Imbalanced datasets can bias the model towards the majority class. Techniques to handle imbalanced data include:

- **Resampling**: Oversampling the minority class or undersampling the majority class.
- **Synthetic Data Generation**: Techniques like SMOTE (Synthetic Minority Over-sampling Technique).
- **Class Weights**: Adjust the weights of the classes to balance the class distribution.

Example using class weights in Grid Search CV:

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'class_weight': ['balanced']
}
```

---

### 9. Advanced Techniques and Best Practices

**Cross-Validation**: Use k-fold cross-validation to ensure your model generalizes well to unseen data.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")
```

**Feature Selection**: Select relevant features to improve model performance.

**Scaling Techniques**: Experiment with different scaling techniques like Min-Max scaling, Standard scaling, or Robust scaling.

**Dimensionality Reduction**: Use PCA (Principal Component Analysis) or other techniques to reduce the dimensionality of the data.

---

### 10. Case Study: Predicting Wine Quality using Grid Search CV

#### Step 1: Load and Prepare Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('wine_dataset.csv')

# Splitting data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Step 2: Define the Parameter Grid

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
```

#### Step 3: Perform Grid Search CV

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

#### Step 4: Evaluate the Best Model

```python
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

---

### 11. Conclusion

Grid Search Cross-Validation (Grid Search CV) is a powerful method for hyperparameter tuning in supervised learning. It systematically searches for the best hyperparameters to improve model performance. By using cross-validation, it ensures that the model generalizes well to unseen data. Proper data preprocessing, feature selection, and handling imbalanced data are crucial for achieving the best performance with Grid Search CV.

For further reading and advanced

 techniques, refer to the [official scikit-learn documentation](https://scikit-learn.org/stable/modules/grid_search.html).

