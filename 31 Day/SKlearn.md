
# **In-Depth Hyperparameter Tuning Using SMOTE with scikit-learn**

### **Introduction**

Hyperparameter tuning is a crucial step in building efficient machine learning models. This tutorial will walk you through an in-depth guide to hyperparameter tuning using GridSearchCV and RandomizedSearchCV, with SMOTE for handling imbalanced datasets.

### **Prerequisites**

- Python installed on your system
- Basic understanding of machine learning, data preprocessing, and Python programming
- Installed libraries: `pandas`, `numpy`, `scikit-learn`, `imblearn`

### **Step 1: Importing Libraries**

First, we need to import the necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
```

### **Step 2: Generate the Dataset**

We will generate an imbalanced dataset using `make_classification`.

```python
# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=42)

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# Display first few rows
print(df.head())
```

### **Step 3: Preprocess the Data**

Separate the features and the target variable. Split the data into training and testing sets.

```python
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### **Step 4: Handle Imbalanced Data with SMOTE**

Apply SMOTE to the training data to handle class imbalance.

```python
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

### **Step 5: Train a Baseline Model**

Before hyperparameter tuning, it's helpful to train a baseline model to understand the initial performance.

```python
# Train a baseline RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### **Step 6: Hyperparameter Tuning Using GridSearchCV**

GridSearchCV exhaustively searches over a specified parameter grid.

```python
# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train_smote, y_train_smote)

# Get the best parameters
best_params_grid = grid_search.best_params_
print(f"Best parameters from GridSearchCV: {best_params_grid}")
```

### **Step 7: Hyperparameter Tuning Using RandomizedSearchCV**

RandomizedSearchCV randomly samples a specified number of candidates from a parameter space.

```python
from scipy.stats import randint

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 300),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': randint(4, 10),
    'criterion': ['gini', 'entropy']
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Fit the model
random_search.fit(X_train_smote, y_train_smote)

# Get the best parameters
best_params_random = random_search.best_params_
print(f"Best parameters from RandomizedSearchCV: {best_params_random}")
```

### **Step 8: Evaluate the Tuned Model**

Train a new model with the best parameters and evaluate its performance.

```python
# Train the model with the best parameters from GridSearchCV
best_rf_grid = RandomForestClassifier(**best_params_grid, random_state=42)
best_rf_grid.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_grid = best_rf_grid.predict(X_test)

# Evaluate the model
print("Model Performance with GridSearchCV Tuned Parameters:")
print(classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred_grid))

# Train the model with the best parameters from RandomizedSearchCV
best_rf_random = RandomForestClassifier(**best_params_random, random_state=42)
best_rf_random.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_random = best_rf_random.predict(X_test)

# Evaluate the model
print("Model Performance with RandomizedSearchCV Tuned Parameters:")
print(classification_report(y_test, y_pred_random))
print(confusion_matrix(y_test, y_pred_random))
```

### **Conclusion**

In this in-depth tutorial, we covered:
- Generating a synthetic imbalanced dataset
- Handling imbalanced data using SMOTE
- Training a baseline RandomForestClassifier model
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Evaluating the performance of the tuned models

This approach can be extended to other machine learning models and datasets, providing a robust framework for improving model performance through hyperparameter tuning.

---

