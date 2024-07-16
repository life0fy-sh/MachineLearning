

# **Hyperparameter Tuning for Advanced Machine Learning Models Using SMOTE**

### **Introduction**

In this tutorial, we will cover the steps to perform hyperparameter tuning for machine learning models on imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique). We will use Python and its popular libraries like scikit-learn for this purpose.

### **Prerequisites**

- Python installed on your system
- Basic understanding of machine learning and data preprocessing
- Installed libraries: `pandas`, `numpy`, `scikit-learn`, `imblearn`

### **Step 1: Importing Libraries**

First, we need to import the necessary libraries.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
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

### **Step 5: Train a Machine Learning Model**

Let's train a RandomForestClassifier as our machine learning model.

```python
# Train a RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### **Step 6: Hyperparameter Tuning**

Use GridSearchCV to tune the hyperparameters of the RandomForestClassifier.

```python
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_smote, y_train_smote)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
```

### **Step 7: Evaluate the Tuned Model**

Train a new model with the best parameters and evaluate its performance.

```python
# Train the model with best parameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_best = best_rf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
```

### **Conclusion**

In this tutorial, we covered how to handle imbalanced datasets using SMOTE and how to perform hyperparameter tuning using GridSearchCV for a RandomForestClassifier. This approach can be applied to other machine learning models and datasets as well.

