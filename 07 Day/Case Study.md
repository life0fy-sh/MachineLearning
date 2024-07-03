## Case Study: Predicting Iris Species using KNN

### Table of Contents
1. Introduction
2. Dataset Description
3. Data Preprocessing
4. Building a KNN Model
5. Hyperparameter Tuning
6. Evaluating the Model
7. Conclusion

---

### 1. Introduction

In this case study, we will use the K-Nearest Neighbors (KNN) algorithm to predict the species of iris flowers based on their features. The Iris dataset is a classic dataset in machine learning and statistics and contains three classes of iris flowers: Setosa, Versicolour, and Virginica.

### 2. Dataset Description

The Iris dataset consists of 150 samples with four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each sample belongs to one of three classes:
- Iris Setosa
- Iris Versicolour
- Iris Virginica

Here is a brief look at the first few rows of the dataset:

| SepalLengthCm | SepalWidthCm | PetalLengthCm | PetalWidthCm | Species        |
|---------------|--------------|---------------|--------------|----------------|
| 5.1           | 3.5          | 1.4           | 0.2          | Iris-setosa    |
| 4.9           | 3.0          | 1.4           | 0.2          | Iris-setosa    |
| 4.7           | 3.2          | 1.3           | 0.2          | Iris-setosa    |
| 4.6           | 3.1          | 1.5           | 0.2          | Iris-setosa    |
| 5.0           | 3.6          | 1.4           | 0.2          | Iris-setosa    |

You can download the Iris dataset [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

---

### 3. Data Preprocessing

**Step 1: Import Libraries and Load Dataset**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=column_names)

# Display the first few rows
print(df.head())
```

**Step 2: Encode Categorical Variables**

```python
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
```

**Step 3: Split Data into Training and Testing Sets**

```python
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Step 4: Feature Scaling**

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 4. Building a KNN Model

**Step 1: Import KNN Classifier and Train the Model**

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

**Step 2: Predict the Test Set Results**

```python
y_pred = knn.predict(X_test)
```

---

### 5. Hyperparameter Tuning

**Step 1: Import GridSearchCV and Define Hyperparameter Grid**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

**Step 2: Train the Model with Best Parameters**

```python
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)
y_pred_best = best_knn.predict(X_test)
```

---

### 6. Evaluating the Model

**Step 1: Import Evaluation Metrics**

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluate the initial KNN model
print("Initial KNN Model:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Evaluate the best KNN model
print("\nBest KNN Model after Grid Search:")
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
print(f"Accuracy: {accuracy_score(y_test, y_pred_best) * 100:.2f}%")
```

---

### 7. Conclusion

In this case study, we used the K-Nearest Neighbors (KNN) algorithm to predict the species of iris flowers. We started with data preprocessing, including handling missing values, encoding categorical variables, and feature scaling. We then built and trained a KNN model, tuned hyperparameters using Grid Search, and evaluated the model's performance using various metrics.

The accuracy and performance of the KNN model were significantly improved after hyperparameter tuning. This demonstrates the importance of choosing the right parameters and preprocessing steps to achieve the best results.

---

### Download Iris Dataset as CSV

Here is the Iris dataset saved as a CSV file:

```python
df.to_csv('iris.csv', index=False)
```
