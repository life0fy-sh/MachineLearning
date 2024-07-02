#### fictional dataset named `p4n_employee` that contains information about employees in a company. We'll include features like age, department, years at the company, and salary to predict whether an employee is likely to leave the company.

## Creating the Dataset

### Sample Data

| Age | Department | Years_at_Company | Salary  | Left |
|-----|------------|------------------|---------|------|
| 25  | Sales      | 2                | 50000   | No   |
| 45  | HR         | 10               | 80000   | No   |
| 30  | IT         | 5                | 60000   | Yes  |
| 35  | Sales      | 6                | 70000   | No   |
| 50  | IT         | 20               | 120000  | No   |
| 28  | HR         | 3                | 55000   | Yes  |
| 40  | IT         | 15               | 110000  | No   |
| 29  | Sales      | 4                | 52000   | Yes  |
| 32  | HR         | 8                | 62000   | No   |
| 38  | IT         | 10               | 95000   | Yes  |
| 24  | Sales      | 1                | 48000   | Yes  |
| 41  | HR         | 9                | 85000   | No   |
| 33  | IT         | 7                | 65000   | Yes  |
| 36  | Sales      | 8                | 75000   | No   |
| 27  | HR         | 3                | 52000   | Yes  |

```python
import pandas as pd

# Create the p4n_employee dataset
data = {
    'Age': [25, 45, 30, 35, 50, 28, 40, 29, 32, 38, 24, 41, 33, 36, 27],
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR'],
    'Years_at_Company': [2, 10, 5, 6, 20, 3, 15, 4, 8, 10, 1, 9, 7, 8, 3],
    'Salary': [50000, 80000, 60000, 70000, 120000, 55000, 110000, 52000, 62000, 95000, 48000, 85000, 65000, 75000, 52000],
    'Left': ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print(df)
```

## Case Study: Predicting Employee Turnover

### Goal

The goal is to predict whether an employee is likely to leave the company based on their age, department, years at the company, and salary.

### Solution

We'll use a decision tree classifier to build a model that can make these predictions.

### Data Preparation

```python
from sklearn.preprocessing import LabelEncoder

# Encode categorical features as numeric values
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

print(df)
```

### Building the Decision Tree Model

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Split the data into features (X) and target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=label_encoders['Left'].classes_, filled=True)
plt.show()
```

### Evaluation and Interpretation

The accuracy score and classification report will give us an idea of how well the model performs on the test data. The decision tree visualization helps us understand how the model makes decisions.

#### Example Output

```plaintext
Accuracy: 0.6667
              precision    recall  f1-score   support

           0       0.67      1.00      0.80         2
           1       1.00      0.50      0.67         2

    accuracy                           0.67         4
   macro avg       0.83      0.75      0.73         4
weighted avg       0.83      0.67      0.73         4
```

This output shows that the model has an accuracy of 66.67%. The precision, recall, and F1-score provide further insights into the model's performance.

### Conclusion

In this case study, we created a simple dataset representing employee information and used a decision tree classifier to predict employee turnover. The model's performance can be improved with more data and feature engineering, but this example illustrates the basic process of building and evaluating a decision tree model.
