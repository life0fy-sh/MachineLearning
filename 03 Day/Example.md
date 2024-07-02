We'll create a dataset with features like age, income, and student status to predict whether someone will buy a computer.

## Creating the Dataset

```python
import pandas as pd

# Create a simple dataset
data = {
    'Age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

print(df)
```

### Sample Dataset

| Age   | Income | Student | Buys_Computer |
|-------|--------|---------|---------------|
| <=30  | High   | No      | No            |
| <=30  | High   | No      | No            |
| 31-40 | High   | No      | Yes           |
| >40   | Medium | No      | Yes           |
| >40   | Low    | Yes     | Yes           |
| >40   | Low    | Yes     | No            |
| 31-40 | Low    | Yes     | Yes           |
| <=30  | Medium | No      | No            |
| <=30  | Low    | Yes     | Yes           |
| >40   | Medium | Yes     | Yes           |
| <=30  | Medium | Yes     | Yes           |
| 31-40 | Medium | No      | Yes           |
| 31-40 | High   | Yes     | Yes           |
| >40   | Medium | No      | No            |

## Implementing a Decision Tree

Let's use the Scikit-Learn library to implement a decision tree for this dataset.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Encode categorical features as numeric values
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features (X) and target (y)
X = df.drop('Buys_Computer', axis=1)
y = df['Buys_Computer']

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=['Age', 'Income', 'Student'], class_names=['No', 'Yes'], filled=True)
plt.show()
```

### Explanation

1. **Data Encoding**: We encode the categorical features (`Age`, `Income`, `Student`, and `Buys_Computer`) into numeric values using `LabelEncoder`.
2. **Splitting Data**: We split the data into features (`X`) and the target variable (`y`).
3. **Training the Model**: We initialize a `DecisionTreeClassifier` with the criterion set to `entropy` (to use Information Gain) and fit it to the data.
4. **Visualizing the Tree**: We use the `plot_tree` function to visualize the decision tree.

### Result

The decision tree plot will show the splits based on the features and the resulting class labels (`Buys_Computer`: Yes or No).

This is a basic example to illustrate how a decision tree can be implemented and visualized using Python and Scikit-Learn. The dataset is simple and small, but the same principles apply to larger and more complex datasets.
