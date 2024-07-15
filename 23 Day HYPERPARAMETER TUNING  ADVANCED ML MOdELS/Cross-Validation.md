# Cross-Validation

### Steps for Cross-Validation

#### Step 1: Import Libraries and Load Data

First, you need to import the necessary libraries and load your dataset. Here, we'll use the Iris dataset for demonstration purposes.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
```

#### Step 2: Load and Prepare Data

Load the dataset and split it into features (X) and target labels (y).

```python
# Load dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Optionally, convert to pandas DataFrame for easier manipulation
# iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_df['target'] = iris.target
```

#### Step 3: Initialize Cross-Validation Method

Choose the type of cross-validation you want to use. Here, we'll use K-Fold cross-validation with 5 folds.

```python
# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

- **`n_splits`**: Number of folds (in this case, 5).
- **`shuffle`**: Whether to shuffle the data before splitting.
- **`random_state`**: Seed for random number generator (for reproducibility).

#### Step 4: Create Model Instance

Instantiate your machine learning model. Here, we'll use a logistic regression model.

```python
# Create a logistic regression model
model = LogisticRegression(max_iter=1000)
```

#### Step 5: Perform Cross-Validation

Now, perform cross-validation using the `cross_val_score` function, which trains the model and computes scores across folds.

```python
# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kf)
```

#### Step 6: Evaluate Results

Print the scores obtained from each fold and calculate the mean accuracy.

```python
# Print scores for each fold
print("Cross-validation scores:", scores)

# Calculate mean accuracy
mean_accuracy = scores.mean()
print("Mean accuracy:", mean_accuracy)
```

#### Step 7: Interpret Results

- The `scores` variable now contains the accuracy achieved in each fold of the cross-validation.
- `mean_accuracy` provides the average accuracy across all folds, which is a more robust estimate of model performance than a single train-test split.

### Complete Example Code

Here's the complete example code encapsulating all the steps:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kf)

# Print scores for each fold
print("Cross-validation scores:", scores)

# Calculate mean accuracy
mean_accuracy = scores.mean()
print("Mean accuracy:", mean_accuracy)
```

### Explanation

- **Step 1-2**: We import necessary libraries and load the Iris dataset.
- **Step 3**: Initialize a K-Fold cross-validation object with 5 folds, ensuring data shuffling and setting a random seed for reproducibility.
- **Step 4**: Create a logistic regression model.
- **Step 5**: Use `cross_val_score` to perform cross-validation. It trains the model on each fold and returns the scores for each fold.
- **Step 6**: Print the scores for each fold and compute the mean accuracy.
- **Step 7**: Interpret the results to understand the model's performance across different data subsets.
