
### 1. AUC-ROC Curve

#### 1.1. Example 1: Email Spam Detection
Consider a spam detection model where you want to evaluate its performance using the AUC-ROC curve.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Example data
y_true = [0, 0, 1, 1]  # 0: not spam, 1: spam
y_scores = [0.1, 0.4, 0.35, 0.8]  # Predicted probabilities

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Email Spam Detection')
plt.legend(loc="lower right")
plt.show()
```

#### 1.2. Example 2: Medical Diagnosis
Evaluate a medical diagnosis model's performance in detecting a disease.

```python
# Example data
y_true = [0, 1, 1, 0, 1, 0, 0, 1]
y_scores = [0.2, 0.8, 0.6, 0.3, 0.7, 0.4, 0.5, 0.9]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Medical Diagnosis')
plt.legend(loc="lower right")
plt.show()
```

### 2. Precision

#### 2.1. Example: Fraud Detection
Evaluate the precision of a fraud detection model.

```python
from sklearn.metrics import precision_score

# Example data
y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 0, 1, 0, 0, 1, 0, 1]

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f'Precision: {precision:.2f}')
```

### 3. Recall

#### 3.1. Example: Disease Outbreak Detection
Evaluate the recall of a model predicting disease outbreaks.

```python
from sklearn.metrics import recall_score

# Example data
y_true = [1, 1, 0, 0, 1, 0, 1, 1]
y_pred = [1, 0, 0, 1, 1, 0, 1, 0]

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f'Recall: {recall:.2f}')
```

### 4. F1 Score

#### 4.1. Example: Sentiment Analysis
Evaluate the F1 score of a sentiment analysis model.

```python
from sklearn.metrics import f1_score

# Example data
y_true = [1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f'F1 Score: {f1:.2f}')
```

### 5. Confusion Matrix

#### 5.1. Example: Image Classification
Evaluate the performance of an image classification model using a confusion matrix.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example data
y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Image Classification')
plt.show()
```

### Combined Example with All Metrics

#### Example: Credit Card Fraud Detection

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Example data
y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
y_scores = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.8, 0.2]
y_pred = [1 if score > 0.5 else 0 for score in y_scores]

# AUC-ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Card Fraud Detection')
plt.legend(loc="lower right")
plt.show()

# Precision, Recall, F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Credit Card Fraud Detection')
plt.show()
```
