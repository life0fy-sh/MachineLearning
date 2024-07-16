
### **Step-by-Step Guide for Model Evaluation**

### **Step 9: Evaluate the Baseline Model**

Let's start by evaluating the baseline RandomForestClassifier model trained earlier.

```python
# Make predictions with the baseline model
y_pred_baseline = rf.predict(X_test)

# Evaluate the baseline model
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred_baseline))
print(confusion_matrix(y_test, y_pred_baseline))

# Confusion Matrix Visualization
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix_baseline = confusion_matrix(y_test, y_pred_baseline)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_baseline, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Baseline Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### **Step 10: Evaluate the Model Tuned with GridSearchCV**

Next, we will evaluate the model tuned using GridSearchCV.

```python
# Make predictions with the GridSearchCV-tuned model
y_pred_grid = best_rf_grid.predict(X_test)

# Evaluate the GridSearchCV-tuned model
print("Model Performance with GridSearchCV Tuned Parameters:")
print(classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred_grid))

# Confusion Matrix Visualization
conf_matrix_grid = confusion_matrix(y_test, y_pred_grid)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_grid, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for GridSearchCV-Tuned Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### **Step 11: Evaluate the Model Tuned with RandomizedSearchCV**

Finally, we will evaluate the model tuned using RandomizedSearchCV.

```python
# Make predictions with the RandomizedSearchCV-tuned model
y_pred_random = best_rf_random.predict(X_test)

# Evaluate the RandomizedSearchCV-tuned model
print("Model Performance with RandomizedSearchCV Tuned Parameters:")
print(classification_report(y_test, y_pred_random))
print(confusion_matrix(y_test, y_pred_random))

# Confusion Matrix Visualization
conf_matrix_random = confusion_matrix(y_test, y_pred_random)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_random, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for RandomizedSearchCV-Tuned Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### **Step 12: Comparison of Models**

Compare the performance of the baseline model and the tuned models using different evaluation metrics.

```python
# Collect evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'Model': ['Baseline', 'GridSearchCV', 'RandomizedSearchCV'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_grid),
        accuracy_score(y_test, y_pred_random)
    ],
    'Precision': [
        precision_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_grid),
        precision_score(y_test, y_pred_random)
    ],
    'Recall': [
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_grid),
        recall_score(y_test, y_pred_random)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_grid),
        f1_score(y_test, y_pred_random)
    ]
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics)

# Display metrics
print(metrics_df)
```

### **Visualizing Model Performance**

Visualize the performance metrics to better understand the improvements.

```python
# Plotting performance metrics
metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(10, 7))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()
```

### **Step 13: ROC Curve and AUC**

To further evaluate the models, we can plot the ROC curve and calculate the AUC (Area Under Curve).

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Compute ROC curve and AUC for each model
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
roc_auc_baseline = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

fpr_grid, tpr_grid, _ = roc_curve(y_test, best_rf_grid.predict_proba(X_test)[:, 1])
roc_auc_grid = roc_auc_score(y_test, best_rf_grid.predict_proba(X_test)[:, 1])

fpr_random, tpr_random, _ = roc_curve(y_test, best_rf_random.predict_proba(X_test)[:, 1])
roc_auc_random = roc_auc_score(y_test, best_rf_random.predict_proba(X_test)[:, 1])

# Plot ROC curves
plt.figure(figsize=(10, 7))
plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline Model (AUC = {roc_auc_baseline:.2f})')
plt.plot(fpr_grid, tpr_grid, label=f'GridSearchCV Model (AUC = {roc_auc_grid:.2f})')
plt.plot(fpr_random, tpr_random, label=f'RandomizedSearchCV Model (AUC = {roc_auc_random:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
```

### **Conclusion**

In this detailed tutorial, we covered:
- Generating a synthetic imbalanced dataset
- Handling imbalanced data using SMOTE
- Training and evaluating a baseline RandomForestClassifier model
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Detailed evaluation of model performance using various metrics and visualizations
- Comparing models to understand the impact of hyperparameter tuning

This comprehensive approach provides a robust framework for improving model performance and ensuring the models are well-tuned and effective.

