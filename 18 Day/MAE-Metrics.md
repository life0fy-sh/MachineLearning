# Mean Absolute Error (MAE)

## Introduction
Mean Absolute Error (MAE) is a commonly used metric for evaluating the performance of regression models. It measures the average magnitude of errors in a set of predictions, without considering their direction.

## Formula

The formula for Mean Absolute Error (MAE) is:

\[ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right| \]

where:
- \( y_i \) = actual value
- \( \hat{y}_i \) = predicted value
- \( n \) = number of observations

## Interpretation

- **Magnitude of Errors**: MAE provides the average magnitude of prediction errors.
- **Model Accuracy**: Lower MAE values indicate a more accurate model.

## Example Calculation

Assume actual values \( y = [3, -0.5, 2, 7] \) and predicted values \( \hat{y} = [2.5, 0.0, 2, 8] \).

1. Calculate the absolute errors:
   \[
   |3 - 2.5| = 0.5, \quad |-0.5 - 0.0| = 0.5, \quad |2 - 2| = 0, \quad |7 - 8| = 1
   \]

2. Sum the absolute errors:
   \[
   0.5 + 0.5 + 0 + 1 = 2
   \]

3. Divide by the number of observations (4):
   \[
   \text{MAE} = \frac{2}{4} = 0.5
   \]

## Implementation in Python

### Using NumPy for Manual Calculation

```python
import numpy as np

# Actual and predicted values
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Calculate MAE manually
mae = np.mean(np.abs(y_true - y_pred))
print(f"MAE: {mae}")
