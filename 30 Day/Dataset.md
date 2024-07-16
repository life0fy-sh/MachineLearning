

1. **Download the Dataset**:
   - Go to the Kaggle dataset page for credit card fraud detection: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
   - Download the `creditcard.csv` file.

2. **Load the Dataset**:
   - You can upload the `creditcard.csv` file to your local environment or a cloud environment like Google Colab.
   - Use the path to this file in your script.

Here's an updated version of the script to load the dataset after downloading it from Kaggle:

### Step 1: Load the Dataset

```python
import pandas as pd

# Load dataset
# Replace 'path_to_your_dataset/creditcard.csv' with the actual path to your downloaded CSV file
data = pd.read_csv('path_to_your_dataset/creditcard.csv')

# Display the first few rows
print(data.head())
```

If you're using Google Colab, you can upload the dataset directly using the Colab file uploader:

```python
from google.colab import files
uploaded = files.upload()

import pandas as pd

# Assuming the uploaded file is named 'creditcard.csv'
data = pd.read_csv('creditcard.csv')

# Display the first few rows
print(data.head())
```

Once you have the dataset loaded, you can proceed with the rest of the steps in your script. Let me know if you need further assistance!