
# Hierarchical Clustering - Case Study with "company_profit" Dataset

**Objective:** To demonstrate the process and application of hierarchical clustering on a dataset of companies, analyzing their financial performance.

**Dataset:** The hypothetical "company_profit" dataset containing the following features:
- Company Name
- Revenue
- Expenses
- Number of Employees
- Profit

### Steps:

1. **Data Preparation:**
   - Create the dataset.
   - Explore the dataset to understand its structure and distribution.

2. **Data Preprocessing:**
   - Handle any missing values (if any).
   - Standardize the features.

3. **Hierarchical Clustering:**
   - Compute the distance matrix.
   - Perform hierarchical clustering using different linkage methods.
   - Visualize the dendrogram to determine the optimal number of clusters.
   
4. **Cluster Analysis:**
   - Assign data points to clusters.
   - Analyze the clusters to understand their characteristics.

5. **Interpretation and Conclusion:**
   - Interpret the results and provide insights.
   - Discuss the advantages and limitations of hierarchical clustering.

### Step-by-Step Implementation

#### Step 1: Data Preparation

Let's first create the hypothetical dataset.

```python
import pandas as pd
import numpy as np

# Create a hypothetical "company_profit" dataset
np.random.seed(0)
num_companies = 100
data = {
    'Company Name': [f'Company {i+1}' for i in range(num_companies)],
    'Revenue': np.random.randint(1e6, 1e7, num_companies),
    'Expenses': np.random.randint(5e5, 9e6, num_companies),
    'Number of Employees': np.random.randint(50, 1000, num_companies),
    'Profit': np.random.randint(-1e6, 5e6, num_companies),
}
company_data = pd.DataFrame(data)

# Save the dataset to a CSV file
file_path = '/mnt/data/company_profit.csv'
company_data.to_csv(file_path, index=False)



### Step-by-Step Implementation

#### Step 1: Data Preparation

```python
import pandas as pd

# Load the dataset
file_path = 'path_to_your_dataset/company_profit.csv'
company_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
company_data.head()
```

#### Step 2: Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# Check for missing values
company_data.isnull().sum()

# Standardize the features (excluding the 'Company Name' column)
features = company_data.columns[1:]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(company_data[features])

# Convert the scaled data back to a DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=features)
```

#### Step 3: Hierarchical Clustering

```python
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Compute the distance matrix
distance_matrix = sch.distance.pdist(scaled_data, metric='euclidean')

# Perform hierarchical clustering using different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']

for method in linkage_methods:
    Z = sch.linkage(distance_matrix, method=method)
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z)
    plt.title(f'Dendrogram using {method} linkage')
    plt.xlabel('Companies')
    plt.ylabel('Distance')
    plt.show()
```

#### Step 4: Cluster Analysis

```python
from scipy.cluster.hierarchy import fcluster

# Determine the optimal number of clusters by visualizing the dendrogram
# For this example, let's assume we choose 3 clusters based on the dendrogram

# Perform clustering
Z = sch.linkage(distance_matrix, method='ward')
clusters = fcluster(Z, 3, criterion='maxclust')

# Add cluster assignments to the data
company_data['cluster'] = clusters

# Analyze the clusters
cluster_summary = company_data.groupby('cluster').mean()

import ace_tools as tools; tools.display_dataframe_to_user(name="Cluster Summary", dataframe=cluster_summary)
```

#### Step 5: Interpretation and Conclusion

```python
# Interpretation and conclusion
# Example interpretation:
# Cluster 1: High revenue and profit, large number of employees
# Cluster 2: Moderate revenue and profit, moderate number of employees
# Cluster 3: Low revenue and profit, small number of employees

# Advantages and limitations
# Advantages: Intuitive, produces a dendrogram
# Limitations: Computationally expensive for large datasets, choice of linkage method can affect results
```

### Conclusion

1. **Cluster Characteristics:**
   - Analyze the characteristics of each cluster based on the mean values of the features.

2. **Interpretation:**
   - Interpret the clusters to understand the financial performance of the companies in each cluster.

3. **Advantages and Limitations:**
   - Hierarchical clustering provides a clear visualization of the clustering process through the dendrogram.
   - It can be computationally expensive and sensitive to the choice of linkage method.

This case study demonstrates the process of hierarchical clustering on a dataset of companies, providing insights into their financial performance and clustering characteristics.