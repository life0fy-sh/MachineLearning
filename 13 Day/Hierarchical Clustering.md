# Hierarchical clustering - unsupervised learning

### Introduction

Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy of clusters. It is particularly useful for discovering the underlying structure of the data when the number of clusters is not known in advance. Hierarchical clustering can be either agglomerative (bottom-up) or divisive (top-down).

### Key Concepts in Hierarchical Clustering

1. **Agglomerative Clustering**: A bottom-up approach where each data point starts as its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
2. **Divisive Clustering**: A top-down approach where all data points start in one cluster, and splits are performed recursively as one moves down the hierarchy.
3. **Dendrogram**: A tree-like diagram that records the sequences of merges or splits.
4. **Linkage Criteria**: Determines the distance between sets of observations as a function of the pairwise distances between observations.

### Types of Linkage Criteria

1. **Single Linkage**: The distance between two clusters is defined as the minimum distance between any single data point in the first cluster and any single data point in the second cluster.
2. **Complete Linkage**: The distance between two clusters is defined as the maximum distance between any single data point in the first cluster and any single data point in the second cluster.
3. **Average Linkage**: The distance between two clusters is defined as the average distance between all pairs of data points, one from each cluster.
4. **Ward's Linkage**: The distance between two clusters is defined as the increase in the sum of squared deviations from the mean of the merged cluster compared to the sum of squared deviations from the means of the two original clusters.

### Steps in Hierarchical Clustering

1. **Compute the Distance Matrix**: Calculate the distance between every pair of data points.
2. **Initialize Clusters**: Each data point starts as its own cluster.
3. **Merge Clusters**: At each step, merge the closest pair of clusters.
4. **Update the Distance Matrix**: Recalculate distances between the new cluster and the existing clusters.
5. **Repeat**: Continue merging until only one cluster remains or a stopping criterion is met.
6. **Plot the Dendrogram**: Visualize the clustering process with a dendrogram.

### Example: Hierarchical Clustering with the Iris Dataset

We'll use the Iris dataset to demonstrate hierarchical clustering.

#### Step 1: Load the Iris Dataset

```python
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = iris.target
```

#### Step 2: Compute the Distance Matrix

We'll use the Euclidean distance to compute the distance matrix.

```python
from scipy.spatial.distance import pdist, squareform

# Compute the distance matrix
distance_matrix = pdist(X, metric='euclidean')
distance_matrix = squareform(distance_matrix)
```

#### Step 3: Apply Hierarchical Clustering

We'll use the `linkage` function from `scipy.cluster.hierarchy` to perform hierarchical clustering.

```python
import scipy.cluster.hierarchy as sch

# Perform hierarchical clustering using Ward's linkage
Z = sch.linkage(X, method='ward')
```

#### Step 4: Plot the Dendrogram

We'll visualize the clustering process with a dendrogram.

```python
import matplotlib.pyplot as plt

# Plot the dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(Z, labels=iris.target, leaf_rotation=90, leaf_font_size=10, color_threshold=0)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

#### Step 5: Cut the Dendrogram to Form Clusters

We'll cut the dendrogram at a chosen distance to form a specified number of clusters.

```python
from scipy.cluster.hierarchy import fcluster

# Cut the dendrogram to form 3 clusters
max_d = 7.0  # Maximum distance
clusters = fcluster(Z, max_d, criterion='distance')

# Add the cluster labels to the DataFrame
df['cluster'] = clusters

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism')
plt.title('Hierarchical Clustering: Iris Dataset')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
```

### Analysis of Results

1. **Dendrogram**: The dendrogram provides a visual representation of the hierarchical clustering process. The height at which two clusters are merged indicates the distance between them. By cutting the dendrogram at a certain height, we can determine the number of clusters.
2. **Cluster Visualization**: By plotting the clusters, we can see the separation and distribution of data points in different clusters. This helps in understanding the underlying structure of the data.

### Practical Applications

1. **Customer Segmentation**: Grouping customers based on their purchasing behavior for targeted marketing.
2. **Document Clustering**: Organizing documents into topics for information retrieval and recommendation systems.
3. **Image Segmentation**: Dividing an image into meaningful parts for computer vision tasks.
4. **Genomics**: Identifying gene expression patterns and grouping similar genes.

### Conclusion

In this detailed tutorial, we covered the fundamentals of hierarchical clustering, including key concepts, types of linkage criteria, and the steps involved in the algorithm. We demonstrated hierarchical clustering using the Iris dataset, visualized the results with a dendrogram, and analyzed the clusters formed.

Hierarchical clustering is a powerful tool for uncovering the hierarchical structure in data and has numerous applications in various domains. By understanding and applying these techniques, you can gain valuable insights into your data and make informed decisions based on the clustering results.

