
1. **Import Libraries**: First, we import the necessary libraries. For K-means clustering, we'll use the `KMeans` class from the `sklearn.cluster` module, and `numpy` for handling arrays.

```python
from sklearn.cluster import KMeans
import numpy as np
```

2. **Prepare Data**: We define our sample data points. In this example, `X` is a NumPy array containing six data points, each with two features.

```python
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
```

3. **Create KMeans Instance**: We create an instance of the `KMeans` class, specifying the number of clusters (`n_clusters`) we want to identify. In this case, we set it to `2`.

```python
kmeans = KMeans(n_clusters=2, random_state=0)
```

4. **Fit the Model**: We fit the KMeans model to our data. This step involves the algorithm iteratively assigning data points to the nearest cluster center and updating the cluster centers based on the mean of the assigned points.

```python
kmeans.fit(X)
```

5. **Get Cluster Centers**: After fitting the model, we can access the cluster centers using the `cluster_centers_` attribute. The cluster centers are the centroids of the clusters identified by the algorithm.

```python
print(kmeans.cluster_centers_)
```

6. **Get Cluster Labels**: We can also access the cluster labels assigned to each data point using the `labels_` attribute. Each data point is assigned a label indicating which cluster it belongs to.

```python
print(kmeans.labels_)
```

This implementation demonstrates how to use the K-means algorithm to cluster data points into a specified number of clusters, using the scikit-learn library in Python. The algorithm is widely used for various clustering tasks and can be applied to datasets with different numbers of dimensions and data points.
