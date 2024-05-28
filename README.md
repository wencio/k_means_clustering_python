

---

## K-Means Clustering in Python

### Overview
This project demonstrates the application of K-Means clustering using Python's `sklearn` library. The objective is to cluster synthetic data points into groups and visualize the results. This project includes a Jupyter Notebook that walks through the entire process of implementing K-Means clustering, from data generation to visualization.

### Project Structure

- **culstering_k_means.ipynb**: A Jupyter Notebook that includes code and explanations for implementing K-Means clustering on a synthetic dataset.
- **K-Means in Python_Solution.pdf**: A detailed solution document by Jessica Cervi that explains the K-Means algorithm and provides a step-by-step guide for its implementation in Python.

### Getting Started

#### Prerequisites
To run the notebooks and scripts in this project, you need the following libraries:
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

#### Running the Project
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/k-means-clustering.git
   cd k-means-clustering
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook culstering_k_means.ipynb
   ```

3. **Follow the steps in the notebook** to execute the code, visualize the data, and understand the K-Means clustering process.

### K-Means Clustering Overview

K-Means is an unsupervised machine learning algorithm used to identify clusters of data objects in a dataset. The algorithm partitions the data into K clusters, where each data point belongs to the cluster with the nearest mean.

#### Steps of K-Means Algorithm:
1. **Initialization**: Randomly choose K initial centroids.
2. **Assignment**: Assign each data point to the nearest centroid.
3. **Update**: Recalculate the centroids as the mean of all data points assigned to each centroid.
4. **Repeat**: Repeat the assignment and update steps until the centroids do not change significantly.

### Example Implementation

#### Data Generation
We use the `make_blobs` function from `sklearn` to generate synthetic data for clustering:
```python
from sklearn.datasets import make_blobs
features, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
```

#### Data Visualization
Visualize the generated data using matplotlib:
```python
import matplotlib.pyplot as plt
plt.scatter(features[:, 0], features[:, 1], s=50)
plt.show()
```

#### Standardizing the Data
Standardize the features to have a mean of 0 and a standard deviation of 1:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

#### Applying K-Means
Fit the K-Means model and predict the clusters:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(init="random", n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
y_kmeans = kmeans.predict(scaled_features)
```

#### Visualizing Clusters
Visualize the clustered data and centroids:
```python
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.show()
```

### Choosing the Optimal Number of Clusters
The Elbow method is used to determine the optimal number of clusters by plotting the sum of squared errors (SSE) for different values of K:
```python
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="random", random_state=42)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
```

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments
Special thanks to Jessica Cervi for the detailed solution and explanation provided in the accompanying PDF document.

---
