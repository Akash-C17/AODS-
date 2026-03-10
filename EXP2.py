import sklearn as sk
from sklearn.datasets import load_iris 
from sklearn.cluster  import KMeans 
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data

Kmeans = KMeans(n_clusters=3, random_state=42)
Kmeans.fit(X)
cluster_labels= Kmeans.labels_
cluster_centers= Kmeans.cluster_centers_
print('Cluster labels for each sample:')
print(cluster_labels)
print("\nCluster Centers (centroids of the cluster):")
print(cluster_centers)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1], c=cluster_labels, cmap='viridis', marker='o', edgecolors='black', s=100)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1], c='red', marker ="X", s=200, label = 'C entroids')
plt.title("K-Means Clustering of iris data")
plt.xlabel("Sepal length") 
plt.ylabel("Sepal Width")
plt.legend()
plt.show()