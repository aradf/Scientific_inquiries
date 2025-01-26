# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage( X, 
                                         method = 'ward' ))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hierarchical_cluster = AgglomerativeClustering( n_clusters = 5, 
                                                affinity = 'euclidean', 
                                                linkage = 'ward')
y_hierarchicalCluster = hierarchical_cluster.fit_predict( X )

# Visualising the clusters
plt.scatter(X[y_hierarchicalCluster == 0, 0], X[y_hierarchicalCluster == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hierarchicalCluster == 1, 0], X[y_hierarchicalCluster == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hierarchicalCluster == 2, 0], X[y_hierarchicalCluster == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hierarchicalCluster == 3, 0], X[y_hierarchicalCluster == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hierarchicalCluster == 4, 0], X[y_hierarchicalCluster == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
print("Hello")