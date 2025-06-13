
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.6, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
data = pd.DataFrame(X_scaled, columns=['Feature1', 'Feature2'])
data['Cluster'] = clusters

plt.figure(figsize=(8,6))
plt.scatter(data['Feature1'], data['Feature2'], c=data['Cluster'], cmap='rainbow', s=30, alpha=0.7)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

print("Cluster labels:")
print(data['Cluster'].value_counts())