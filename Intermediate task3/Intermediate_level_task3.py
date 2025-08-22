import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

print("Sample Data:")
print(df.head())

wcss = []  
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.show()


silhouette_scores = {}
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores[k] = score

print("\nSilhouette Scores:")
for k, score in silhouette_scores.items():
    print(f"k={k} -> Score={score:.3f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n Best k based on Silhouette Score: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)
df['Cluster'] = labels


pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)
df_pca = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = labels

plt.figure(figsize=(6,4))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="Set1", s=50)
plt.title(f"K-Means Clusters (k={best_k})")
plt.show()

print("\nCluster Centers (original feature space):")
print(kmeans.cluster_centers_)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())
