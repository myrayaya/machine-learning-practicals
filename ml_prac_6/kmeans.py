import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load the dataset
df = pd.read_csv(r"machine-learning-practicals/ml_prac_6/Mall_Customers.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# feature selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features:")
print(X.head())

# plotting original data
plt.figure(figsize=(8, 5))
plt.scatter(X['Annual Income (k$)'],
            X['Spending Score (1-100)'])

plt.title("Original Customer Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data (First 5 Rows):")
print(X_scaled[:5])

# finding optimal k using elbow method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# 7. k-means with 5 clusters
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X_scaled)

# cluster labels in dataset
df['Cluster'] = y_kmeans

print("\nCluster Assignments:")
print(df.head(10))

# plotting the clusters
plt.figure(figsize=(10, 6))

for i in range(5):
    plt.scatter(
        X_scaled[y_kmeans == i, 0],
        X_scaled[y_kmeans == i, 1],
        s=100,
        label=f'Cluster {i}'
    )

# plotting centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=250,
    marker='X',
    color='black',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

# evaluation using silhouette score
score = silhouette_score(X_scaled, y_kmeans)

print("\nSilhouette Score:", round(score, 4))

# cluster analysis
cluster_analysis = df.groupby('Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print("\nAverage Income and Spending Score per Cluster:")
print(cluster_analysis.round(2))

# no. of customers in each cluster
print("\nCustomer Count per Cluster:")
print(df['Cluster'].value_counts().sort_index())

# final dataset
print("\nFinal Dataset:")
print(df.head(10))

# clustered dataset
df.to_csv("Mall_Customers_Clustered.csv", index=False)