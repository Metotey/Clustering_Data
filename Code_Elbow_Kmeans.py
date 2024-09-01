import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Load the data (use the actual path)
df = pd.read_csv('/path/to/Models_z_score.csv')

# Display the first few rows and basic information
print(df.head())
print("\
Dataframe Info:")
print(df.info())

# Prepare the data for clustering
features = ['MetaRNN_z_score', 'SNPred_z_score', 'CPT-1_z_score', 'GEMME_z_score']
X = df[features]
inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_curve.png')
plt.close()

# Find the optimal K using KneeLocator
kl = KneeLocator(K, inertias, curve="convex", direction="decreasing")
optimal_k = kl.elbow

print(f"\
Optimal number of clusters: {optimal_k}")

# Perform K-means clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Display the entire dataframe with clustering results
print(df)

# Save the clustering results to a new CSV file
df.to_csv('Result_clustering.csv', index=False)