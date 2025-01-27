
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set file paths
transactions_path = "C:/Users/ASUS/Downloads/Transactions.csv"
customers_path = "C:/Users/ASUS/Downloads/Customers.csv"

# Load datasets
transactions = pd.read_csv(transactions_path)
customers = pd.read_csv(customers_path)

# Merge datasets to create a combined customer profile
data = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total transaction value
    'Quantity': 'sum',    # Total products purchased
}).reset_index()

# Add customer profile information
data = data.merge(customers, on='CustomerID', how='left')

# One-hot encode categorical variables (Region)
region_encoded = pd.get_dummies(data['Region'], prefix='Region')

# Combine numerical and encoded features
features = pd.concat([data[['TotalValue', 'Quantity']], region_encoded], axis=1)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform K-Means Clustering
# You can test different cluster values (2-10) and choose the optimal one based on DB Index
db_scores = []
for k in range(2, 11):  # Testing clusters from 2 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    db_index = davies_bouldin_score(features_scaled, labels)
    db_scores.append((k, db_index))

# Select the best number of clusters based on the lowest DB Index
optimal_k = sorted(db_scores, key=lambda x: x[1])[0][0]
print(f"Optimal Number of Clusters: {optimal_k}")

# Re-run K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Calculate the final DB Index for the optimal clusters
final_db_index = davies_bouldin_score(features_scaled, data['Cluster'])
print(f"Davies-Bouldin Index for Optimal Clusters: {final_db_index:.2f}")

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=features_scaled[:, 0],  # Scaled TotalValue
    y=features_scaled[:, 1],  # Scaled Quantity
    hue=data['Cluster'],
    palette='viridis',
    legend='full'
)
plt.title(f"Customer Segmentation (K={optimal_k})")
plt.xlabel("TotalValue (scaled)")
plt.ylabel("Quantity (scaled)")
plt.legend(title="Cluster")
plt.show()

# Visualize Cluster Sizes
plt.figure(figsize=(8, 5))
data['Cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()

# Visualize Average Features by Cluster
cluster_summary = data.groupby('Cluster')[['TotalValue', 'Quantity']].mean()
cluster_summary.plot(kind='bar', figsize=(10, 6), colormap='coolwarm')
plt.title("Average Features by Cluster")
plt.ylabel("Average Value")
plt.xlabel("Cluster")
plt.xticks(rotation=0)
plt.show()

# Print Cluster Summary
print("\nCluster Summary (Average Values):")
print(cluster_summary)

# Show the DB Index for all tested clusters
print("\nDB Index Scores for Tested Clusters:")
for k, db_score in db_scores:
    print(f"K={k}: DB Index = {db_score:.2f}")
