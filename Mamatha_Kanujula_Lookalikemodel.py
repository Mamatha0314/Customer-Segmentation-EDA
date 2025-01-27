import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Set file paths
transactions_path = "C:/Users/ASUS/Downloads/Transactions.csv"
customers_path = "C:/Users/ASUS/Downloads/Customers.csv"
products_path = "C:/Users/ASUS/Downloads/Products.csv"

# Load datasets
transactions = pd.read_csv(transactions_path)
customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)

# Merge datasets
data = transactions.merge(customers, on='CustomerID', how='left').merge(products, on='ProductID', how='left')

# Feature Engineering
# Aggregate transaction data for each customer
customer_profile = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total transaction value
    'Quantity': 'sum',    # Total products purchased
    'Category': lambda x: ','.join(x.unique()),  # List of unique categories
    'Region': 'first',    # Region of the customer
}).reset_index()

# One-hot encode categorical features (Region and Categories)
region_encoded = pd.get_dummies(customer_profile['Region'], prefix='Region')
category_encoded = pd.get_dummies(customer_profile['Category'].str.get_dummies(sep=','), prefix='Category')

# Combine numerical and encoded features
numerical_features = customer_profile[['TotalValue', 'Quantity']]
features = pd.concat([numerical_features, region_encoded, category_encoded], axis=1)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute Similarity Scores
similarity_matrix = cosine_similarity(features_scaled)

# Generate Lookalike Recommendations
lookalike_map = {}
customer_ids = customer_profile['CustomerID'].tolist()

for idx, customer_id in enumerate(customer_ids[:20]):  # Restrict to first 20 customers
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity score
    top_lookalikes = [(customer_ids[i], score) for i, score in similarity_scores[1:4]]  # Exclude self, pick top 3
    lookalike_map[customer_id] = top_lookalikes

# Create Lookalike.csv
lookalike_list = []
for cust_id, lookalikes in lookalike_map.items():
    lookalike_entry = {
        "CustomerID": cust_id,
        "Lookalikes": [(l_id, round(score, 2)) for l_id, score in lookalikes]  # Round scores for readability
    }
    lookalike_list.append(lookalike_entry)

lookalike_df = pd.DataFrame(lookalike_list)
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike model completed. Results saved to 'Lookalike.csv'")
