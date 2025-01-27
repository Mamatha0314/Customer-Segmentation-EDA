import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
transactions_path = "C:/Users/ASUS/Downloads/Transactions.csv"
customers_path = "C:/Users/ASUS/Downloads/Customers.csv"
products_path = "C:/Users/ASUS/Downloads/Products.csv"
for path, label in zip([transactions_path, customers_path, products_path], 
                       ["Transactions", "Customers", "Products"]):
    if not os.path.exists(path):
        print(f"Error: {label} file not found at {path}")
        exit()  # Stop execution if files are missing

# Load datasets
transactions = pd.read_csv(transactions_path)
customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)

# Preview the datasets
print("Transactions Dataset:")
print(transactions.head(), "\n")

print("Customers Dataset:")
print(customers.head(), "\n")

print("Products Dataset:")
print(products.head(), "\n")

# Check for missing values
print("Missing Values in Transactions:")
print(transactions.isnull().sum(), "\n")

print("Missing Values in Customers:")
print(customers.isnull().sum(), "\n")

print("Missing Values in Products:")
print(products.isnull().sum(), "\n")

# Data Cleaning
# Convert date columns to datetime
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Merge the datasets
data = transactions.merge(customers, on='CustomerID', how='left').merge(products, on='ProductID', how='left')

# Preview the merged dataset
print("Merged Dataset:")
print(data.head(), "\n")

# Descriptive Statistics
print("Descriptive Statistics:")
print(data.describe(), "\n")

# EDA Visualizations
# 1. Total Revenue
total_revenue = data['TotalValue'].sum()
print(f"Total Revenue: ${total_revenue:.2f}")

# 2. Revenue by Region
revenue_by_region = data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
print("Revenue by Region:")
print(revenue_by_region)

plt.figure(figsize=(10, 6))
sns.barplot(x=revenue_by_region.index, y=revenue_by_region.values, palette="viridis")
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# 3. Most Purchased Products
top_products = data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)
print("Top 10 Most Purchased Products:")
print(top_products)

plt.figure(figsize=(12, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title("Top 10 Most Purchased Products")
plt.xlabel("Product Name")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.show()

# 4. Monthly Revenue Trends
data['Month'] = data['TransactionDate'].dt.to_period('M')
monthly_revenue = data.groupby('Month')['TotalValue'].sum()

plt.figure(figsize=(12, 6))
monthly_revenue.plot(kind='line', marker='o', color='orange')
plt.title("Monthly Revenue Trends")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# 5. Revenue by Product Category
category_revenue = data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=category_revenue.index, y=category_revenue.values, palette="coolwarm")
plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# 6. Top 10 Customers by Revenue
top_customers = data.groupby('CustomerName')['TotalValue'].sum().sort_values(ascending=False).head(10)
print("Top 10 Customers by Revenue:")
print(top_customers)

plt.figure(figsize=(12, 6))
top_customers.plot(kind='bar', color='green')
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer Name")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()
