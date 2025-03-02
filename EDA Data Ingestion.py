import os
import pandas as pd
import numpy as np
import requests
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


# Function to standardize JSON data with inconsistent feature names
def standardize_json_keys(json_data):
    standardized_data = []
    key_mappings = {
        "invoice": "invoice_id",
        "date": "transaction_date",
        "amount": "sales"
    }

    for record in json_data:
        standardized_record = {key_mappings.get(k, k): v for k, v in record.items()}
        standardized_data.append(standardized_record)

    return pd.DataFrame(standardized_data)


# Function to load data from different sources
def load_data(source):
    try:
        if source.endswith(".csv"):
            data = pd.read_csv(source)
        elif source.endswith(".xlsx"):
            data = pd.read_excel(source)
        elif source.endswith(".json"):
            with open(source, "r") as file:
                json_data = json.load(file)
            data = standardize_json_keys(json_data)
        else:
            raise ValueError("Unsupported file format")
        return data
    except Exception as e:
        print(f"Error loading data from {source}: {e}")
        return None


# Function to fetch data from an API
def fetch_api_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        json_data = response.json()
        return standardize_json_keys(json_data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None


# Function to fetch data from SQL database
def fetch_db_data(db_config, query="SELECT * FROM sales"):
    try:
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(connection_string)
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


# Function to preprocess data and return feature matrix
def create_feature_matrix(data_sources):
    data_frames = []

    for source in data_sources:
        if source.startswith("http"):
            df = fetch_api_data(source)
        elif isinstance(source, dict):
            df = fetch_db_data(source)
        else:
            df = load_data(source)

        if df is not None:
            data_frames.append(df)

    if not data_frames:
        print("No valid data sources provided.")
        return None

    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data.dropna(inplace=True)  # Removing missing values for now

    # Clean invoice IDs by removing letters
    if "invoice_id" in combined_data.columns:
        combined_data["invoice_id"] = combined_data["invoice_id"].astype(str).apply(lambda x: re.sub(r'[^0-9]', '', x))

    # Aggregate transactions by day
    if "transaction_date" in combined_data.columns:
        combined_data["transaction_date"] = pd.to_datetime(combined_data["transaction_date"])
        combined_data = combined_data.groupby("transaction_date").agg({"sales": "sum"}).reset_index()

    return combined_data


# Function to perform EDA and generate visualizations
def perform_eda(data, target_column):
    print("\nBasic Information about Dataset:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Distribution of Target Variable
    plt.figure(figsize=(8, 5))
    sns.histplot(data[target_column], bins=30, kde=True)
    plt.title(f"Distribution of {target_column}")
    plt.show()

    # Time-Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data["transaction_date"], data[target_column], marker="o", linestyle="-")
    plt.xlabel("Date")
    plt.ylabel(target_column)
    plt.title("Sales Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


# Main Execution
if __name__ == "__main__":
    data_sources = [
        "data/sales_data.csv",  # Example CSV file
        "data/customer_data.xlsx",  # Example Excel file
        "data/sales_data.json",  # Example JSON file
        "https://api.example.com/data",  # Example API
        {  # Example Database Config
            "host": "localhost",
            "port": "5432",
            "dbname": "business_db",
            "user": "admin",
            "password": "password"
        }
    ]

    feature_matrix = create_feature_matrix(data_sources)
    if feature_matrix is not None:
        print("Feature matrix successfully created.")
        perform_eda(feature_matrix, target_column="sales")