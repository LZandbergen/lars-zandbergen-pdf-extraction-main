import pandas as pd
import os
import json

# Function to get a valid file path
def get_valid_file_path():
    while True:
        file_path = input("Enter the absolute path to the CSV file: ") or "/home/lzandbergen/Squadra/lars-zandbergen-pdf-extraction/data/export_lars.csv"
        if os.path.isfile(file_path):
            return file_path
        else:
            print("Invalid file path. Please try again.")

# Function to get a valid choice for columns
def get_valid_choice(headers):
    while True:
        choice = input("Do you want to retain all headers (Y) or select certain columns (enter numbers separated by comma): ")
        if choice.upper() == 'Y':
            return headers
        else:
            try:
                indices = [int(column.strip()) for column in choice.split(",")]
                selected_columns = [headers[i] for i in indices]
                return selected_columns
            except (ValueError, IndexError):
                print("Invalid input. Please enter numbers separated by commas corresponding to the column indices.")

# Function to ask user for column, and check its validity
def get_column(headers, text="Enter which column you want to select: ", default=""):
    while True:
        column = input(text) or default
        if column in headers:
            return column
        else:
            print("Invalid column. Please try again.")

# Get the valid file path
file_path = get_valid_file_path()

# Read the CSV file into a pandas dataframe
df = pd.read_csv(file_path)

# Get the headers from the dataframe and print them with indices
headers = list(df.columns)
for i, header in enumerate(headers):
    print(f"{i}: {header}")

# Get the valid choice for columns
selected_columns = get_valid_choice(headers)

# Filter the dataframe based on selected columns
df = df[selected_columns]
print(f"Selected colums: \n {df.iloc[0].to_markdown()}")

# Ask user for column to be used as unique identifier
unique_column = get_column(headers, text="Select the column to be used as unique identifier (key): ", default="product_identifier")

# Ask user for column containing feature
feature_column = get_column(headers, text="Select the column containing feature names: ", default="feature")

# Ask user for column containing description
description_column = get_column(headers, text="Select the column containing descriptions: ", default="product_description")

# Remove the unique_column and feature_column from selected_columns if they are included
columns_to_include = [col for col in selected_columns if col not in [unique_column, feature_column, description_column]]

# Create a dictionary to hold the data
data_dict = {}

# Group the data by the unique identifier and feature column
grouped = df.groupby([unique_column, feature_column])

# Iterate over each group
for (product_id, feature_name), group in grouped:
    # Initialize the product_id in data_dict if not already present
    if product_id not in data_dict:
        data_dict[str(product_id)] = {"product_description": str(group[description_column].iloc[0])}
    # For each feature, collect the selected columns and their values
    # Assuming one row per feature per product_id
    row = group.iloc[0]
    feature_data = {str(col): str(row[col]) for col in columns_to_include}
    data_dict[product_id][feature_name] = feature_data

# Ask for output file path
output_file = input("\nEnter the absolute path for the output JSON file: ") or "/home/lzandbergen/Squadra/lars-zandbergen-pdf-extraction/data/product_features.json"

# Write the data_dict to a JSON file
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

with open(output_file, 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)

print(f"\nData has been written to {output_file}")