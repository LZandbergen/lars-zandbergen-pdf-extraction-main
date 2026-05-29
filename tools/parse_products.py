import json
import pandas as pd
import os
from typing import Dict, Any

def get_product_data(product_file: str = None, selected_column: str = None) -> Dict[str, Any]:
    """
    Prompts the user to provide the location of a CSV or JSON file, reads the file, and returns its contents as a dictionary.
    For CSV files:
    - The user is warned that CSV files do not support multiple features at the moment.
    - The user is prompted to specify which column contains the unique identifier (key) if not provided.
    - The function reads the CSV file and constructs a dictionary where the keys are the unique identifiers and the values are dictionaries of the remaining row data.
    For JSON files:
    - The function reads the JSON file and loads its contents into a dictionary.
    Returns:
        dict: A dictionary containing the data from the provided CSV or JSON file.
    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the provided file is not in CSV or JSON format.
    Usage:
        dictionary = get_product_data()
    Note:
        - Ensure that the file path provided exists and is accessible.
        - For CSV files, the user must specify the correct column index for the unique identifier.
    """
    if product_file is None:
        while True:
            file_path = input("Provide the location of the CSV or JSON file containing the product data: ") or "/home/lzandbergen/Squadra/lars-zandbergen-pdf-extraction/data/allshoes_catalogus/products.csv"
            if file_path.endswith('.csv') or file_path.endswith('.json'):
                print("Be careful, csv files do not support multiple features at the moment.") if file_path.endswith('.csv') else None
                if os.path.exists(file_path):
                    break
                else:
                    print("File not found. Please try again.")
            else:
                print("Invalid file format. Please provide a CSV or JSON file.")
    else:
        file_path = product_file

    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.csv'):
            # NOTE: Currently discards additional features if multiple features exist for a product
            # NOTE: Currently assumes that the first column contains the unique identifier
            with open(file_path, mode='r', encoding='utf-8') as file:
                df = pd.read_csv(file)
                if selected_column is None:
                    column_index = input(f"Which column contains the unique identifier (key)? \n {', '.join([f'{i}: {item}' for i, item in enumerate(df.columns)])} \n") or "0"
                    column_index = int(column_index)
                else:
                    column_index = int(selected_column)
                dictionary = {}
                for i, row in df.iterrows():
                    key = row.iloc[column_index]
                    value = row.drop(df.columns[column_index]).to_dict()
                    dictionary[key] = value
        else:
            dictionary = json.load(file)

    print("Example of data:")
    for key, value in list(dictionary.items())[:2]:
        print(f"{key}: {value}")

    return dictionary