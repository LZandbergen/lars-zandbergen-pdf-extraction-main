import json
import os
import pandas as pd
from pathlib import Path  

def get_class_info(json_file_path, class_code):
    """
    Parses the JSON document and returns the class_name and features dictionary for a given class_code.
    
    Parameters:
        json_file_path (str): The path to the JSON file.
        class_code (str): The code of the class you want to retrieve information for.
        
    Returns:
        tuple: (class_name, features_dict)
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    class_data = data.get(class_code)
    if not class_data:
        raise ValueError(f"Class code '{class_code}' not found in the JSON data.")
    
    class_name = class_data['class_name']
    features_dict = class_data['features']
    
    return class_name, features_dict

# Example of usage
if __name__ == "__main__":
    def get_valid_file_path(prompt):
        while True:
            file_path = Path(input(prompt)).resolve()
            if file_path.exists():
                return file_path
            else:
                print(f"File not found: {file_path}. Please try again.")

    json_file = get_valid_file_path("Enter the path to the JSON file: ")
    csv_file = get_valid_file_path("Enter the path to the CSV file: ")

    df = pd.read_csv(csv_file)
    random_row = df.sample(n=1).iloc[0]
    class_code = random_row['taxonomy_class_code']

    try:
        class_name, features = get_class_info(json_file, class_code)
        print(f"Class Name: {class_name}")
        print("Features:")
        for feature_code, feature_info in features.items():
            print(f"  Feature Code: {feature_code}")
            print(f"    Feature Name: {feature_info['feature_name']}")
            print(f"    Values:")
            for value, code in feature_info['values'].items():
                print(f"      {value}: {code}")
    except ValueError as e:
        print(f"Error encountered: {e}")