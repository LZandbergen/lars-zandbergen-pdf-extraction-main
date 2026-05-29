import os
import csv
import json

# Define the file paths
data_folder = '/home/lzandbergen/Squadra/lars-zandbergen-pdf-extraction/data'
taxonomy_folder = os.path.join(data_folder, 'taxonomy')

taxonomy_classes_file = os.path.join(taxonomy_folder, 'taxonomy_classes.csv')
taxonomy_features_file = os.path.join(taxonomy_folder, 'taxonomy_features.csv')
taxonomy_feature_values_file = os.path.join(taxonomy_folder, 'taxonomy_feature_values.csv')

# Read taxonomy_classes.csv into a dictionary
taxonomy_classes = {}
with open(taxonomy_classes_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        code = row['code']
        taxonomy_classes[code] = {
            'class_name': row['name'],
            'id': row['id']
        }

# Read taxonomy_features.csv into a list
taxonomy_features = []
with open(taxonomy_features_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        taxonomy_features.append(row)

# Map taxonomy_class_code to features
class_to_features = {}
for feature in taxonomy_features:
    class_code = feature['taxonomy_class_code']
    if class_code not in class_to_features:
        class_to_features[class_code] = []
    class_to_features[class_code].append(feature)

# Read taxonomy_feature_values.csv into a list
taxonomy_feature_values = []
with open(taxonomy_feature_values_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        taxonomy_feature_values.append(row)

# Map taxonomy_feature_code to its possible values
feature_code_to_values = {}
for value in taxonomy_feature_values:
    feature_code = value['taxonomy_feature_code']
    if feature_code not in feature_code_to_values:
        feature_code_to_values[feature_code] = {}
    feature_value = value['value']
    feature_value_code = value['code']
    feature_code_to_values[feature_code][feature_value] = feature_value_code

# Build the final nested dictionary
final_dict = {}
for class_code, class_info in taxonomy_classes.items():
    class_dict = {'class_name': class_info['class_name']}
    features = class_to_features.get(class_code, [])
    features_dict = {}
    for feature in features:
        feature_code = feature['code']
        feature_name = feature['name']
        values_dict = feature_code_to_values.get(feature_code, {})
        features_dict[feature_code] = {
            'feature_name': feature_name,
            'values': values_dict
        }
    class_dict['features'] = features_dict
    final_dict[class_code] = class_dict

# Save the dictionary as a JSON file
output_file = 'taxonomy_data.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(final_dict, json_file, ensure_ascii=False, indent=4)

print(f"Data has been successfully saved to {output_file}")
