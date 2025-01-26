import json
from sklearn.metrics import cohen_kappa_score

# Load JSON data from a file
def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}")
        return None

# Create a mapping of text to labels with normalization
def map_text_to_labels(data):
    """
    Maps each text segment to its labels from the provided JSON data.
    Returns a dictionary: {normalized_text: label}.
    """
    text_label_mapping = {}
    for entry in data:
        for annotation in entry.get('annotations', []):
            for result in annotation.get('result', []):
                if 'value' in result and 'text' in result['value']:
                    key = result['value']['text'].strip().lower()  # Normalize key
                    labels = result['value'].get('labels', [])
                    if labels:
                        text_label_mapping[key] = labels[0]  # Assumes one label per text
    return text_label_mapping

# Align labels from two annotators based on text
def align_labels(mapping1, mapping2):
    """
    Aligns labels from two mappings based on the text keys.
    Only includes texts present in both mappings.
    Returns two aligned lists of labels.
    """
    aligned_labels1 = []
    aligned_labels2 = []

    # Find common texts
    common_texts = set(mapping1.keys()).intersection(mapping2.keys())
    
    for text in common_texts:
        aligned_labels1.append(mapping1[text])
        aligned_labels2.append(mapping2[text])
    
    return aligned_labels1, aligned_labels2

# File paths
file_path_annotator_1 = r'C:\Users\yugde\Downloads\23110373.json'
file_path_annotator_2 = r'C:\Users\yugde\Downloads\23110370-assignment3.json'


# Load JSON files
nlp1 = load_json(file_path_annotator_1)
nlp2 = load_json(file_path_annotator_2)

if nlp1 is None or nlp2 is None:
    print("Error: Unable to load one or both files. Exiting...")
    exit()

# Create mappings of text to labels
mapping1 = map_text_to_labels(nlp1)
mapping2 = map_text_to_labels(nlp2)

# Debugging: Print mapping details
print(f"Annotator 1 mapping sample: {list(mapping1.items())[:5]}")
print(f"Annotator 2 mapping sample: {list(mapping2.items())[:5]}")

# Align labels based on common text
annotator_1_labels, annotator_2_labels = align_labels(mapping1, mapping2)

# Debugging: Check alignment
if not annotator_1_labels or not annotator_2_labels:
    print("Debug: No aligned labels found. Exiting...")
    print(f"Annotator 1 keys: {list(mapping1.keys())[:5]}")
    print(f"Annotator 2 keys: {list(mapping2.keys())[:5]}")
    exit()

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(annotator_1_labels, annotator_2_labels)

# Output results
print(f"Cohen's Kappa: {kappa}")
if kappa < 0:
    print("Interpretation: Poor agreement.")
elif 0 <= kappa < 0.2:
    print("Interpretation: Slight agreement.")
elif 0.2 <= kappa < 0.4:
    print("Interpretation: Fair agreement.")
elif 0.4 <= kappa < 0.6:
    print("Interpretation: Moderate agreement.")
elif 0.6 <= kappa < 0.8:
    print("Interpretation: Substantial agreement.")
else:
    print("Interpretation: Almost perfect agreement.")
