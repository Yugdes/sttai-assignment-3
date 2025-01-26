import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# Function to extract labels dynamically
def extract_labels(file, nested=False):
    if nested:
        # Safely extract truck_label from nested structure
        def safe_extract(x):
            try:
                return x[0]["result"][0]["value"].get("choices", [None])[0]
            except (IndexError, KeyError, TypeError):
                return None
        return file["annotations"].apply(safe_extract).map({"Truck": 1, "No Truck": 0, "Trucks": 1, "No Trucks": 0})
    else:
        # Handle case where 'truck_label' might not exist
        if "truck_label" in file:
            return file["truck_label"].map({"Truck": 1, "No Truck": 0})
        else:
            # Attempt to extract from annotations if no direct key exists
            def safe_extract(x):
                try:
                    return x["result"][0]["value"].get("choices", [None])[0]
                except (IndexError, KeyError, TypeError):
                    return None
            return file["annotations"].apply(safe_extract).map({"Truck": 1, "No Truck": 0, "Trucks": 1, "No Trucks": 0})

# Load files using corrected paths
file1 = pd.read_json(r'C:\Users\yugde\Downloads\project-1-at-2025-01-26-22-13-cfdf2f2c.json')
file2 = pd.read_json(r'C:\Users\yugde\Downloads\cv final anno.json')
file3 = pd.read_json(r'C:\Users\yugde\Downloads\project-3-at-2025-01-26-22-12-ec898b7c.json')

# Extract labels
labels_1 = extract_labels(file1, nested=True)
labels_2 = extract_labels(file2, nested=True)  # Changed to nested=True for this structure
labels_3 = extract_labels(file3, nested=True)

# Combine the labels into a single DataFrame
data = pd.DataFrame({
    'Rater1': labels_1,
    'Rater2': labels_2,
    'Rater3': labels_3
})

# Create the matrix for Fleiss' kappa
category_counts = data.apply(pd.Series.value_counts, axis=1).fillna(0).astype(int)
category_counts = category_counts.reindex(columns=[0, 1], fill_value=0)

# Calculate Fleiss' kappa
kappa = fleiss_kappa(category_counts.values)
print(f"Fleiss' kappa: {kappa}")
