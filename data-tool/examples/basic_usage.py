"""
Basic Usage Example for data-tool Package

This script demonstrates a complete data preprocessing pipeline using:
1. Handling missing values
2. Removing duplicates
3. Clipping outliers
4. Encoding categorical features
5. Scaling numerical features
"""

import pandas as pd
import numpy as np
from data_tool import (
    handle_missing_values,
    remove_duplicates,
    clip_outliers,
    one_hot_encode,
    label_encode,
    minmax_scale,
    standard_scale
)
    
# Create sample dataset with various data issues
data = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 3, 9],
    'age': [25, 32, np.nan, 45, 60, 22, 45, 22, 30, 1000],  # Contains missing value and outlier
    'income': [50000, 75000, 100000, np.nan, 250000, 50000, 75000, 50000, 100000, 50000],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M'],
    'city': ['NY', 'LA', 'LA', 'SF', 'NY', 'LA', 'SF', 'LA', 'LA', 'NY'],
    'purchase_amount': [120, 85, 200, 75, 500, 60, 110, 90, 200, 70]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)
print("\n" + "="*80 + "\n")

# Step 1: Handle missing values
print("Handling missing values...")
df_clean = handle_missing_values(
    df, 
    strategy={
        'age': 'median',     # Use median for age
        'income': 'mean',    # Use mean for income
        'purchase_amount': 'constant',  # Fill with 0 for purchase amount
    },
    fill_value={'purchase_amount': 0}  # Значение только для purchase_amount
)
print("\nAfter handling missing values:")
print(df_clean)
print("\n" + "="*80 + "\n")

# Step 2: Remove duplicates
print("Removing duplicates...")
df_dedup = remove_duplicates(
    df_clean, 
    subset=['customer_id'],  # Identify duplicates by customer_id
    keep='first'             # Keep first occurrence
)
print("\nAfter removing duplicates:")
print(df_dedup)
print("\n" + "="*80 + "\n")

# Step 3: Clip outliers
print("Clipping outliers...")
df_clipped = clip_outliers(
    df_dedup, 
    column='age',
    method='iqr',            # Use IQR method for outlier detection
    threshold=1.5            # 1.5 * IQR
)

df_clipped = clip_outliers(
    df_clipped, 
    column='purchase_amount',
    method='quantile',       # Use quantile method
    lower_quantile=0.05,     # Lower 5% quantile
    upper_quantile=0.95      # Upper 95% quantile
)
print("\nAfter clipping outliers:")
print(df_clipped)
print("\n" + "="*80 + "\n")

# Step 4: Encode categorical features
print("Encoding categorical features...")
# One-Hot Encoding for city
df_encoded = one_hot_encode(
    df_clipped, 
    columns=['city'],
    drop_first=True          # Drop first category to avoid dummy trap
)

# Label Encoding for gender
df_encoded = label_encode(
    df_encoded, 
    columns=['gender']
)
print("\nAfter encoding:")
print(df_encoded)
print("\n" + "="*80 + "\n")

# Step 5: Scale numerical features
print("Scaling numerical features...")
# Min-Max Scaling for age
df_scaled = minmax_scale(
    df_encoded, 
    columns=['age']
)

# Standard Scaling for income and purchase_amount
df_scaled = standard_scale(
    df_scaled, 
    columns=['income', 'purchase_amount']
)
print("\nFinal preprocessed data:")
print(df_scaled)
print("\n" + "="*80 + "\n")

# Save processed data
df_scaled.to_csv('processed_data.csv', index=False)
print("Preprocessed data saved to 'processed_data.csv'")
