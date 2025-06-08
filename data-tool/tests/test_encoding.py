import pytest
import pandas as pd
import numpy as np
from data_tool.encoding import one_hot_encode, label_encode

@pytest.fixture
def sample_categorical_data():
    return pd.DataFrame({
        'color': ['red', 'blue', 'green', 'blue', 'red'],
        'size': ['S', 'M', 'L', 'M', 'XL'],
        'price': [10, 20, 30, 20, 50]
    })

def test_one_hot_encode_basic(sample_categorical_data):
    # Test basic one-hot encoding
    encoded = one_hot_encode(sample_categorical_data, columns=['color'])
    
    # Check expected columns
    assert 'color_red' in encoded.columns
    assert 'color_blue' in encoded.columns
    assert 'color_green' in encoded.columns
    assert 'size' in encoded.columns  # Should remain unchanged
    
    # Check values
    assert encoded['color_red'].tolist() == [1, 0, 0, 0, 1]
    assert encoded['color_blue'].tolist() == [0, 1, 0, 1, 0]
    assert encoded['color_green'].tolist() == [0, 0, 1, 0, 0]

def test_one_hot_encode_drop_first(sample_categorical_data):
    # Test one-hot encoding with drop_first
    encoded = one_hot_encode(sample_categorical_data, columns=['size'], drop_first=True)
    
    assert 'size_L' not in encoded.columns
    
    assert 'size_M' in encoded.columns
    assert 'size_S' in encoded.columns
    assert 'size_XL' in encoded.columns
    
    assert len([c for c in encoded.columns if c.startswith('size_')]) == 3
    
    assert encoded['size_M'].tolist() == [0, 1, 0, 1, 0]
    assert encoded['size_S'].tolist() == [1, 0, 0, 0, 0]
    assert encoded['size_XL'].tolist() == [0, 0, 0, 0, 1]
    
    assert 'color' in encoded.columns
    assert 'price' in encoded.columns

def test_one_hot_encode_multiple_columns(sample_categorical_data):
    # Test encoding multiple columns
    encoded = one_hot_encode(sample_categorical_data, columns=['color', 'size'])
    
    # Check expected columns
    assert 'color_red' in encoded.columns
    assert 'color_blue' in encoded.columns
    assert 'size_S' in encoded.columns
    assert 'size_M' in encoded.columns
    assert 'price' in encoded.columns  # Should remain unchanged
    
    # Check number of new columns
    color_cols = [c for c in encoded.columns if c.startswith('color_')]
    size_cols = [c for c in encoded.columns if c.startswith('size_')]
    assert len(color_cols) == 3
    assert len(size_cols) == 4

def test_label_encode_basic(sample_categorical_data):
    # Test basic label encoding
    encoded = label_encode(sample_categorical_data, columns=['color'])
    
    # Check encoding - alphabetical order: blue=0, green=1, red=2
    color_mapping = {'blue': 0, 'green': 1, 'red': 2}
    expected = [color_mapping[c] for c in sample_categorical_data['color']]
    assert encoded['color'].tolist() == expected

def test_label_encode_multiple_columns(sample_categorical_data):
    # Test encoding multiple columns
    encoded = label_encode(sample_categorical_data, columns=['color', 'size'])
    
    # Color: blue=0, green=1, red=2
    color_mapping = {'blue': 0, 'green': 1, 'red': 2}
    expected_color = [color_mapping[c] for c in sample_categorical_data['color']]
    assert encoded['color'].tolist() == expected_color
    
    # Size: L=0, M=1, S=2, XL=3 (alphabetical order)
    size_mapping = {'L': 0, 'M': 1, 'S': 2, 'XL': 3}
    expected_size = [size_mapping[s] for s in sample_categorical_data['size']]
    assert encoded['size'].tolist() == expected_size

def test_label_encode_preserves_numeric(sample_categorical_data):
    # Test that numeric columns are preserved
    encoded = label_encode(sample_categorical_data, columns=['size'])
    assert encoded['price'].equals(sample_categorical_data['price'])
    assert encoded['color'].equals(sample_categorical_data['color'])
