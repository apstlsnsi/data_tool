import pytest
import pandas as pd
import numpy as np
from data_tool.cleaning import handle_missing_values, remove_duplicates, clip_outliers

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 1, 1, 1, 1]
    })

def test_handle_missing_values_drop(sample_data):
    result = handle_missing_values(sample_data, strategy='drop')
    assert result.shape == (3, 3)

def test_handle_missing_values_mean(sample_data):
    result = handle_missing_values(sample_data, strategy='mean', columns=['A'])
    assert result['A'].isna().sum() == 0
    assert result['A'].mean() == pytest.approx(3.0)

def test_remove_duplicates():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
    result = remove_duplicates(df)
    assert result.shape == (2, 2)

def test_clip_outliers():
    df = pd.DataFrame({'A': [1, 2, 3, 100]})
    result = clip_outliers(df, 'A', method='iqr', threshold=1.5)
    assert result['A'].max() < 100

def test_handle_missing_values_dict_strategy():
    data = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [np.nan, 5, 6],
        'C': ['x', np.nan, 'z']
    })
    
    result = handle_missing_values(
        data,
        strategy={'A': 'median', 'B': 'constant', 'C': 'mode'},
        fill_value={'B': 100}
    )
    
    assert result['A'].tolist() == [1, 2, 3]  # median(1,3)=2
    assert result['B'].tolist() == [100, 5, 6]  # constant 100
    assert result['C'].tolist() == ['x', 'x', 'z']  # mode 'x'

def test_handle_missing_values_mixed_fill_value():
    data = pd.DataFrame({
        'numeric': [1, np.nan, 3],
        'text': [np.nan, 'b', 'c']
    })
    
    result = handle_missing_values(
        data,
        strategy={'numeric': 'constant', 'text': 'constant'},
        fill_value={'numeric': 0, 'text': 'missing'}
    )
    
    assert result['numeric'].tolist() == [1, 0, 3]
    assert result['text'].tolist() == ['missing', 'b', 'c']

def test_clip_outliers_no_downcasting():
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 1000],  # 1000 - выброс
        'floats': [1.1, 2.2, 3.3, 4.4, 1000.5]
    })
    
    original_types = df.dtypes.to_dict()
    
    result = clip_outliers(df, 'values', method='iqr', threshold=1.5)
    result = clip_outliers(result, 'floats', method='iqr', threshold=1.5)
    
    assert result['values'].dtype == original_types['values']
    assert result['floats'].dtype == original_types['floats']

    assert result['values'].max() < 1000
    assert result['floats'].max() < 1000.5
