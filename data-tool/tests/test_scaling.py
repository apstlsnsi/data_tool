import pytest
import pandas as pd
import numpy as np
from data_tool.scaling import minmax_scale, standard_scale, robust_scale

@pytest.fixture
def sample_numeric_data():
    return pd.DataFrame({
        'age': [20, 30, 40, 50, 60],
        'income': [30000, 60000, 90000, 120000, 150000],
        'score': [0.1, 0.5, 0.9, 1.3, 1.7]
    })

def test_minmax_scale_basic(sample_numeric_data):
    # Test basic min-max scaling
    scaled = minmax_scale(sample_numeric_data, columns=['age', 'income'])
    
    # Check age scaling
    age_min, age_max = 20, 60
    expected_age = [(x - age_min) / (age_max - age_min) for x in sample_numeric_data['age']]
    np.testing.assert_allclose(scaled['age'], expected_age, atol=1e-7)
    
    # Check income scaling
    income_min, income_max = 30000, 150000
    expected_income = [(x - income_min) / (income_max - income_min) for x in sample_numeric_data['income']]
    np.testing.assert_allclose(scaled['income'], expected_income, atol=1e-7)
    
    # Check unchanged column
    assert scaled['score'].equals(sample_numeric_data['score'])

def test_minmax_scale_range(sample_numeric_data):
    # Verify that scaled values are in [0, 1] range
    scaled = minmax_scale(sample_numeric_data, columns=['age', 'income', 'score'])
    
    for col in ['age', 'income', 'score']:
        assert scaled[col].min() == pytest.approx(0.0)
        assert scaled[col].max() == pytest.approx(1.0)

def test_standard_scale_basic(sample_numeric_data):
    # Test basic standard scaling
    scaled = standard_scale(sample_numeric_data, columns=['age', 'income'])
    
    # Check age scaling with scikit-learn's StandardScaler (ddof=0)
    age = sample_numeric_data['age']
    age_mean = age.mean()
    age_std = age.std(ddof=0)  # Population std (matching scikit-learn)
    expected_age = [(x - age_mean) / age_std for x in age]
    np.testing.assert_allclose(scaled['age'], expected_age, atol=1e-7)

def test_standard_scale_stats(sample_numeric_data):
    # Verify mean=0 and std=1 after scaling with population std
    scaled = standard_scale(sample_numeric_data, columns=['age', 'income', 'score'])
    
    for col in ['age', 'income', 'score']:
        assert scaled[col].mean() == pytest.approx(0.0, abs=1e-7)
        # Use population std (ddof=0) to match scikit-learn
        assert scaled[col].std(ddof=0) == pytest.approx(1.0, abs=1e-7)

def test_robust_scale_basic(sample_numeric_data):
    # Test basic robust scaling
    scaled = robust_scale(sample_numeric_data, columns=['income'])
    
    # Calculate expected values
    income = sample_numeric_data['income']
    q1 = income.quantile(0.25)
    q3 = income.quantile(0.75)
    iqr = q3 - q1
    median = income.median()
    
    expected_income = (income - median) / iqr
    np.testing.assert_allclose(scaled['income'], expected_income, atol=1e-7)
    
    # Check unchanged columns
    assert scaled['age'].equals(sample_numeric_data['age'])
    assert scaled['score'].equals(sample_numeric_data['score'])

def test_robust_scale_outliers():
    # Test robust scaling with outliers
    data = pd.DataFrame({
        'values': [10, 20, 30, 40, 50, 1000]  # 1000 is an outlier
    })
    
    scaled = robust_scale(data, columns=['values'])
    
    # Calculate expected values
    values = data['values']
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    median = values.median()
    
    expected = (values - median) / iqr
    np.testing.assert_allclose(scaled['values'], expected, atol=1e-7)
    
    # Verify the outlier is still present but scaled
    assert abs(scaled['values'].iloc[-1]) > 10  # Outlier remains large

def test_multiple_scaling_methods(sample_numeric_data):
    # Test applying different scaling methods to different columns
    minmax_scaled = minmax_scale(sample_numeric_data, columns=['age'])
    standard_scaled = standard_scale(minmax_scaled, columns=['income'])
    robust_scaled = robust_scale(standard_scaled, columns=['score'])
    
    # Verify each column was scaled appropriately
    assert robust_scaled['age'].min() == pytest.approx(0.0)
    assert robust_scaled['age'].max() == pytest.approx(1.0)
    
    assert robust_scaled['income'].mean() == pytest.approx(0.0, abs=1e-7)
    assert robust_scaled['income'].std()
