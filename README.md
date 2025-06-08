
Data-Tool is a lightweight Python package for efficient data preprocessing. Designed for data scientists and analysts, it provides essential tools for cleaning, encoding, and scaling data with minimal code.

## Installation

```bash
pip install "git+https://github.com/apstlsnsi/data_tool.git#subdirectory=data-tool&quot;
```

## Features

- **Data Cleaning**: Handle missing values, remove duplicates, clip outliers
- **Data Encoding**: One-Hot and Label Encoding for categorical features
- **Data Scaling**: Min-Max, Standard, and Robust scaling
- **Pandas Integration**: Works seamlessly with Pandas DataFrames
- **Tested & Reliable**: Full test coverage and CI/CD pipeline

## Quick Start

```python
import pandas as pd
from data_tool import handle_missing_values, one_hot_encode, minmax_scale

# Create sample data
data = {
    'age': [25, None, 35, 40],
    'income': [50000, 60000, None, 70000],
    'city': ['NY', 'LA', 'NY', 'SF']
}
df = pd.DataFrame(data)

# Clean and transform
df = handle_missing_values(df, strategy={'age':'median', 'income':'mean'})
df = one_hot_encode(df, columns=['city'])
df = minmax_scale(df, columns=['age', 'income'])

print(df)
```

## Core Functions

### Data Cleaning
```python
# Handle missing values
df = handle_missing_values(df, strategy='mean')

# Remove duplicates
df = remove_duplicates(df)

# Clip outliers
df = clip_outliers(df, 'price', method='iqr')
```

### Data Encoding
```python
# One-Hot Encoding
df = one_hot_encode(df, ['category'])

# Label Encoding
df = label_encode(df, ['status'])
```

### Data Scaling
```python
# Min-Max Scaling (0-1)
df = minmax_scale(df, ['age'])

# Standard Scaling (mean=0, std=1)
df = standard_scale(df, ['income'])

# Robust Scaling (outlier-resistant)
df = robust_scale(df, ['price'])
```



## Documentation

|       Function         |      Description      |            Parameters           |
|------------------------|-----------------------|---------------------------------|
| handle_missing_values()| Handle missing data   | `strategy`, `fill_value`        |
| remove_duplicates()    | Remove duplicate rows | `subset`, `keep`                |
| clip_outliers()        | Clip extreme values   | `column`, `method`, `threshold` |
| one_hot_encode()       | One-Hot Encoding      | `columns`, `drop_first`         |
| label_encode()         | Label Encoding        | `columns`                       |
| minmax_scale()         | Min-Max Scaling       | `columns`                       |
| standard_scale()       | Standard Scaling      | `columns`                       |
| robust_scale()         | Robust Scaling        | `columns`                       |


