import pandas as pd
import numpy as np

def handle_missing_values(df, strategy='drop', columns=None, fill_value=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    strategy : str or dict, optional
        Handling strategy: 
        - String: 'drop', 'mean', 'median', 'mode', 'constant' (applied to all columns)
        - Dictionary: {column: strategy} for per-column strategies
    columns : list, optional
        Columns to process (default all columns)
    fill_value : scalar or dict, optional
        Value for 'constant' strategy (single value or {column: value})
        
    Returns:
    pandas.DataFrame
        Processed DataFrame
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.columns
    
    if isinstance(strategy, dict):
        for col, col_strategy in strategy.items():
            if col not in df_copy.columns:
                continue
                
            col_fill_value = fill_value.get(col) if isinstance(fill_value, dict) else fill_value
            
            df_copy = _apply_missing_value_strategy(
                df_copy, 
                col, 
                col_strategy, 
                col_fill_value
            )
        return df_copy
    
    if strategy == 'drop':
        return df_copy.dropna(subset=columns)
    
    for col in columns:
        df_copy = _apply_missing_value_strategy(
            df_copy, 
            col, 
            strategy, 
            fill_value
        )
    
    return df_copy

def _apply_missing_value_strategy(df, col, strategy, fill_value=None):
    """Вспомогательная функция для обработки пропусков в одном столбце"""
    if strategy == 'drop':
        return df.dropna(subset=[col])
    
    if pd.api.types.is_numeric_dtype(df[col]):
        if strategy == 'mean':
            fill_val = df[col].mean()
        elif strategy == 'median':
            fill_val = df[col].median()
        elif strategy == 'mode':
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else None
        elif strategy == 'constant':
            fill_val = fill_value
        else:
            raise ValueError(f"Unknown strategy: {strategy} for column {col}")
    else:  
        if strategy == 'mode':
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else None
        elif strategy == 'constant':
            fill_val = fill_value
        else:
            raise ValueError(f"Strategy {strategy} not supported for non-numeric column {col}")

    if fill_val is None:
        return df
    df[col] = df[col].fillna(fill_val)
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    subset : list, optional
        Columns to consider (default all columns)
    keep : {'first', 'last', False}, optional
        Which duplicates to keep
        
    Returns:
    pandas.DataFrame
        DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def clip_outliers(df, column, method='iqr', threshold=1.5, 
                 lower_quantile=0.05, upper_quantile=0.95):
    df_copy = df.copy()
    series = df_copy[column]
    
    if series.empty:
        return df_copy

    if series.nunique() == 1:
        return df_copy
    
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
    elif method == 'quantile':
        lower_bound = series.quantile(lower_quantile)
        upper_bound = series.quantile(upper_quantile)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    clipped = series.clip(lower_bound, upper_bound)
    df_copy[column] = clipped.astype(series.dtype, copy=False)
    return df_copy
