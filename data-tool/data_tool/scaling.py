import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def minmax_scale(df, columns):
    """
    Scale features to [0, 1] range.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        Columns to scale
        
    Returns:
    pandas.DataFrame
        Scaled DataFrame
    """
    scaler = MinMaxScaler()
    df_copy = df.copy()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy


def standard_scale(df, columns):
    """
    Standardize features (mean=0, std=1).
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        Columns to scale
        
    Returns:
    pandas.DataFrame
        Scaled DataFrame
    """
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy


def robust_scale(df, columns):
    """
    Scale features using robust statistics.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        Columns to scale
        
    Returns:
    pandas.DataFrame
        Scaled DataFrame
    """
    scaler = RobustScaler()
    df_copy = df.copy()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy
