import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot_encode(df, columns, drop_first=False):
    """
    Perform one-hot encoding on categorical columns.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        Columns to encode
    drop_first : bool, optional
        Whether to drop first category
        
    Returns:
    pandas.DataFrame
        Encoded DataFrame
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)


def label_encode(df, columns):
    """
    Perform label encoding on categorical columns.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        Columns to encode
        
    Returns:
    pandas.DataFrame
        Encoded DataFrame
    """
    df_copy = df.copy()
    le = LabelEncoder()
    
    for col in columns:
        df_copy[col] = le.fit_transform(df_copy[col])
    
    return df_copy
