import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot_encode(df, columns, drop_first=False):
    """
    Perform one-hot encoding on categorical columns.
    
    Parameters:
    ... (existing docstring)
    """
    df_copy = df.copy()
    
    for col in columns:
        categories = sorted(df_copy[col].unique())

        dummies = pd.get_dummies(
            df_copy[col], 
            prefix=col, 
            drop_first=drop_first,
            dtype=int
        )

        df_copy = df_copy.drop(col, axis=1)
        
        df_copy = pd.concat([df_copy, dummies], axis=1)
    
    return df_copy


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
