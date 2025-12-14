import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
    )
    return df

def handle_missing_values(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df

def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X,y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_test_split_data(X,y, test_size = 0.2, random_state = 42):
    return train_test_split(
        X,y,
        test_size = test_size,
        random_state = random_state,
        stratify = y 
    )