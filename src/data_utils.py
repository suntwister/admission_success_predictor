import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def show_basic_info(df):
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isna().sum())
    print("\nFirst 5 rows:")
    print(df.head())

def check_duplicates(df):
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        print("No duplicate rows found")
    else:
        print(f"{duplicate_count} duplicate row(s) found")
    return duplicate_count

def detect_outliers(df, columns):
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers[col] = outlier_count
    
    return outliers