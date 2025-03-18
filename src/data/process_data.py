import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(file_path):
    """
    Load the renewable energy data from CSV file
    """
    try:
        # Read CSV with the first column as index and handle wrapped headers
        df = pd.read_csv(file_path,
                         index_col=0,
                         parse_dates=True,
                         on_bad_lines='skip')  # Skip problematic lines

        print(f"Data loaded successfully with shape: {df.shape}")

        # Drop any unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            print(f"Dropped unnamed columns: {unnamed_cols}")

        # Drop rows where index is NaN
        if df.index.isnull().any():
            df = df.dropna(subset=[df.index.name or 'index'])
            print("Dropped rows with NaN index values")

        # Convert string 'null' or empty strings to NaN
        df = df.replace(['null', ''], np.nan)

        # Convert numeric columns to float
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                continue

        print(f"Final shape after cleaning: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            non_null = df[col].count()
            dtype = df[col].dtype
            print(f"{col}: {non_null} non-null values ({dtype})")

        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def clean_data(df):
    """
    Clean the data by handling missing values and outliers
    """
    print("Initial missing values:")
    print(df.isnull().sum())

    # Remove columns with too many missing values (more than 80%)
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > 0.8].index
    if len(columns_to_drop) > 0:
        print(
            f"\nDropping columns with >80% missing values: {', '.join(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)

    # Handle missing values
    df = df.ffill()  # Forward fill
    df = df.bfill()  # Backward fill for any remaining NAs

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"\nRemoved {initial_rows - len(df)} duplicate rows")

    # Remove rows where all numeric columns are 0 (likely measurement errors)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        zero_rows = df[numeric_cols].eq(0).all(axis=1)
        df = df[~zero_rows]
        if zero_rows.sum() > 0:
            print(
                f"\nRemoved {zero_rows.sum()} rows where all numeric values were 0")

    print("\nRemaining missing values:")
    print(df.isnull().sum())

    return df


def create_features(df):
    """
    Create time series features from datetime index
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {str(e)}")

    # Create time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear

    # Identify the target column (assuming it's related to power or energy)
    power_columns = [col for col in df.columns if any(term in col.lower()
                                                      for term in ['power', 'energy', 'generation', 'output'])]

    if not power_columns:
        raise ValueError("No power/energy related column found in the dataset")

    target_col = power_columns[0]  # Use the first matching column as target
    print(f"Using '{target_col}' as target variable")

    # Create lag features
    df[f'lag_1h_{target_col}'] = df[target_col].shift(1)
    df[f'lag_24h_{target_col}'] = df[target_col].shift(24)
    df[f'lag_7d_{target_col}'] = df[target_col].shift(168)  # 7 days * 24 hours

    # Create rolling mean features
    df[f'rolling_mean_24h_{target_col}'] = df[target_col].rolling(
        window=24).mean()
    df[f'rolling_mean_7d_{target_col}'] = df[target_col].rolling(
        window=168).mean()

    return df


def scale_features(df, scaler=None, is_training=True):
    """
    Scale numerical features using StandardScaler
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns

    if is_training:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[numeric_features])
        # Save the scaler
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        if scaler is None:
            scaler = joblib.load('models/scaler.pkl')
        scaled_features = scaler.transform(df[numeric_features])

    scaled_df = pd.DataFrame(
        scaled_features, columns=numeric_features, index=df.index)
    return scaled_df, scaler


def prepare_data_for_training(df, test_size=0.2):
    """
    Prepare data for training by splitting into features and target
    """
    # Drop rows with NaN values created by lag features
    df = df.dropna()

    # Identify the target column (assuming it's related to power or energy)
    power_columns = [col for col in df.columns if any(term in col.lower()
                                                      for term in ['power', 'energy', 'generation', 'output'])]

    if not power_columns:
        raise ValueError("No power/energy related column found in the dataset")

    target_col = power_columns[0]  # Use the first matching column as target

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into train and test sets
    train_size = int(len(df) * (1 - test_size))
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to run the data processing pipeline
    """
    try:
        # Create necessary directories if they don't exist
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        print("Loading data...")
        # Load data
        df = load_data('data/raw/Turbine_Data.csv')
        if df is None:
            return

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")

        print("\nCleaning data...")
        # Process data
        df = clean_data(df)

        print("\nCreating features...")
        df = create_features(df)
        print(f"Features created. New shape: {df.shape}")
        print(f"New columns: {', '.join(df.columns)}")

        print("\nScaling features...")
        # Scale features
        scaled_df, _ = scale_features(df)

        print("\nPreparing training data...")
        # Prepare data for training
        X_train, X_test, y_train, y_test = prepare_data_for_training(scaled_df)

        # Save processed data
        print("\nSaving processed data...")
        X_train.to_csv('data/processed/X_train.csv')
        X_test.to_csv('data/processed/X_test.csv')
        y_train.to_csv('data/processed/y_train.csv')
        y_test.to_csv('data/processed/y_test.csv')

        print("\nData processing completed successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

    except Exception as e:
        print(f"\nError during data processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
