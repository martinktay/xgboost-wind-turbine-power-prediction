import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
from datetime import datetime


def load_processed_data():
    """
    Load the processed training and testing data
    """
    X_train = pd.read_csv('data/processed/X_train.csv', index_col=0)
    X_test = pd.read_csv('data/processed/X_test.csv', index_col=0)
    y_train = pd.read_csv('data/processed/y_train.csv', index_col=0).iloc[:, 0]
    y_test = pd.read_csv('data/processed/y_test.csv', index_col=0).iloc[:, 0]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the XGBoost model
    """
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50
    )

    # Create evaluation set
    eval_set = [(X_train, y_train)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return performance metrics
    """
    predictions = model.predict(X_test)

    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions)
    }

    return metrics, predictions


def save_model_and_metrics(model, metrics):
    """
    Save the trained model and its performance metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the model
    model_filename = f'models/xgboost_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    joblib.dump(model, model_filename)

    # Save the metrics
    metrics_filename = f'models/model_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved as: {model_filename}")
    print(f"Metrics saved as: {metrics_filename}")

    return model_filename, metrics_filename


def main():
    """
    Main function to run the model training pipeline
    """
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics, predictions = evaluate_model(model, X_test, y_test)

    print("\nModel Performance Metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")

    print("\nSaving model and metrics...")
    model_file, metrics_file = save_model_and_metrics(model, metrics)

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
