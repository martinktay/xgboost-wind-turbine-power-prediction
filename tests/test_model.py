import pytest
import pandas as pd
import numpy as np
from src.models.train_model import train_model, evaluate_model
from src.data.process_data import clean_data, create_features, scale_features


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
    np.random.seed(42)

    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': np.random.normal(15, 5, len(dates)),
        'wind_speed': np.random.normal(5, 2, len(dates)),
        'power_output': np.random.normal(100, 20, len(dates))
    })
    return df


def test_data_cleaning(sample_data):
    """Test data cleaning function"""
    # Add some NaN values
    sample_data.loc[0:5, 'temperature'] = np.nan

    cleaned_data = clean_data(sample_data)

    assert cleaned_data.isna().sum().sum() == 0
    assert len(cleaned_data) == len(sample_data)


def test_feature_creation(sample_data):
    """Test feature creation function"""
    featured_data = create_features(sample_data)

    expected_features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                         'lag_1h', 'lag_24h', 'lag_7d', 'rolling_mean_24h', 'rolling_mean_7d']

    for feature in expected_features:
        assert feature in featured_data.columns


def test_feature_scaling(sample_data):
    """Test feature scaling function"""
    scaled_data, scaler = scale_features(sample_data)

    assert scaled_data.shape == sample_data.shape
    # Scaled data should be centered around 0
    assert (scaled_data.mean().abs() < 1e-10).all()
    # Scaled data should have unit variance
    assert (scaled_data.std() - 1.0 < 1e-10).all()


def test_model_training(sample_data):
    """Test model training function"""
    # Prepare data
    X = sample_data.drop('power_output', axis=1)
    y = sample_data['power_output']

    # Train model
    model = train_model(X, y)

    # Make predictions
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)


def test_model_evaluation(sample_data):
    """Test model evaluation function"""
    # Prepare data
    X = sample_data.drop('power_output', axis=1)
    y = sample_data['power_output']

    # Train model
    model = train_model(X, y)

    # Evaluate model
    metrics, predictions = evaluate_model(model, X, y)

    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert len(predictions) == len(y)
