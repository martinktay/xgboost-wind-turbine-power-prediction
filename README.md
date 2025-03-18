# Wind Turbine Power Prediction

This project implements a machine learning model to predict wind turbine power output based on various environmental and operational parameters. The system provides both real-time predictions through a web interface and batch prediction capabilities via API endpoints.

## Features

- Real-time power output prediction
- Interactive web interface for single predictions
- REST API endpoints for batch predictions
- Model performance metrics visualization
- Support for time-series features including lag and rolling means

## Project Structure

```
renewable-energy-timeseries/
├── models/                    # Trained models and scalers
│   ├── xgboost_model_*.pkl   # XGBoost model files
│   ├── scaler.pkl            # Feature scaler
│   └── model_metrics_*.json  # Model performance metrics
├── src/
│   ├── data/                 # Data processing scripts
│   │   └── process_data.py   # Data preprocessing pipeline
│   ├── models/               # Model training scripts
│   │   └── train_model.py    # Model training pipeline
│   └── web/                  # Web application
│       ├── app.py            # Flask application
│       └── templates/        # HTML templates
│           └── index.html    # Main prediction interface
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/renewable-energy-timeseries.git
cd renewable-energy-timeseries
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

1. Start the Flask application:

```bash
python src/web/app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

### Making Predictions

The application provides two ways to make predictions:

1. **Web Interface**: Fill in the form with current turbine parameters to get real-time predictions.

2. **API Endpoints**:

   - Single Prediction:

   ```bash
   curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{
            "WindSpeed": 8.5,
            "WindDirection": 180,
            "AmbientTemperatue": 15.0,
            ...
        }'
   ```

   - Batch Prediction:

   ```bash
   curl -X POST http://localhost:5000/batch_predict \
        -H "Content-Type: application/json" \
        -d '{
            "features": [
                {
                    "WindSpeed": 8.5,
                    ...
                },
                ...
            ]
        }'
   ```

## Model Features

The model uses the following features for prediction:

- Environmental Conditions:

  - Wind Speed
  - Wind Direction
  - Ambient Temperature

- Turbine Parameters:

  - Blade Pitch Angles
  - Generator RPM
  - Rotor RPM
  - Various Temperature Readings
  - Turbine Status

- Time-based Features:

  - Hour of Day
  - Day of Week
  - Month
  - Quarter
  - Year

- Historical Features:
  - 1-hour Lag
  - 24-hour Lag
  - 7-day Lag
  - 24-hour Rolling Mean
  - 7-day Rolling Mean

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Martin Tay - martin.k.tay@hotmail.com

Project Link: https://github.com/martinktay/xgboost-wind-turbine-power-prediction
