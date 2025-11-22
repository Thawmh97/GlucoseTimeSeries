# GlucoseTimeSeries

Time Series Project to predict glucose levels at 30 minutes and 60 minutes ahead using machine learning models.

## Overview

This project implements and compares multiple machine learning approaches for predicting glucose levels in time series data:
- **Baseline Models**: Logistic Regression (Linear Regression) and ARIMA
- **Deep Learning Models**: RNN and LSTM

## Features

- Synthetic glucose time series data generation with realistic patterns (circadian rhythm, meal effects)
- Data preprocessing and sequence preparation
- Implementation of 4 different prediction models
- Training and evaluation pipeline
- Comprehensive model comparison with multiple metrics (MAE, RMSE, MAPE)
- Visualization of results and predictions

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- statsmodels

## Project Structure

```
GlucoseTimeSeries/
├── data_utils.py              # Data generation and preprocessing utilities
├── baseline_models.py         # Logistic Regression and ARIMA baseline models
├── deep_learning_models.py    # RNN and LSTM models
├── train_models.py            # Main training and evaluation script
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Usage

### Quick Start

Run the complete training and evaluation pipeline:

```bash
python train_models.py
```

This will:
1. Generate synthetic glucose time series data (30 days, 5-minute intervals)
2. Prepare sequences with 2-hour lookback windows
3. Train all 4 models for both 30-minute and 60-minute prediction horizons
4. Evaluate models using MAE, RMSE, and MAPE metrics
5. Generate visualizations comparing model performance
6. Save results to `comparison_results.png` and `prediction_samples.png`

### Custom Usage

You can also use the models individually in your own scripts:

```python
from data_utils import generate_synthetic_glucose_data, prepare_sequences, split_train_test
from baseline_models import LogisticRegressionBaseline, ARIMABaseline
from deep_learning_models import RNNModel, LSTMModel

# Generate or load your data
data = generate_synthetic_glucose_data(n_days=30)

# Prepare sequences
X, y = prepare_sequences(data, lookback_minutes=120, horizon_minutes=[30, 60])

# Split data
X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)

# Train a model (e.g., LSTM for 30-minute prediction)
model = LSTMModel(horizon_minutes=30, units=64, epochs=50)
model.train(X_train, y_train[30])
predictions = model.predict(X_test)
```

## Models

### 1. Logistic Regression (Linear Regression)
- Simple baseline using linear regression on flattened sequences
- Fast training and inference
- Good for establishing baseline performance

### 2. ARIMA (AutoRegressive Integrated Moving Average)
- Classical time series forecasting method
- Captures temporal dependencies and trends
- Traditional statistical approach

### 3. RNN (Recurrent Neural Network)
- Simple recurrent architecture with 2 RNN layers
- Captures sequential dependencies
- Uses dropout for regularization

### 4. LSTM (Long Short-Term Memory)
- Advanced recurrent architecture with LSTM layers
- Better at capturing long-term dependencies
- Uses dropout for regularization

## Data Format

The system expects glucose time series data with the following format:
- **timestamp**: datetime of measurement
- **glucose**: glucose level in mg/dL

The synthetic data generator creates realistic glucose patterns including:
- Circadian rhythm (daily cycles)
- Meal effects (breakfast, lunch, dinner spikes)
- Random variations
- Values constrained to realistic range (70-200 mg/dL)

## Evaluation Metrics

Models are evaluated using three metrics:
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and true values
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

## Results

The system generates two visualization files:
1. **comparison_results.png**: Bar charts comparing all models across metrics
2. **prediction_samples.png**: Line plots showing sample predictions vs. true values

Expected performance (on synthetic data):
- Deep learning models (RNN/LSTM) typically outperform baseline models
- LSTM often shows the best performance due to its ability to capture long-term dependencies
- Performance varies by prediction horizon (30 vs. 60 minutes)

## Future Enhancements

- Support for real glucose monitoring data (CGM data)
- Additional features (meal timing, activity, insulin doses)
- Hyperparameter tuning and model optimization
- Ensemble methods combining multiple models
- Deployment-ready inference pipeline

## License

This project is provided for educational and research purposes.

## Contributors

Developed for glucose level prediction research using time series analysis and machine learning.
