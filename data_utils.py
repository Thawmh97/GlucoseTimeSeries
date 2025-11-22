"""
Data generation and preprocessing utilities for glucose time series.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_glucose_data(n_days=30, sampling_interval_minutes=5, seed=42):
    """
    Generate synthetic glucose time series data.
    
    Parameters:
    -----------
    n_days : int
        Number of days to generate data for
    sampling_interval_minutes : int
        Sampling interval in minutes (default 5 minutes)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp and glucose readings
    """
    np.random.seed(seed)
    
    # Calculate number of samples
    n_samples = int(n_days * 24 * 60 / sampling_interval_minutes)
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i*sampling_interval_minutes) 
                  for i in range(n_samples)]
    
    # Generate base glucose level with daily patterns
    t = np.arange(n_samples)
    
    # Daily cycle (circadian rhythm)
    daily_cycle = 20 * np.sin(2 * np.pi * t / (24 * 60 / sampling_interval_minutes))
    
    # Meal effects (3 meals per day)
    meal_times = []
    for day in range(n_days):
        meal_times.extend([
            day * (24 * 60 / sampling_interval_minutes) + (7 * 60 / sampling_interval_minutes),   # Breakfast at 7am
            day * (24 * 60 / sampling_interval_minutes) + (12 * 60 / sampling_interval_minutes),  # Lunch at 12pm
            day * (24 * 60 / sampling_interval_minutes) + (19 * 60 / sampling_interval_minutes)   # Dinner at 7pm
        ])
    
    meal_effects = np.zeros(n_samples)
    for meal_time in meal_times:
        if meal_time < n_samples:
            # Glucose spike after meal (peaks at 1 hour, returns to baseline in 3 hours)
            for i in range(n_samples):
                time_since_meal = i - meal_time
                if 0 <= time_since_meal <= (3 * 60 / sampling_interval_minutes):
                    meal_effects[i] += 50 * np.exp(-((time_since_meal - 12) ** 2) / 200)
    
    # Random noise
    noise = np.random.normal(0, 5, n_samples)
    
    # Baseline glucose level
    baseline = 100
    
    # Combine all components
    glucose = baseline + daily_cycle + meal_effects + noise
    
    # Ensure glucose values are within realistic range (70-200 mg/dL)
    glucose = np.clip(glucose, 70, 200)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'glucose': glucose
    })
    
    return df


def prepare_sequences(data, lookback_minutes=120, horizon_minutes=[30, 60], 
                      sampling_interval_minutes=5):
    """
    Prepare sequences for time series prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with glucose time series data
    lookback_minutes : int
        How many minutes of history to use for prediction
    horizon_minutes : list
        Prediction horizons in minutes (e.g., [30, 60])
    sampling_interval_minutes : int
        Sampling interval in minutes
        
    Returns:
    --------
    X : np.ndarray
        Input sequences (lookback window)
    y : dict
        Target values for each prediction horizon
    """
    glucose_values = data['glucose'].values
    
    lookback_steps = lookback_minutes // sampling_interval_minutes
    horizon_steps = [h // sampling_interval_minutes for h in horizon_minutes]
    
    X = []
    y = {h: [] for h in horizon_minutes}
    
    for i in range(lookback_steps, len(glucose_values) - max(horizon_steps)):
        # Input sequence
        X.append(glucose_values[i-lookback_steps:i])
        
        # Target values for each horizon
        for h_min, h_steps in zip(horizon_minutes, horizon_steps):
            y[h_min].append(glucose_values[i + h_steps])
    
    X = np.array(X)
    y = {k: np.array(v) for k, v in y.items()}
    
    return X, y


def split_train_test(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : np.ndarray
        Input sequences
    y : dict or np.ndarray
        Target values
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    if isinstance(y, dict):
        y_train = {k: v[:split_idx] for k, v in y.items()}
        y_test = {k: v[split_idx:] for k, v in y.items()}
    else:
        y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test
