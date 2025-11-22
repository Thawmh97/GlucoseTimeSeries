"""
Example script demonstrating how to use the glucose prediction models.
"""
from data_utils import generate_synthetic_glucose_data, prepare_sequences, split_train_test
from baseline_models import LogisticRegressionBaseline, ARIMABaseline, evaluate_model
from deep_learning_models import LSTMModel

# Set random seed for reproducibility
import numpy as np
np.random.seed(42)

def example_single_model():
    """
    Example of training and using a single model.
    """
    print("Example: Training LSTM model for 30-minute glucose prediction")
    print("=" * 70)
    
    # 1. Generate or load data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_glucose_data(n_days=30, sampling_interval_minutes=5)
    print(f"   Generated {len(data)} samples")
    
    # 2. Prepare sequences
    print("\n2. Preparing sequences...")
    X, y = prepare_sequences(
        data, 
        lookback_minutes=120,  # Use 2 hours of history
        horizon_minutes=[30],   # Predict 30 minutes ahead
        sampling_interval_minutes=5
    )
    print(f"   X shape: {X.shape}, y[30] shape: {y[30].shape}")
    
    # 3. Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Train model
    print("\n4. Training LSTM model...")
    model = LSTMModel(horizon_minutes=30, units=64, epochs=50, batch_size=32)
    model.train(X_train, y_train[30])
    print("   Training completed!")
    
    # 5. Make predictions
    print("\n5. Making predictions...")
    predictions = model.predict(X_test)
    print(f"   Generated {len(predictions)} predictions")
    
    # 6. Evaluate
    print("\n6. Evaluating model...")
    metrics = evaluate_model(y_test[30], predictions)
    print(f"   MAE: {metrics['MAE']:.2f} mg/dL")
    print(f"   RMSE: {metrics['RMSE']:.2f} mg/dL")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    # 7. Example prediction
    print("\n7. Example prediction:")
    sample_idx = 0
    print(f"   Input sequence (last 5 values): {X_test[sample_idx][-5:]}")
    print(f"   Predicted glucose (30 min): {predictions[sample_idx]:.2f} mg/dL")
    print(f"   Actual glucose (30 min): {y_test[30][sample_idx]:.2f} mg/dL")
    print(f"   Error: {abs(predictions[sample_idx] - y_test[30][sample_idx]):.2f} mg/dL")


def example_multiple_horizons():
    """
    Example of training models for multiple prediction horizons.
    """
    print("\n\nExample: Training for multiple prediction horizons")
    print("=" * 70)
    
    # Generate data
    data = generate_synthetic_glucose_data(n_days=30, sampling_interval_minutes=5)
    
    # Prepare sequences for both 30 and 60 minute predictions
    X, y = prepare_sequences(
        data, 
        lookback_minutes=120,
        horizon_minutes=[30, 60],  # Multiple horizons
        sampling_interval_minutes=5
    )
    
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    
    # Train separate models for each horizon
    for horizon in [30, 60]:
        print(f"\nTraining LSTM for {horizon}-minute prediction...")
        model = LSTMModel(horizon_minutes=horizon, units=64, epochs=30, batch_size=32)
        model.train(X_train, y_train[horizon])
        
        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test[horizon], predictions)
        
        print(f"  Results: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")


def example_compare_models():
    """
    Example of comparing different models.
    """
    print("\n\nExample: Comparing different models")
    print("=" * 70)
    
    # Generate data
    data = generate_synthetic_glucose_data(n_days=30, sampling_interval_minutes=5)
    X, y = prepare_sequences(data, lookback_minutes=120, horizon_minutes=[30])
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    
    models = {
        'Logistic Regression': LogisticRegressionBaseline(horizon_minutes=30),
        'ARIMA': ARIMABaseline(horizon_minutes=30),
        'LSTM': LSTMModel(horizon_minutes=30, units=64, epochs=30, batch_size=32)
    }
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\n{name}:")
        model.train(X_train, y_train[30])
        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test[30], predictions)
        print(f"  MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")


if __name__ == "__main__":
    # Run examples
    example_single_model()
    example_multiple_horizons()
    example_compare_models()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
