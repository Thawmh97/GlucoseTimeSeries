"""
Main script to train and compare all glucose prediction models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import generate_synthetic_glucose_data, prepare_sequences, split_train_test
from baseline_models import LogisticRegressionBaseline, ARIMABaseline, evaluate_model
from deep_learning_models import RNNModel, LSTMModel

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Set style
sns.set_style('whitegrid')


def main():
    """
    Main function to train and evaluate all models.
    """
    print("="*80)
    print("Glucose Time Series Prediction")
    print("="*80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic glucose data...")
    data = generate_synthetic_glucose_data(n_days=30, sampling_interval_minutes=5)
    print(f"   Generated {len(data)} samples over 30 days")
    print(f"   Glucose range: {data['glucose'].min():.2f} - {data['glucose'].max():.2f} mg/dL")
    
    # Prepare sequences for prediction
    print("\n2. Preparing sequences for prediction...")
    X, y = prepare_sequences(
        data, 
        lookback_minutes=120,  # Use 2 hours of history
        horizon_minutes=[30, 60],  # Predict 30 and 60 minutes ahead
        sampling_interval_minutes=5
    )
    print(f"   Input sequences shape: {X.shape}")
    print(f"   Target values for 30 min: {y[30].shape}")
    print(f"   Target values for 60 min: {y[60].shape}")
    
    # Split data
    print("\n3. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Dictionary to store results
    results = {}
    
    # Prediction horizons
    horizons = [30, 60]
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"Prediction Horizon: {horizon} minutes")
        print(f"{'='*80}")
        
        results[horizon] = {}
        
        # 1. Logistic Regression (Linear Regression) Baseline
        print(f"\n4. Training Logistic Regression baseline for {horizon} min...")
        lr_model = LogisticRegressionBaseline(horizon_minutes=horizon)
        lr_model.train(X_train, y_train[horizon])
        lr_pred = lr_model.predict(X_test)
        lr_metrics = evaluate_model(y_test[horizon], lr_pred)
        results[horizon]['Logistic Regression'] = lr_metrics
        print(f"   MAE: {lr_metrics['MAE']:.2f}, RMSE: {lr_metrics['RMSE']:.2f}, MAPE: {lr_metrics['MAPE']:.2f}%")
        
        # 2. ARIMA Baseline
        print(f"\n5. Training ARIMA baseline for {horizon} min...")
        arima_model = ARIMABaseline(horizon_minutes=horizon)
        arima_model.train(X_train, y_train[horizon])
        arima_pred = arima_model.predict(X_test)
        arima_metrics = evaluate_model(y_test[horizon], arima_pred)
        results[horizon]['ARIMA'] = arima_metrics
        print(f"   MAE: {arima_metrics['MAE']:.2f}, RMSE: {arima_metrics['RMSE']:.2f}, MAPE: {arima_metrics['MAPE']:.2f}%")
        
        # 3. RNN Model
        print(f"\n6. Training RNN model for {horizon} min...")
        rnn_model = RNNModel(horizon_minutes=horizon, units=64, epochs=50, batch_size=32)
        rnn_model.train(X_train, y_train[horizon])
        rnn_pred = rnn_model.predict(X_test)
        rnn_metrics = evaluate_model(y_test[horizon], rnn_pred)
        results[horizon]['RNN'] = rnn_metrics
        print(f"   MAE: {rnn_metrics['MAE']:.2f}, RMSE: {rnn_metrics['RMSE']:.2f}, MAPE: {rnn_metrics['MAPE']:.2f}%")
        
        # 4. LSTM Model
        print(f"\n7. Training LSTM model for {horizon} min...")
        lstm_model = LSTMModel(horizon_minutes=horizon, units=64, epochs=50, batch_size=32)
        lstm_model.train(X_train, y_train[horizon])
        lstm_pred = lstm_model.predict(X_test)
        lstm_metrics = evaluate_model(y_test[horizon], lstm_pred)
        results[horizon]['LSTM'] = lstm_metrics
        print(f"   MAE: {lstm_metrics['MAE']:.2f}, RMSE: {lstm_metrics['RMSE']:.2f}, MAPE: {lstm_metrics['MAPE']:.2f}%")
        
        # Save predictions for visualization
        results[horizon]['predictions'] = {
            'true': y_test[horizon],
            'Logistic Regression': lr_pred,
            'ARIMA': arima_pred,
            'RNN': rnn_pred,
            'LSTM': lstm_pred
        }
    
    # Print summary
    print(f"\n{'='*80}")
    print("Model Comparison Summary")
    print(f"{'='*80}")
    
    for horizon in horizons:
        print(f"\n{horizon}-minute prediction:")
        print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 55)
        for model_name, metrics in results[horizon].items():
            if model_name != 'predictions':
                print(f"{model_name:<25} {metrics['MAE']:<10.2f} {metrics['RMSE']:<10.2f} {metrics['MAPE']:<10.2f}%")
    
    # Visualize results
    print("\n8. Creating visualizations...")
    visualize_results(results)
    print("   Saved: comparison_results.png")
    print("   Saved: prediction_samples.png")
    
    print("\n" + "="*80)
    print("Training and evaluation completed!")
    print("="*80)


def visualize_results(results):
    """
    Create visualizations comparing model performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for all models and horizons
    """
    horizons = [30, 60]
    models = ['Logistic Regression', 'ARIMA', 'RNN', 'LSTM']
    
    # 1. Comparison bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['MAE', 'RMSE', 'MAPE']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        data = []
        labels = []
        for horizon in horizons:
            for model in models:
                data.append(results[horizon][model][metric])
                labels.append(f"{model}\n({horizon}min)")
        
        x_pos = np.arange(len(labels))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] * 2
        
        ax.bar(x_pos, data, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample predictions
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        
        # Show first 100 samples
        n_samples = min(100, len(results[horizon]['predictions']['true']))
        x_range = np.arange(n_samples)
        
        ax.plot(x_range, results[horizon]['predictions']['true'][:n_samples], 
                'k-', label='True', linewidth=2, alpha=0.7)
        
        colors = ['blue', 'orange', 'green', 'red']
        for model, color in zip(models, colors):
            ax.plot(x_range, results[horizon]['predictions'][model][:n_samples],
                   label=model, alpha=0.6, linewidth=1.5, color=color)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title(f'{horizon}-minute Ahead Predictions')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
