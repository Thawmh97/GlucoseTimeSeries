"""
Baseline models for glucose prediction: Logistic Regression and ARIMA.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline model for glucose prediction.
    Note: Despite the name, this is actually Linear Regression (not Logistic)
    as we're doing regression, not classification.
    """
    
    def __init__(self, horizon_minutes=30):
        """
        Initialize the model.
        
        Parameters:
        -----------
        horizon_minutes : int
            Prediction horizon in minutes (30 or 60)
        """
        self.horizon_minutes = horizon_minutes
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
    
    def train(self, X_train, y_train):
        """
        Train the linear regression model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences (n_samples, n_timesteps)
        y_train : np.ndarray
            Training target values (n_samples,)
        """
        # Flatten sequences for linear regression
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.trained = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences (n_samples, n_timesteps)
            
        Returns:
        --------
        np.ndarray
            Predictions (n_samples,)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Flatten and scale
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        return predictions


class ARIMABaseline:
    """
    ARIMA baseline model for glucose prediction.
    """
    
    def __init__(self, horizon_minutes=30, order=(2, 1, 2)):
        """
        Initialize the ARIMA model.
        
        Parameters:
        -----------
        horizon_minutes : int
            Prediction horizon in minutes (30 or 60)
        order : tuple
            ARIMA order (p, d, q)
        """
        self.horizon_minutes = horizon_minutes
        self.order = order
        self.trained = False
        self.last_sequence = None
    
    def train(self, X_train, y_train):
        """
        Train the ARIMA model.
        Note: ARIMA trains on individual sequences, so we'll store training data.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences (n_samples, n_timesteps)
        y_train : np.ndarray
            Training target values (n_samples,)
        """
        # Store the last training sequence for initialization
        self.last_sequence = X_train[-1]
        self.trained = True
    
    def predict(self, X):
        """
        Make predictions using ARIMA.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences (n_samples, n_timesteps)
            
        Returns:
        --------
        np.ndarray
            Predictions (n_samples,)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        
        for sequence in X:
            try:
                # Fit ARIMA model on the sequence
                model = ARIMA(sequence, order=self.order)
                model_fit = model.fit()
                
                # Forecast steps ahead
                horizon_steps = self.horizon_minutes // 5  # Assuming 5-minute intervals
                forecast = model_fit.forecast(steps=horizon_steps)
                
                # Take the prediction at the specified horizon
                predictions.append(forecast[-1])
            except:
                # If ARIMA fails, use simple persistence model (last value)
                predictions.append(sequence[-1])
        
        return np.array(predictions)


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
