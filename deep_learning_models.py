"""
Deep learning models for glucose prediction: RNN and LSTM.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


class RNNModel:
    """
    Simple RNN model for glucose prediction.
    """
    
    def __init__(self, horizon_minutes=30, units=64, epochs=50, batch_size=32):
        """
        Initialize the RNN model.
        
        Parameters:
        -----------
        horizon_minutes : int
            Prediction horizon in minutes (30 or 60)
        units : int
            Number of RNN units
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        """
        self.horizon_minutes = horizon_minutes
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.trained = False
    
    def build_model(self, input_shape):
        """
        Build the RNN model architecture.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (timesteps, features)
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.SimpleRNN(self.units, activation='tanh', return_sequences=True),
            layers.Dropout(0.2),
            layers.SimpleRNN(self.units // 2, activation='tanh'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the RNN model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences (n_samples, n_timesteps)
        y_train : np.ndarray
            Training target values (n_samples,)
        X_val : np.ndarray, optional
            Validation input sequences
        y_val : np.ndarray, optional
            Validation target values
        """
        # Reshape X for RNN (add feature dimension)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val, y_val)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.trained = True
        return history
    
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
        
        # Reshape X for RNN
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()


class LSTMModel:
    """
    LSTM model for glucose prediction.
    """
    
    def __init__(self, horizon_minutes=30, units=64, epochs=50, batch_size=32):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        horizon_minutes : int
            Prediction horizon in minutes (30 or 60)
        units : int
            Number of LSTM units
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        """
        self.horizon_minutes = horizon_minutes
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.trained = False
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (timesteps, features)
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(self.units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.units // 2),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input sequences (n_samples, n_timesteps)
        y_train : np.ndarray
            Training target values (n_samples,)
        X_val : np.ndarray, optional
            Validation input sequences
        y_val : np.ndarray, optional
            Validation target values
        """
        # Reshape X for LSTM (add feature dimension)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val, y_val)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.trained = True
        return history
    
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
        
        # Reshape X for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
