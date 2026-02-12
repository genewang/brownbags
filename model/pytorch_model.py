"""
PyTorch Neural Network Regression Models

This module provides PyTorch-based neural network models for regression tasks.
Supports various architectures including feedforward networks and deep networks.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib
import helpers as hel


class RegressionDataset(Dataset):
    """PyTorch Dataset for regression tasks."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeedforwardNet(nn.Module):
    """Feedforward Neural Network for regression."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(FeedforwardNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class DeepNet(nn.Module):
    """Deep Neural Network with residual connections."""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3):
        super(DeepNet, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn_input = nn.BatchNorm1d(hidden_sizes[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
    
    def forward(self, x):
        x = torch.relu(self.bn_input(self.input_layer(x)))
        
        for i in range(0, len(self.hidden_layers), 4):
            residual = x
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i+1](x)
            x = self.hidden_layers[i+2](x)
            x = self.hidden_layers[i+3](x)
            # Residual connection if dimensions match
            if residual.shape == x.shape:
                x = x + residual
        
        return self.output_layer(x).squeeze()


class PyTorchRegressor:
    """PyTorch-based regressor wrapper compatible with scikit-learn interface."""
    
    def __init__(self, model_type='feedforward', hidden_sizes=[128, 64, 32], 
                 dropout_rate=0.2, learning_rate=0.001, batch_size=32, 
                 epochs=100, early_stopping_patience=10, device=None):
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.input_size = None
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _create_model(self, input_size):
        """Create the appropriate model architecture."""
        if self.model_type == 'feedforward':
            return FeedforwardNet(input_size, self.hidden_sizes, self.dropout_rate)
        elif self.model_type == 'deep':
            return DeepNet(input_size, self.hidden_sizes, self.dropout_rate)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the PyTorch model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.input_size = X_scaled.shape[1]
        
        # Create model
        self.model = self._create_model(self.input_size).to(self.device)
        
        # Create datasets
        train_dataset = RegressionDataset(
            pd.DataFrame(X_scaled, columns=X.columns if hasattr(X, 'columns') else None),
            pd.Series(y) if hasattr(y, 'values') else y
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation dataset if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = RegressionDataset(
                pd.DataFrame(X_val_scaled, columns=X_val.columns if hasattr(X_val, 'columns') else None),
                pd.Series(y_val) if hasattr(y_val, 'values') else y_val
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        self.model.train()
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(self.best_model_state)
                        break
            else:
                scheduler.step(avg_train_loss)
            
            if (epoch + 1) % 10 == 0:
                val_info = f", Val Loss: {val_loss:.4f}" if val_loader else ""
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}{val_info}")
        
        self.model.eval()
        return self
    
    def _validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return val_loss / num_batches if num_batches > 0 else 0
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def get_params(self, deep=True):
        """Get parameters for scikit-learn compatibility."""
        return {
            'model_type': self.model_type,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device
        }
    
    def set_params(self, **params):
        """Set parameters for scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def pytorch_cv_score(estimator, X, y, cv=5, scoring=None):
    """Cross-validation scoring for PyTorch models."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=1)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create a new estimator for each fold
        fold_estimator = PyTorchRegressor(**estimator.get_params())
        fold_estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        
        # Predict and score
        y_pred = fold_estimator.predict(X_val_fold)
        if scoring:
            score = scoring(y_val_fold, y_pred)
        else:
            score = hel.mean_relative_accuracy(y_pred, y_val_fold)
        scores.append(score)
    
    return np.mean(scores)

