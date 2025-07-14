import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lstm import LSTM
from model.transformer import Transformer

class Predictor:
    def __init__(self, sequence_length=90):
        self.sequence_length = sequence_length 
        self.target_scaler = MinMaxScaler(feature_range=(0, 1)) 
        self.feature_scalers = {} 
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = [] 
        self.target = None 

        # Add multi-scale window parameters
        self.multi_scale_enabled = False
        self.weekly_window = 7    # Weekly window
        self.monthly_window = 30  # Monthly window
        self.quarterly_window = 90  # Quarterly window
    
    def set_device(self, use_gpu=False, use_cpu=False):
        """Set computing device"""
        if use_cpu:
            self.device = torch.device('cpu')
            print("Force using CPU for computation")
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
    
    def build_model(self, model_name='lstm'):
        print(model_name)
        input_dim = 1 + len(self.features)

        if model_name == 'lstm':
            model = LSTM(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.3,
            )
        elif model_name == 'transformer':
            model = Transformer(
                input_size=input_dim
            )
        else:
            model = LSTM(
                input_dim=input_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.3,
            )
        model = model.to(self.device)
        self.model = model
        return self.model

    def get_model(self, model_name='lstm'):
        if self.model is None:
            self.build_model(model_name)
        return self.model


    def train_model(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32, prediction_type='short', model_name='lstm'):
        """Train model"""
        model = self.get_model(model_name)
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-7
        )
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience = 20
        best_model_state = None
        
        print(f"Start training {prediction_type}-term model")
        
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            epoch_train_mae = 0
            batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                x, y = batch
                outputs = model(x)
                
                y = y.unsqueeze(1)
                
                loss = criterion(outputs, y)

                mae = torch.mean(torch.abs(outputs - y))
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_mae += mae.item()
                batches += 1
            
            epoch_train_loss /= batches
            epoch_train_mae /= batches
            
            model.eval()
            epoch_val_loss = 0
            epoch_val_mae = 0
            batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    outputs = model(x)
                    
                    y = y.unsqueeze(1)
                    
                    loss = criterion(outputs, y)

                    mae = torch.mean(torch.abs(outputs - y))
                    
                    epoch_val_loss += loss.item()
                    epoch_val_mae += mae.item()
                    batches += 1
            
            epoch_val_loss /= batches
            epoch_val_mae /= batches
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_mae'].append(epoch_train_mae)
            history['val_mae'].append(epoch_val_mae)
            
            scheduler.step(epoch_val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                      f"Train MAE: {epoch_train_mae:.4f}, Val MAE: {epoch_val_mae:.4f}")
            
            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered, stopping training at Epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Restored best model (validation loss: {best_val_loss:.4f})")
        
        return history
    
    def predict(self, X_test, prediction_type='short', model_name='lstm'):
        """Make predictions"""
        model = self.get_model(model_name)
        model.eval()
        X_test = X_test.to(self.device)
        
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
        
        # Check prediction validity
        if not np.isfinite(predictions).all():
            print("Warning: Predictions contain invalid values, cleaning up")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Inverse scaling
        try:
            predictions = self.target_scaler.inverse_transform(predictions)
        except Exception as e:
            print(f"Inverse scaling failed: {e}")
            # If inverse scaling fails, use simple scaling
            predictions = predictions * self.target_scaler.scale_[0] + self.target_scaler.min_[0]
        
        if not np.isfinite(predictions).all():
            print("Warning: Invalid values after inverse scaling, performing final cleanup")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return predictions
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        
        y_true_rescaled = self.target_scaler.inverse_transform(y_true.reshape(-1, 1))
        
        # Calculate MSE and MAE
        mse = mean_squared_error(y_true_rescaled, y_pred)
        mae = mean_absolute_error(y_true_rescaled, y_pred)
        
        return mse, mae
    
    def plot_prediction_comparison(self, y_true, y_pred, dates=None, prediction_type='short', save_path=None):
        """Plot comparison between predicted and true values"""
        plt.figure(figsize=(15, 8))
        # Inverse scale true values
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
            
        y_true_rescaled = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_flat = y_pred.flatten()
        
        if not np.isfinite(y_true_rescaled).all():
            print("Warning: True values contain invalid values, cleaning up")
            y_true_rescaled = np.nan_to_num(y_true_rescaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if not np.isfinite(y_pred_flat).all():
            print("Warning: Predicted values contain invalid values, cleaning up")
            y_pred_flat = np.nan_to_num(y_pred_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if dates is None:
            x_axis = range(len(y_true_rescaled))
            xlabel = 'Time Steps'
        else:
            x_axis = dates
            xlabel = 'Date'
        # Plot comparison curves
        plt.plot(x_axis, y_true_rescaled, label='real', color='blue', linewidth=2, alpha=0.8)
        plt.plot(x_axis, y_pred_flat, label='prediction', color='red', linewidth=2, alpha=0.8)
        
        title_text = 'short term' if prediction_type == 'short' else 'long term'
        plt.title(f'{title_text} power prediction', fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Global Active Power (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if dates is not None:
            plt.xticks(rotation=45)
        
        try:
            mse = mean_squared_error(y_true_rescaled, y_pred_flat)
            mae = mean_absolute_error(y_true_rescaled, y_pred_flat)
            
            if not np.isfinite(mse) or not np.isfinite(mae):
                print("Warning: MSE or MAE calculation results are invalid")
                mse = 0.0 if not np.isfinite(mse) else mse
                mae = 0.0 if not np.isfinite(mae) else mae
            
        except Exception as e:
            print(f"Error calculation failed: {e}")
            mse, mae = 0.0, 0.0
        
        # Add error information
        error_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\ntype: {title_text} prediction'
        plt.text(0.02, 0.98, error_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
        
        return mse, mae
    
    def save_model(self, prediction_type='short'):
        """Save model"""
        model = self.get_model(model_name)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'saved')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{prediction_type}_term_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"{prediction_type}-term model saved to: {model_path}")
    
    def load_model(self, model_path, prediction_type='short'):
        """Load model"""
        model = self.get_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Loaded {prediction_type}-term model: {model_path}")
        return model

    def prepare_data_separate(self, train_data, test_data, target_column='Global_active_power', prediction_days=90):
        """
        Process training and test data separately
        
        Parameters:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            target_column: Target prediction column
            prediction_days: Number of days to predict
            
        Returns:
            Tensors of training and test data
        """
        print("Processing training and test data separately...")
        
        train_data = train_data.sort_values('date').reset_index(drop=True)
        test_data = test_data.sort_values('date').reset_index(drop=True)
        
        train_values = train_data[target_column].values.reshape(-1, 1)
        self.target_scaler.fit(train_values)
        
        scaled_train_values = self.target_scaler.transform(train_values)
        
        test_values = test_data[target_column].values.reshape(-1, 1)
        scaled_test_values = self.target_scaler.transform(test_values)
        
        # Create training sequences
        X_train, y_train = [], []
        for i in range(self.sequence_length, len(scaled_train_values)):
            X_train.append(scaled_train_values[i-self.sequence_length:i, 0])
            y_train.append(scaled_train_values[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Create test sequences
        if len(scaled_test_values) < prediction_days:
            raise ValueError(f"Insufficient test data: need at least {prediction_days} days but only have {len(scaled_test_values)} days")
        

        X_test, y_test = [], []
        

        if len(scaled_test_values) >= self.sequence_length + prediction_days:
            for i in range(self.sequence_length, self.sequence_length + prediction_days):
                X_test.append(scaled_test_values[i-self.sequence_length:i, 0])
                y_test.append(scaled_test_values[i, 0])
        else:

            combined_values = np.concatenate([scaled_train_values[-(self.sequence_length):], scaled_test_values])
            for i in range(self.sequence_length, min(len(combined_values), self.sequence_length + prediction_days)):
                X_test.append(combined_values[i-self.sequence_length:i, 0])
                y_test.append(combined_values[i, 0])
        
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)))
        X_test = torch.FloatTensor(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_multifeature_data_separate(self, train_data, test_data, target_column='Global_active_power', 
                                         feature_columns=None, prediction_days=90):
        """
        Process multi-feature training and test data separately
        """
        print("Processing multi-feature training and test data separately...")
        
        # Ensure data is sorted by date
        train_data = train_data.sort_values('date').reset_index(drop=True)
        test_data = test_data.sort_values('date').reset_index(drop=True)
        
        # Check and handle missing values
        print(f"Training data missing values: {train_data.isnull().sum().sum()}")
        print(f"Test data missing values: {test_data.isnull().sum().sum()}")
        
        # Fill missing values
        train_data = train_data.fillna(method='ffill').fillna(method='bfill')
        test_data = test_data.fillna(method='ffill').fillna(method='bfill')
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in train_data.columns 
                             if col not in ['date', target_column]]
        
        # Filter invalid feature columns
        valid_features = []
        for col in feature_columns:
            if col in train_data.columns and col in test_data.columns:
                train_col_data = train_data[col]
                test_col_data = test_data[col]
                
                if (np.isfinite(train_col_data).all() and np.isfinite(test_col_data).all() and
                    train_col_data.std() > 1e-8):
                    valid_features.append(col)
                else:
                    print(f"Skipping feature {col}: contains invalid values or is constant")
        
        feature_columns = valid_features
        
        # Save feature and target information
        self.features = feature_columns
        self.target = target_column
        
        print(f"Using valid features: {', '.join(feature_columns)}")
        
        # Check target variable
        train_target_values = train_data[target_column].values
        test_target_values = test_data[target_column].values
        
        # Check target variable validity
        if not np.isfinite(train_target_values).all():
            print("Error: Training target variable contains invalid values")
            train_target_values = np.nan_to_num(train_target_values, nan=np.nanmean(train_target_values))
        
        if not np.isfinite(test_target_values).all():
            print("Error: Test target variable contains invalid values")
            test_target_values = np.nan_to_num(test_target_values, nan=np.nanmean(test_target_values))
        
        # Fit scaler using training data
        train_target_values = train_target_values.reshape(-1, 1)
        self.target_scaler.fit(train_target_values)
        scaled_train_target = self.target_scaler.transform(train_target_values)
        
        # Scale test target variable
        test_target_values = test_target_values.reshape(-1, 1)
        scaled_test_target = self.target_scaler.transform(test_target_values)
        
        scaled_train_features = {}
        scaled_test_features = {}
        
        for feature in feature_columns:
            # Fit feature scaler using training data
            train_feature_values = train_data[feature].values.reshape(-1, 1)
            self.feature_scalers[feature] = MinMaxScaler(feature_range=(0, 1))
            scaled_train_features[feature] = self.feature_scalers[feature].fit_transform(train_feature_values)
            
            # Scale test feature
            test_feature_values = test_data[feature].values.reshape(-1, 1)
            scaled_test_features[feature] = self.feature_scalers[feature].transform(test_feature_values)
        
        # Create training sequences
        X_train, y_train = [], []
        feature_count = len(feature_columns)
        
        for i in range(self.sequence_length, len(scaled_train_target)):
            target_seq = scaled_train_target[i-self.sequence_length:i, 0]
            
            if not np.isfinite(target_seq).all():
                continue
            
            feature_seq = np.zeros((self.sequence_length, feature_count))
            valid_sequence = True
            
            for j, feature in enumerate(feature_columns):
                feature_values = scaled_train_features[feature][i-self.sequence_length:i, 0]
                if not np.isfinite(feature_values).all():
                    valid_sequence = False
                    break
                feature_seq[:, j] = feature_values
            
            if not valid_sequence:
                continue
            
            combined_seq = np.column_stack((target_seq.reshape(-1, 1), feature_seq))
            
            if np.isfinite(combined_seq).all() and np.isfinite(scaled_train_target[i, 0]):
                X_train.append(combined_seq)
                y_train.append(scaled_train_target[i, 0])
        
        X_test, y_test = [], []
        
        if len(scaled_test_target) >= self.sequence_length + prediction_days:
            for i in range(self.sequence_length, self.sequence_length + prediction_days):
                target_seq = scaled_test_target[i-self.sequence_length:i, 0]
                
                feature_seq = np.zeros((self.sequence_length, feature_count))
                for j, feature in enumerate(feature_columns):
                    feature_seq[:, j] = scaled_test_features[feature][i-self.sequence_length:i, 0]
                
                combined_seq = np.column_stack((target_seq.reshape(-1, 1), feature_seq))
                
                X_test.append(combined_seq)
                y_test.append(scaled_test_target[i, 0])
        else:
            combined_target = np.concatenate([scaled_train_target[-(self.sequence_length):], scaled_test_target])
            combined_features = {}
            for feature in feature_columns:
                combined_features[feature] = np.concatenate([
                    scaled_train_features[feature][-(self.sequence_length):], 
                    scaled_test_features[feature]
                ])
            
            for i in range(self.sequence_length, min(len(combined_target), self.sequence_length + prediction_days)):
                target_seq = combined_target[i-self.sequence_length:i, 0]
                
                feature_seq = np.zeros((self.sequence_length, feature_count))
                for j, feature in enumerate(feature_columns):
                    feature_seq[:, j] = combined_features[feature][i-self.sequence_length:i, 0]
                
                combined_seq = np.column_stack((target_seq.reshape(-1, 1), feature_seq))
                
                X_test.append(combined_seq)
                y_test.append(combined_target[i, 0])
        
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("No valid training or test sequences generated, please check data quality")
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        return X_train, X_test, y_train, y_test


    def extract_multi_scale_features(self, data, target_column, feature_columns=None):
        """
        Extract multi-scale features
        
        Parameters:
            data: Input data DataFrame
            target_column: Target column
            feature_columns: List of feature columns
            
        Returns:
            DataFrame containing multi-scale features
        """
        print("Extracting multi-scale features...")
        
        # Ensure data is sorted by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Select columns to process
        columns_to_process = [target_column]
        if feature_columns:
            columns_to_process.extend([col for col in feature_columns if col in data.columns])
        
        # Create multi-scale feature DataFrame
        multi_scale_data = data[['date'] + columns_to_process].copy()
        
        for col in columns_to_process:
            values = data[col].values
            
            # Weekly scale features (7 days)
            weekly_mean = self._calculate_rolling_features(values, self.weekly_window, 'mean')
            weekly_std = self._calculate_rolling_features(values, self.weekly_window, 'std')
            weekly_max = self._calculate_rolling_features(values, self.weekly_window, 'max')
            weekly_min = self._calculate_rolling_features(values, self.weekly_window, 'min')
            weekly_trend = self._calculate_trend_features(values, self.weekly_window)
            
            # Monthly scale features (30 days)
            monthly_mean = self._calculate_rolling_features(values, self.monthly_window, 'mean')
            monthly_std = self._calculate_rolling_features(values, self.monthly_window, 'std')
            monthly_max = self._calculate_rolling_features(values, self.monthly_window, 'max')
            monthly_min = self._calculate_rolling_features(values, self.monthly_window, 'min')
            monthly_trend = self._calculate_trend_features(values, self.monthly_window)
            
            # Quarterly scale features (90 days)
            quarterly_mean = self._calculate_rolling_features(values, self.quarterly_window, 'mean')
            quarterly_std = self._calculate_rolling_features(values, self.quarterly_window, 'std')
            quarterly_max = self._calculate_rolling_features(values, self.quarterly_window, 'max')
            quarterly_min = self._calculate_rolling_features(values, self.quarterly_window, 'min')
            quarterly_trend = self._calculate_trend_features(values, self.quarterly_window)
            
            # Add features to DataFrame
            multi_scale_data[f'{col}_weekly_mean'] = weekly_mean
            multi_scale_data[f'{col}_weekly_std'] = weekly_std
            multi_scale_data[f'{col}_weekly_max'] = weekly_max
            multi_scale_data[f'{col}_weekly_min'] = weekly_min
            multi_scale_data[f'{col}_weekly_trend'] = weekly_trend
            
            multi_scale_data[f'{col}_monthly_mean'] = monthly_mean
            multi_scale_data[f'{col}_monthly_std'] = monthly_std
            multi_scale_data[f'{col}_monthly_max'] = monthly_max
            multi_scale_data[f'{col}_monthly_min'] = monthly_min
            multi_scale_data[f'{col}_monthly_trend'] = monthly_trend
            
            multi_scale_data[f'{col}_quarterly_mean'] = quarterly_mean
            multi_scale_data[f'{col}_quarterly_std'] = quarterly_std
            multi_scale_data[f'{col}_quarterly_max'] = quarterly_max
            multi_scale_data[f'{col}_quarterly_min'] = quarterly_min
            multi_scale_data[f'{col}_quarterly_trend'] = quarterly_trend
        
        multi_scale_data = multi_scale_data.iloc[self.quarterly_window:].reset_index(drop=True)
        
        print(f"Generated multi-scale features: {len(multi_scale_data.columns) - len(columns_to_process) - 1}")
        
        return multi_scale_data
    
    def _calculate_rolling_features(self, values, window, stat_type):
        """Calculate rolling statistical features"""
        series = pd.Series(values)
        
        if stat_type == 'mean':
            return series.rolling(window=window, min_periods=1).mean().values
        elif stat_type == 'std':
            return series.rolling(window=window, min_periods=1).std().fillna(0).values
        elif stat_type == 'max':
            return series.rolling(window=window, min_periods=1).max().values
        elif stat_type == 'min':
            return series.rolling(window=window, min_periods=1).min().values
        else:
            return series.rolling(window=window, min_periods=1).mean().values
    
    def _calculate_trend_features(self, values, window):
        """Calculate trend features"""
        series = pd.Series(values)
        trends = []
        
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            window_data = series.iloc[start_idx:end_idx]
            
            if len(window_data) < 2:
                trends.append(0.0)
                continue
            
            # Calculate linear trend slope
            x = np.arange(len(window_data))
            y = window_data.values
            
            try:
                slope = np.polyfit(x, y, 1)[0]
                trends.append(slope)
            except:
                trends.append(0.0)
        
        return np.array(trends)


    def prepare_multiscale_data_separate(self, train_data, test_data, target_column='Global_active_power', 
                                       feature_columns=None, prediction_days=90):
        """
        Process multi-scale training and test data separately
        
        Parameters:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            target_column: Target prediction column
            feature_columns: List of feature columns
            prediction_days: Number of days to predict
            
        Returns:
            Tensors of training and test data
        """
        print("Processing multi-scale training and test data separately...")
        
        # Extract multi-scale features
        train_multiscale = self.extract_multi_scale_features(train_data, target_column, feature_columns)
        test_multiscale = self.extract_multi_scale_features(test_data, target_column, feature_columns)
        
        # Determine multi-scale feature columns
        original_features = feature_columns if feature_columns else []
        multiscale_feature_columns = [col for col in train_multiscale.columns 
                                    if col not in ['date', target_column] and 
                                    (not original_features or any(orig in col for orig in original_features + [target_column]))]
        
        print(f"Using multi-scale features: {len(multiscale_feature_columns)}")
        
        # Use multi-feature data processing method
        return self.prepare_multifeature_data_separate(
            train_multiscale, test_multiscale, target_column, 
            multiscale_feature_columns, prediction_days
        )

    def run_multifeature_prediction_with_plot(self, train_data, test_data, target_column='Global_active_power', feature_columns=None, prediction_days=90, 
prediction_type='short', model_name='lstm', multiscale=False):
        print(f"Starting {prediction_type}-term prediction ({prediction_days} days) and plotting comparison...")
        
        if multiscale:
            self.multi_scale_enabled = True
            X_train, X_test, y_train, y_test = self.prepare_multiscale_data_separate(
                train_data, test_data, target_column, feature_columns, prediction_days
            )
        else:
            X_train, X_test, y_train, y_test = self.prepare_multifeature_data_separate(
                train_data, test_data, target_column, feature_columns, prediction_days
            )
        
        val_split = int(0.8 * len(X_train))
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        X_train, y_train = X_train[:val_split], y_train[:val_split]
        
        self.get_model(model_name)
        self.train_model(X_train, y_train, X_val, y_val, 
                       epochs=150 if prediction_type == 'short' else 200,
                       prediction_type=prediction_type, model_name=model_name)
        
        predictions = self.predict(X_test, prediction_type, model_name=model_name)
        
        test_data_sorted = test_data.sort_values('date').reset_index(drop=True)
        if len(test_data_sorted) >= len(y_test):
            test_dates = test_data_sorted['date'].iloc[:len(y_test)].values
        else:
            import pandas as pd
            start_date = test_data_sorted['date'].iloc[0]
            test_dates = pd.date_range(start=start_date, periods=len(y_test), freq='D').values
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{prediction_type}_term_multifeature_prediction_comparison.png')
        
        mse, mae = self.plot_prediction_comparison(y_test, predictions, test_dates, prediction_type, save_path)
        
        return {
            'predictions': predictions,
            'true_values': y_test,
            'dates': test_dates,
            'mse': mse,
            'mae': mae
        }
