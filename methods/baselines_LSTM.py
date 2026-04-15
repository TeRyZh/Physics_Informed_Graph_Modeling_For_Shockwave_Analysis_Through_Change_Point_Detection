import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings


class TrajectoryChangePointDataset(Dataset):
    """
    Dataset for LSTM-based change point detection on single trajectories.
    
    Creates sequences of trajectory features for temporal change point learning.
    """
    
    def __init__(self, trajectories: Dict, sequence_length: int = 100, 
                 prediction_horizon: int = 1, overlap: float = 0.5,
                 feature_engineering: bool = False, normalize: bool = True,
                 change_point_tolerance: int = 5):
        """
        Initialize dataset for sequence-based change point detection.
        
        Args:
            trajectories: Dictionary of trajectory data {traj_id: {time, distance, velocity, changepoints}} 
            sequence_length: Length of input sequences (number of time steps) 
            prediction_horizon: How many steps ahead to predict change points 
            overlap: Overlap ratio between consecutive sequences (0-1) 
            feature_engineering: Whether to add engineered features
            normalize: Whether to normalize features
            change_point_tolerance: Tolerance window around true change points for labeling
        """
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.overlap = overlap
        self.feature_engineering = feature_engineering
        self.normalize = normalize
        self.change_point_tolerance = change_point_tolerance
        
        # Prepare features and labels
        self.sequences, self.labels, self.metadata = self._prepare_sequences()
        
        # Feature normalization
        if self.normalize and len(self.sequences) > 0:
            self.scaler = StandardScaler() 
            # Reshape for fitting: (n_samples * sequence_length, n_features)
            original_shape = self.sequences.shape
            reshaped_sequences = self.sequences.reshape(-1, self.sequences.shape[-1])
            normalized_sequences = self.scaler.fit_transform(reshaped_sequences)
            self.sequences = normalized_sequences.reshape(original_shape)
        
        print(f"Generated {len(self.sequences)} sequences with {self.sequences.shape[-1]} features each")
        
        # Print label distribution for debugging
        if len(self.labels) > 0:
            unique, counts = np.unique(self.labels, return_counts=True)
            label_dist = dict(zip(unique, counts))
            print(f"Label distribution: {label_dist}")
            print(f"  No Change Point: {label_dist.get(0, 0)} ({100*label_dist.get(0, 0)/len(self.labels):.1f}%)")
            print(f"  Change Point: {label_dist.get(1, 0)} ({100*label_dist.get(1, 0)/len(self.labels):.1f}%)")
    
    def _prepare_sequences(self):
        """Prepare sequences and labels from trajectory data."""
        all_sequences = []
        all_labels = []
        all_metadata = []
        
        for traj_id, traj_data in self.trajectories.items():
            time = np.array(traj_data['time'])
            distance = np.array(traj_data['distance'])
            velocity = np.array(traj_data['velocity'])
            
            # Get change points if available
            changepoints = traj_data.get('changepoints', [])
            changepoint_indices = self._time_to_indices(time, changepoints) if changepoints else []
            
            if len(time) < self.sequence_length + self.prediction_horizon:
                continue  # Skip short trajectories
            
            # Create base features
            features = self._create_features(time, distance, velocity)
            
            # Generate overlapping sequences
            step_size = max(1, int(self.sequence_length * (1 - self.overlap)))
            
            for start_idx in range(0, len(features) - self.sequence_length - self.prediction_horizon + 1, step_size):
                end_idx = start_idx + self.sequence_length
                prediction_idx = end_idx + self.prediction_horizon - 1
                
                # Extract sequence
                sequence = features[start_idx:end_idx]
                
                # Create label based on whether there's a change point in the prediction window
                label = self._create_changepoint_label(
                    changepoint_indices, start_idx, end_idx, prediction_idx
                )
                
                all_sequences.append(sequence)
                all_labels.append(label)
                all_metadata.append({
                    'traj_id': traj_id,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'prediction_idx': prediction_idx,
                    'start_time': time[start_idx],
                    'end_time': time[end_idx-1],
                    'prediction_time': time[prediction_idx] if prediction_idx < len(time) else time[-1]
                })
        
        if not all_sequences:
            return np.array([]), np.array([]), []
        
        return np.array(all_sequences), np.array(all_labels), all_metadata
    
    def _time_to_indices(self, time_array, time_points):
        """Convert time points to array indices."""
        indices = []
        for t in time_points:
            idx = np.argmin(np.abs(time_array - t))
            indices.append(idx)
        return indices
    
    def _create_features(self, time, distance, velocity):
        """Create feature matrix from basic trajectory data."""
        features = []
        
        # Basic features
        features.append(velocity)  # Raw velocity
        features.append(distance)  # Raw distance (position)
        acceleration = np.gradient(velocity, time)
        features.append(acceleration)

        if self.feature_engineering:
            # Temporal derivatives
            # acceleration = np.gradient(velocity, time)
            jerk = np.gradient(acceleration, time)
            features.append(jerk)
            
            # Moving statistics (window size = 5)
            window_size = 5
            velocity_ma = self._moving_average(velocity, window_size)
            velocity_std = self._moving_std(velocity, window_size)
            acceleration_ma = self._moving_average(acceleration, window_size)
            features.extend([velocity_ma, velocity_std, acceleration_ma])
            
            # Velocity change indicators
            velocity_diff = np.diff(velocity, prepend=velocity[0])
            velocity_diff_ma = self._moving_average(velocity_diff, window_size)
            features.extend([velocity_diff, velocity_diff_ma])
            
            # Speed regime indicators
            is_low_speed = (velocity < 15).astype(float)  # Below 15 ft/s
            is_high_speed = (velocity > 50).astype(float)  # Above 50 ft/s
            is_medium_speed = ((velocity >= 15) & (velocity <= 50)).astype(float)
            features.extend([is_low_speed, is_medium_speed, is_high_speed])
            
            # Trend indicators
            velocity_trend = self._calculate_trend(velocity, window_size)
            acceleration_trend = self._calculate_trend(acceleration, window_size)
            features.extend([velocity_trend, acceleration_trend])
            
            # Relative time and position (normalized)
            normalized_time = (time - time[0]) / (time[-1] - time[0]) if len(time) > 1 else np.zeros_like(time)
            normalized_distance = (distance - distance[0]) / (distance[-1] - distance[0]) if distance[-1] != distance[0] else np.zeros_like(distance)
            features.extend([normalized_time, normalized_distance])
            
            # Velocity variability indicators
            velocity_rolling_var = self._moving_variance(velocity, window_size)
            features.append(velocity_rolling_var)
        
        # Stack features and transpose to get shape (n_timesteps, n_features)
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values that might arise from gradients/moving averages
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    def _moving_average(self, data, window_size):
        """Calculate moving average."""
        if len(data) < window_size:
            return np.full_like(data, np.mean(data))
        
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            result[i] = np.mean(data[start_idx:end_idx])
        return result
    
    def _moving_std(self, data, window_size):
        """Calculate moving standard deviation."""
        if len(data) < window_size:
            return np.full_like(data, np.std(data))
        
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            result[i] = np.std(data[start_idx:end_idx])
        return result
    
    def _moving_variance(self, data, window_size):
        """Calculate moving variance."""
        if len(data) < window_size:
            return np.full_like(data, np.var(data))
        
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            result[i] = np.var(data[start_idx:end_idx])
        return result
    
    def _calculate_trend(self, data, window_size):
        """Calculate local trend (slope) over a moving window."""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            
            if end_idx - start_idx >= 2:
                x = np.arange(end_idx - start_idx)
                y = data[start_idx:end_idx]
                try:
                    result[i] = np.polyfit(x, y, 1)[0]  # Slope
                except:
                    result[i] = 0
        return result
    
    def _create_changepoint_label(self, changepoint_indices, start_idx, end_idx, prediction_idx):
        """
        Create binary label for change point detection.
        
        Label is 1 if there's a change point within tolerance of the prediction point,
        0 otherwise.
        """
        if not changepoint_indices:
            return 0  # No change points in this trajectory
        
        # Check if any change point is within tolerance of the prediction index
        for cp_idx in changepoint_indices:
            if abs(cp_idx - prediction_idx) <= self.change_point_tolerance:
                return 1
        
        return 0
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sequence, label


class ChangePointLSTM(nn.Module):
    """
    Bidirectional LSTM model for change point detection in single trajectories.
    
    Uses bidirectional processing and attention to identify change points
    in time series data.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 dropout=0.3, attention=True):
        """
        Initialize the LSTM change point detector.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention: Whether to use attention mechanism
        """
        super(ChangePointLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism (optional)
        if attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Classification layers for binary change point detection
        lstm_output_size = hidden_size * 2  # Bidirectional
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(hidden_size // 2, 2)  # Binary classification: 0=no change, 1=change point
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.attention:
            # Apply attention mechanism
            attention_weights = self.attention_layer(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # Use the last output
            attended_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(attended_output)
        
        return output


class LSTMChangePointDetector:
    """
    Main class for LSTM-based change point detection on single trajectories.
    """
    
    def __init__(self, sequence_length=50, hidden_size=64, num_layers=2,
                 change_point_tolerance=5, confidence_threshold=0.7,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the LSTM-based change point detector.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Hidden size for LSTM
            num_layers: Number of LSTM layers
            change_point_tolerance: Tolerance for labeling change points during training
            confidence_threshold: Confidence threshold for predictions
            device: Device to run the model on
        """
        self.device = device
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.change_point_tolerance = change_point_tolerance
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.training_scaler = None
    
    def _initialize_model(self, input_size):
        """Initialize the model with the correct input size."""
        self.model = ChangePointLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.3,
            attention=True
        ).to(self.device)
        
        # Use weighted loss to handle class imbalance
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
    
    def detect(self, trajectory_data):
        """
        Detect change points in a single trajectory using the trained LSTM model.
        
        Args:
            trajectory_data: Dictionary containing 'time', 'distance', 'velocity' arrays
            
        Returns:
            changepoints: List of detected changepoint indices
            diagnostics: Additional information about the detection
        """
        if self.model is None:
            return [], {"error": "Model not trained. Call train() first."}
        
        # Create dataset from single trajectory (without ground truth change points)
        trajectories = {0: trajectory_data}
        dataset = TrajectoryChangePointDataset(
            trajectories, 
            sequence_length=self.sequence_length,
            change_point_tolerance=self.change_point_tolerance,
            normalize=False  # We'll use the training scaler
        )
        
        if len(dataset) == 0:
            return [], {"message": "No valid sequences generated from trajectory"}
        
        # Apply the same normalization as training
        if hasattr(self, 'training_scaler') and self.training_scaler is not None:
            original_shape = dataset.sequences.shape
            reshaped_sequences = dataset.sequences.reshape(-1, dataset.sequences.shape[-1])
            normalized_sequences = self.training_scaler.transform(reshaped_sequences)
            dataset.sequences = normalized_sequences.reshape(original_shape)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        confidences = []
        prediction_indices = []
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                
                outputs = self.model(sequences)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                max_confidence = torch.max(probabilities, dim=1)[0]
                
                predictions.extend(predicted_class.cpu().numpy())
                confidences.extend(max_confidence.cpu().numpy())
        
        # Extract change points from high-confidence positive predictions
        changepoints = []
        for i, (pred, conf, meta) in enumerate(zip(predictions, confidences, dataset.metadata)):
            if pred == 1 and conf >= self.confidence_threshold:
                prediction_idx = meta['prediction_idx']
                prediction_indices.append(prediction_idx)
                changepoints.append(prediction_idx)
        
        # Post-process change points to remove duplicates and nearby points
        changepoints = self._post_process_changepoints(
            changepoints, trajectory_data, min_distance=10
        )
        
        # Calculate additional diagnostics
        positive_predictions = sum(predictions)
        high_confidence_predictions = sum(1 for c in confidences if c >= self.confidence_threshold)
        
        diagnostics = {
            "lstm": {
                "sequence_length": self.sequence_length,
                "total_sequences": len(predictions),
                "positive_predictions": positive_predictions,
                "high_confidence_predictions": high_confidence_predictions,
                "mean_confidence": np.mean(confidences),
                "confidence_threshold": self.confidence_threshold,
                "raw_predictions": predictions,
                "raw_confidences": confidences
            },
            "changepoints": changepoints,
            "num_segments": len(changepoints) + 1,
            "avg_segment_length": len(trajectory_data['time']) / (len(changepoints) + 1) if changepoints else len(trajectory_data['time'])
        }
        
        return changepoints, diagnostics
    
    def _post_process_changepoints(self, changepoints, trajectory_data, min_distance=10):
        """Post-process change points to remove duplicates and nearby points."""
        if not changepoints:
            return []
        
        # Sort change points
        changepoints = sorted(list(set(changepoints)))
        
        # Remove change points that are too close to each other
        filtered_changepoints = []
        for cp in changepoints:
            if not filtered_changepoints or cp - filtered_changepoints[-1] >= min_distance:
                filtered_changepoints.append(cp)
        
        # Ensure change points are within valid range
        max_idx = len(trajectory_data['time']) - 1
        filtered_changepoints = [cp for cp in filtered_changepoints if 0 <= cp <= max_idx]
        
        return filtered_changepoints
    
    def train(self, train_trajectories, val_trajectories=None, epochs=100, batch_size=32):
        """
        Train the LSTM model on trajectories with known change points.
        
        Args:
            train_trajectories: Dictionary of training trajectories with 'changepoints' key
            val_trajectories: Dictionary of validation trajectories (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create training dataset
        train_dataset = TrajectoryChangePointDataset(
            train_trajectories,
            sequence_length=self.sequence_length,
            change_point_tolerance=self.change_point_tolerance,
            normalize=True
        )
        
        if len(train_dataset) == 0:
            raise ValueError("No valid training sequences generated")
        
        # Store the scaler for later use
        self.training_scaler = train_dataset.scaler
        
        # Initialize model
        input_size = train_dataset.sequences.shape[-1]
        self._initialize_model(input_size)
        
        # Create validation dataset if provided
        val_dataset = None
        if val_trajectories:
            val_dataset = TrajectoryChangePointDataset(
                val_trajectories,
                sequence_length=self.sequence_length,
                change_point_tolerance=self.change_point_tolerance,
                normalize=False  # Use training scaler
            )
            
            if len(val_dataset) > 0:
                # Apply training scaler to validation data
                original_shape = val_dataset.sequences.shape
                reshaped_sequences = val_dataset.sequences.reshape(-1, val_dataset.sequences.shape[-1])
                normalized_sequences = self.training_scaler.transform(reshaped_sequences)
                val_dataset.sequences = normalized_sequences.reshape(original_shape)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"Training on {len(train_dataset)} sequences with {input_size} features")
        print(f"Change point tolerance: {self.change_point_tolerance}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences = sequences.to(self.device)
                        labels = labels.squeeze().to(self.device)
                        
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            if val_loader:
                val_loss /= len(val_loader)
                val_acc = 100 * val_correct / val_total
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_lstm_changepoint_model.pth')
            else:
                val_loss = 0
                val_acc = 0
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                if val_loader:
                    print(f'           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def visualize_predictions(self, trajectory_data, predictions_info, figsize=(15, 10)):
        """Visualize trajectory with LSTM change point predictions."""
        time = trajectory_data['time']
        distance = trajectory_data['distance']
        velocity = trajectory_data['velocity']
        changepoints = predictions_info.get('changepoints', [])
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot trajectory
        axes[0].plot(time, distance, 'b-', linewidth=1.5, label='Distance')
        axes[0].set_ylabel('Distance (ft)')
        axes[0].set_title('Vehicle Trajectory with LSTM Change Point Detection')
        axes[0].grid(True, alpha=0.3)
        
        # Plot velocity
        axes[1].plot(time, velocity, 'g-', linewidth=1.5, label='Velocity')
        axes[1].set_ylabel('Velocity (ft/s)')
        axes[1].grid(True, alpha=0.3)
        
        # Add detected change points
        for cp_idx in changepoints:
            if cp_idx < len(time):
                cp_time = time[cp_idx]
                axes[0].axvline(cp_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
                axes[1].axvline(cp_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
                axes[0].plot(cp_time, distance[cp_idx], 'ro', markersize=8)
                axes[1].plot(cp_time, velocity[cp_idx], 'ro', markersize=8)
        
        # Plot prediction confidence over time (if available)
        lstm_info = predictions_info.get('lstm', {})
        confidences = lstm_info.get('raw_confidences', [])
        predictions = lstm_info.get('raw_predictions', [])
        
        if confidences and len(confidences) > 0:
            # Create approximate time points for predictions (simplified)
            pred_times = np.linspace(time[0], time[-1], len(confidences))
            
            # Plot confidence
            axes[2].plot(pred_times, confidences, 'k-', linewidth=1, label='Prediction Confidence')
            
            # Color code by prediction
            for i, (pred, conf, pred_time) in enumerate(zip(predictions, confidences, pred_times)):
                color = 'red' if pred == 1 else 'green'
                alpha = min(1.0, conf)
                axes[2].scatter(pred_time, conf, c=color, alpha=alpha, s=20)
            
            axes[2].axhline(y=self.confidence_threshold, color='orange', 
                           linestyle=':', alpha=0.7, label=f'Confidence Threshold ({self.confidence_threshold})')
            axes[2].set_ylabel('Confidence')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
        else:
            # If no confidence data, just show detected change points
            axes[2].set_ylabel('Detected Change Points')
            cp_signal = np.zeros_like(time)
            for cp_idx in changepoints:
                if cp_idx < len(time):
                    cp_signal[cp_idx] = 1
            axes[2].plot(time, cp_signal, 'r-', linewidth=2, label='Change Points')
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].legend()
        
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True, alpha=0.3)
        
        # Add legends
        axes[0].legend()
        axes[1].legend()
        
        plt.tight_layout()
        return fig


# Example usage and testing
def create_synthetic_trajectory_with_changepoints():
    """Create synthetic trajectory data with known change points for testing."""
    np.random.seed(42)
    time = np.linspace(0, 120, 1200)  # 2 minutes, 0.1s resolution
    
    # Base velocity with regime changes at known points
    velocity = np.full_like(time, 35.0)  # Start with constant speed
    
    # Add regime changes (these will be our ground truth change points)
    changepoint_times = [30, 45, 70, 85]  # Times where regimes change
    changepoint_indices = [np.argmin(np.abs(time - t)) for t in changepoint_times]
    
    # Regime 1: t=0-30, steady speed ~35 ft/s
    mask1 = time <= 30
    velocity[mask1] = 35 + np.random.normal(0, 2, np.sum(mask1))
    
    # Regime 2: t=30-45, deceleration to ~15 ft/s  
    mask2 = (time > 30) & (time <= 45)
    decel_factor = 1 - 0.6 * (time[mask2] - 30) / 15
    velocity[mask2] = 35 * decel_factor + np.random.normal(0, 1.5, np.sum(mask2))
    
    # Regime 3: t=45-70, low speed ~15 ft/s
    mask3 = (time > 45) & (time <= 70)
    velocity[mask3] = 15 + np.random.normal(0, 1, np.sum(mask3))
    
    # Regime 4: t=70-85, acceleration back to ~25 ft/s
    mask4 = (time > 70) & (time <= 85)
    accel_factor = (time[mask4] - 70) / 15
    velocity[mask4] = 15 + 10 * accel_factor + np.random.normal(0, 1.5, np.sum(mask4))
    
    # Regime 5: t=85-120, steady speed ~25 ft/s
    mask5 = time > 85
    velocity[mask5] = 25 + np.random.normal(0, 2, np.sum(mask5))
    
    # Ensure no negative velocities
    velocity = np.maximum(velocity, 1.0)
    
    # Calculate distance
    distance = np.cumsum(velocity * 0.1)  # 0.1s time steps
    
    return {
        'time': time,
        'distance': distance, 
        'velocity': velocity,
        'changepoints': changepoint_times  # Ground truth change points
    }


def example_lstm_changepoint_detection():
    """Example demonstrating LSTM-based change point detection."""
    
    print("Creating synthetic trajectories with known change points...")
    
    # Create multiple synthetic trajectories for training
    train_trajectories = {}
    for i in range(50):  # 50 training trajectories
        # Create variation in each trajectory
        base_traj = create_synthetic_trajectory_with_changepoints()
        
        # Add variation to timing and values
        time_noise = np.random.uniform(-2, 2, len(base_traj['changepoints']))
        modified_changepoints = [max(10, cp + noise) for cp, noise in 
                               zip(base_traj['changepoints'], time_noise)]
        
        # Add noise to velocity
        velocity_noise = np.random.normal(0, 1, len(base_traj['velocity']))
        modified_velocity = base_traj['velocity'] + velocity_noise
        modified_velocity = np.maximum(modified_velocity, 1.0)  # Ensure positive
        
        # Recalculate distance
        modified_distance = np.cumsum(modified_velocity * 0.1)
        
        train_trajectories[i] = {
            'time': base_traj['time'] + i * 0.5,  # Slight time offset
            'distance': modified_distance,
            'velocity': modified_velocity,
            'changepoints': modified_changepoints
        }
    
    # Create validation trajectories
    val_trajectories = {}
    for i in range(10):  # 10 validation trajectories
        base_traj = create_synthetic_trajectory_with_changepoints()
        
        # Different variation pattern for validation
        time_noise = np.random.uniform(-1, 1, len(base_traj['changepoints']))
        modified_changepoints = [max(10, cp + noise) for cp, noise in 
                               zip(base_traj['changepoints'], time_noise)]
        
        velocity_noise = np.random.normal(0, 0.8, len(base_traj['velocity']))
        modified_velocity = base_traj['velocity'] + velocity_noise
        modified_velocity = np.maximum(modified_velocity, 1.0)
        
        modified_distance = np.cumsum(modified_velocity * 0.1)
        
        val_trajectories[i] = {
            'time': base_traj['time'] + i * 0.3,
            'distance': modified_distance,
            'velocity': modified_velocity,
            'changepoints': modified_changepoints
        }
    
    print(f"Created {len(train_trajectories)} training and {len(val_trajectories)} validation trajectories")
    
    # Initialize and train detector
    detector = LSTMChangePointDetector(
        sequence_length=40,
        hidden_size=64,
        num_layers=2,
        change_point_tolerance=8,  # Tolerance for labeling
        confidence_threshold=0.6
    )
    
    print("Training LSTM model...")
    train_losses, val_losses, train_accs, val_accs = detector.train(
        train_trajectories, 
        val_trajectories,
        epochs=80, 
        batch_size=32
    )
    
    # Test on a new trajectory
    print("\nTesting on a new trajectory...")
    test_trajectory = create_synthetic_trajectory_with_changepoints()
    
    # Remove changepoints from test data (detector shouldn't see them)
    test_data = {
        'time': test_trajectory['time'],
        'distance': test_trajectory['distance'],
        'velocity': test_trajectory['velocity']
    }
    
    # Detect change points
    changepoints, diagnostics = detector.detect(test_data)
    
    print(f"\nTrue change points (times): {test_trajectory['changepoints']}")
    print(f"True change points (indices): {[np.argmin(np.abs(test_trajectory['time'] - t)) for t in test_trajectory['changepoints']]}")
    print(f"Detected change points (indices): {changepoints}")
    print(f"Detected change points (times): {[test_trajectory['time'][cp] for cp in changepoints if cp < len(test_trajectory['time'])]}")
    
    print(f"\nDetection Summary:")
    print(f"  Total sequences analyzed: {diagnostics['lstm']['total_sequences']}")
    print(f"  Positive predictions: {diagnostics['lstm']['positive_predictions']}")
    print(f"  High confidence predictions: {diagnostics['lstm']['high_confidence_predictions']}")
    print(f"  Mean confidence: {diagnostics['lstm']['mean_confidence']:.3f}")
    print(f"  Number of segments: {diagnostics['num_segments']}")
    print(f"  Average segment length: {diagnostics['avg_segment_length']:.1f}")
    
    # Evaluate detection performance
    true_changepoints = [np.argmin(np.abs(test_trajectory['time'] - t)) 
                        for t in test_trajectory['changepoints']]
    
    # Calculate precision, recall, F1
    tp = 0  # true positives
    fp = 0  # false positives
    tolerance = 10  # indices tolerance for matching
    
    for detected_cp in changepoints:
        matched = False
        for true_cp in true_changepoints:
            if abs(detected_cp - true_cp) <= tolerance:
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1
    
    fn = len(true_changepoints) - tp  # false negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    fig = detector.visualize_predictions(test_data, diagnostics)
    plt.show()
    
    # Plot training history
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(train_losses, label='Training Loss')
    if val_losses:
        ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Training Accuracy')
    if val_accs:
        ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot confidence distribution
    confidences = diagnostics['lstm']['raw_confidences']
    predictions = diagnostics['lstm']['raw_predictions']
    
    ax3.hist([c for c, p in zip(confidences, predictions) if p == 0], 
             alpha=0.7, label='No Change Point', bins=20, color='green')
    ax3.hist([c for c, p in zip(confidences, predictions) if p == 1], 
             alpha=0.7, label='Change Point', bins=20, color='red')
    ax3.axvline(detector.confidence_threshold, color='orange', linestyle='--', 
                label=f'Threshold ({detector.confidence_threshold})')
    ax3.set_title('Prediction Confidence Distribution')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot detection timeline
    pred_timeline = np.zeros_like(test_trajectory['time'])
    for cp in changepoints:
        if cp < len(pred_timeline):
            pred_timeline[cp] = 1
    
    true_timeline = np.zeros_like(test_trajectory['time'])
    for cp in true_changepoints:
        if cp < len(true_timeline):
            true_timeline[cp] = 1
    
    ax4.plot(test_trajectory['time'], true_timeline + 0.1, 'g-', linewidth=3, 
             label='True Change Points', alpha=0.8)
    ax4.plot(test_trajectory['time'], pred_timeline - 0.1, 'r-', linewidth=3, 
             label='Detected Change Points', alpha=0.8)
    ax4.set_title('Change Point Detection Timeline')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Detection')
    ax4.set_ylim(-0.5, 0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return detector, test_data, changepoints, diagnostics


if __name__ == "__main__":
    print("LSTM Change Point Detection Example")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run the example
    detector, test_data, changepoints, diagnostics = example_lstm_changepoint_detection()
    
    print("\nExample completed successfully!")
    print("The detector has been trained and tested on synthetic trajectory data.")
    print("You can now use the detector on real trajectory data by calling:")
    print("  changepoints, diagnostics = detector.detect(your_trajectory_data)")
    print("\nWhere your_trajectory_data should be a dictionary with keys:")
    print("  - 'time': array of time points")
    print("  - 'distance': array of position/distance values") 
    print("  - 'velocity': array of velocity values")