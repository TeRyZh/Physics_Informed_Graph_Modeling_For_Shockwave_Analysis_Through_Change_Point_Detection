"""
Transformer-based Change Point Detector with Wavelet-Enhanced Labeling

This module implements a Transformer detector for comparison with LSTM.
Can be trained with the same wavelet-enhanced labeling modes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import math

from common_utils import WaveletEnhancedDataset, FocalLoss


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerChangePointModel(nn.Module):
    """Transformer model for change point detection."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super(TransformerChangePointModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # FIX: Use last token instead of global average pooling
        # Change points are local events at specific time steps
        # Averaging washes out the sharp signal
        x = x[:, -1, :]  # Take last time step only
        
        x = self.fc(x)
        return x


# ============================================================================
# TRANSFORMER DETECTOR
# ============================================================================

class TransformerChangePointDetector:
    """Transformer-based change point detector (for comparison with LSTM)."""
    
    def __init__(self, sequence_length: int = 100, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, change_point_tolerance: int = 5,
                 confidence_threshold: float = 0.5, device: str = None,
                 labeling_mode: str = 'wavelet_enhanced', loss_type: str = 'weighted_ce'):
        """
        Initialize Transformer change point detector.
        
        Args:
            sequence_length: Length of input sequences
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            change_point_tolerance: Tolerance for labeling
            confidence_threshold: Threshold for detection
            device: 'cuda' or 'cpu'
            labeling_mode: 'moving_window_only', 'wavelet_enhanced', or 'wavelet_only'
            loss_type: 'weighted_ce' (weighted cross entropy) or 'focal' (focal loss)
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.change_point_tolerance = change_point_tolerance
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.labeling_mode = labeling_mode
        self.loss_type = loss_type
        self.model = None
        self.scaler = None
    
    def train(self, train_trajectories: Dict, val_trajectories: Dict = None,
              epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
              wavelet_config: Dict = None):
        """
        Train the Transformer model with wavelet-enhanced labels.
        
        Args:
            train_trajectories: Training data dictionary
            val_trajectories: Validation data dictionary (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            wavelet_config: Configuration for wavelet label generator
        """
        
        print(f"\n{'='*80}")
        print(f"Training Transformer with labeling mode: {self.labeling_mode}")
        print(f"{'='*80}")
        
        train_dataset = WaveletEnhancedDataset(
            train_trajectories,
            sequence_length=self.sequence_length,
            change_point_tolerance=self.change_point_tolerance,
            feature_engineering=True,
            normalize=True,
            labeling_mode=self.labeling_mode,
            wavelet_config=wavelet_config
        )
        
        if len(train_dataset) == 0:
            raise ValueError("No training sequences generated")
        
        self.scaler = train_dataset.scaler if hasattr(train_dataset, 'scaler') else None
        
        input_size = train_dataset.sequences.shape[-1]
        self.model = TransformerChangePointModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_trajectories:
            val_dataset = WaveletEnhancedDataset(
                val_trajectories,
                sequence_length=self.sequence_length,
                change_point_tolerance=self.change_point_tolerance,
                feature_engineering=True,
                normalize=True,
                labeling_mode=self.labeling_mode,
                wavelet_config=wavelet_config
            )
            if len(val_dataset) > 0:
                if self.scaler:
                    original_shape = val_dataset.sequences.shape
                    reshaped_sequences = val_dataset.sequences.reshape(-1, val_dataset.sequences.shape[-1])
                    normalized_sequences = self.scaler.transform(reshaped_sequences)
                    val_dataset.sequences = normalized_sequences.reshape(original_shape)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Calculate class weights to handle imbalance
        pos_count = np.sum(train_dataset.labels == 1)
        neg_count = np.sum(train_dataset.labels == 0)
        total = len(train_dataset.labels)
        
        if pos_count > 0 and neg_count > 0:
            # Choose loss function based on loss_type
            if self.loss_type == 'focal':
                print(f"  Using Focal Loss (alpha=0.25, gamma=2.0)")
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
            else:  # weighted_ce
                class_weights = torch.FloatTensor([
                    total / (2 * neg_count),  # Weight for class 0 (no change)
                    total / (2 * pos_count)   # Weight for class 1 (change point)
                ]).to(self.device)
                print(f"  Using Weighted Cross Entropy")
                print(f"  Class weights: No Change={class_weights[0]:.3f}, Change Point={class_weights[1]:.3f}")
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            print("  Warning: Imbalanced classes detected but using standard loss")
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            if val_loader:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences = sequences.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(sequences)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total
                val_losses.append(avg_val_loss)
                val_accs.append(val_acc)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%")
        
        return train_losses, val_losses, train_accs, val_accs
    
    def detect(self, trajectory: Dict) -> Tuple[List[int], Dict]:
        """
        Detect change points in a trajectory.
        
        Args:
            trajectory: Dictionary with 'time', 'distance', 'velocity'
        
        Returns:
            changepoints: List of detected change point indices
            diagnostics: Dictionary with detection statistics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        time = np.array(trajectory['time'])
        distance = np.array(trajectory['distance'])
        velocity = np.array(trajectory['velocity'])
        
        temp_trajectories = {0: {'time': time, 'distance': distance, 'velocity': velocity}}
        
        dataset = WaveletEnhancedDataset(
            temp_trajectories,
            sequence_length=self.sequence_length,
            change_point_tolerance=self.change_point_tolerance,
            feature_engineering=True,
            normalize=False,
            labeling_mode='moving_window_only'
        )
        
        if len(dataset) == 0:
            return [], {'total_sequences': 0}
        
        if self.scaler:
            original_shape = dataset.sequences.shape
            reshaped_sequences = dataset.sequences.reshape(-1, dataset.sequences.shape[-1])
            normalized_sequences = self.scaler.transform(reshaped_sequences)
            dataset.sequences = normalized_sequences.reshape(original_shape)
        
        self.model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sequence, _ = dataset[i]
                sequence = sequence.unsqueeze(0).to(self.device)
                outputs = self.model(sequence)
                probs = F.softmax(outputs, dim=1)
                confidence = probs[0, 1].item()
                prediction = 1 if confidence >= self.confidence_threshold else 0
                
                predictions.append(prediction)
                confidences.append(confidence)
        
        changepoints = []
        for i, pred in enumerate(predictions):
            if pred == 1:
                pred_idx = dataset.metadata[i]['prediction_idx']
                if pred_idx < len(time):
                    changepoints.append(pred_idx)
        
        changepoints = sorted(list(set(changepoints)))
        
        diagnostics = {
            'total_sequences': len(predictions),
            'positive_predictions': sum(predictions),
            'mean_confidence': np.mean(confidences) if confidences else 0,
            'labeling_mode_used': self.labeling_mode
        }
        
        return changepoints, diagnostics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from common_utils import create_synthetic_trajectory_with_changepoints
    
    print("Transformer Change Point Detector Example")
    print("="*80)
    
    # Create training data
    print("\nCreating training data...")
    train_trajectories = {}
    for i in range(20):
        train_trajectories[i] = create_synthetic_trajectory_with_changepoints()
    
    val_trajectories = {}
    for i in range(5):
        base = create_synthetic_trajectory_with_changepoints()
        noise = np.random.normal(0, 0.8, len(base['velocity']))
        val_trajectories[i] = {
            'time': base['time'],
            'distance': np.cumsum((base['velocity'] + noise) * 0.1),
            'velocity': base['velocity'] + noise,
            'changepoints': base['changepoints']
        }
    
    # Train Transformer with wavelet-enhanced labels
    print("\nTraining Transformer with wavelet-enhanced labels...")
    transformer = TransformerChangePointDetector(
        sequence_length=40,
        labeling_mode='wavelet_enhanced'
    )
    transformer.train(train_trajectories, val_trajectories, epochs=50)
    
    # Test
    print("\nTesting on new trajectory...")
    test_trajectory = create_synthetic_trajectory_with_changepoints()
    test_data = {
        'time': test_trajectory['time'],
        'distance': test_trajectory['distance'],
        'velocity': test_trajectory['velocity']
    }
    
    changepoints, diagnostics = transformer.detect(test_data)
    
    # Evaluate
    true_cps = [np.argmin(np.abs(test_trajectory['time'] - t)) 
               for t in test_trajectory['changepoints']]
    
    tp = sum(1 for cp in changepoints 
            if any(abs(cp - true_cp) <= 10 for true_cp in true_cps))
    fp = len(changepoints) - tp
    fn = len(true_cps) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nResults:")
    print(f"  True change points: {test_trajectory['changepoints']}")
    print(f"  Detected: {len(changepoints)} change points")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    print("\n✅ Transformer example completed successfully!")
