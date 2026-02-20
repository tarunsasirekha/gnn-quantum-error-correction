"""
Training Module for GNN Decoders
Includes loss functions, training loops, and evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import time


class DecoderLoss(nn.Module):
    """Loss function for quantum error correction"""
    
    def __init__(self, loss_type: str = 'bce', weight_positive: float = 1.0):
        """
        Initialize loss function
        
        Args:
            loss_type: 'bce' (binary cross entropy), 'focal', or 'weighted_bce'
            weight_positive: Weight for positive class (errors) in weighted BCE
        """
        super().__init__()
        self.loss_type = loss_type
        self.weight_positive = weight_positive
        
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'weighted_bce':
            pos_weight = torch.tensor([weight_positive])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == 'focal':
            self.alpha = 0.25
            self.gamma = 2.0
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Model predictions (logits) [batch_size, 1] or [batch_size]
            targets: Ground truth errors [batch_size, 1] or [batch_size]
        """
        # Ensure both have same shape - squeeze to 1D
        predictions = predictions.squeeze()
        targets = targets.squeeze().float()
        
        if self.loss_type == 'focal':
            # Focal loss for imbalanced data
            bce = nn.functional.binary_cross_entropy_with_logits(
                predictions, targets, reduction='none'
            )
            p_t = torch.exp(-bce)
            focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
            return focal_loss.mean()
        else:
            return self.criterion(predictions, targets)


class LogicalErrorLoss(nn.Module):
    """Loss based on logical error rate"""
    
    def __init__(self, logical_operators: np.ndarray, base_loss_weight: float = 1.0):
        """
        Initialize logical error loss
        
        Args:
            logical_operators: Logical operator matrix [k, n]
            base_loss_weight: Weight for base BCE loss
        """
        super().__init__()
        self.logical_ops = torch.tensor(logical_operators, dtype=torch.float32)
        self.base_loss = nn.BCEWithLogitsLoss()
        self.base_loss_weight = base_loss_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss
        
        Returns:
            (total_loss, metrics_dict)
        """
        # Base BCE loss
        base = self.base_loss(predictions.squeeze(), targets)
        
        # Logical error penalty
        pred_binary = (torch.sigmoid(predictions.squeeze()) > 0.5).float()
        logical_overlap = (self.logical_ops.to(predictions.device) @ pred_binary.unsqueeze(-1)) % 2
        logical_error_rate = logical_overlap.float().mean()
        
        # Combined loss
        total_loss = self.base_loss_weight * base + logical_error_rate
        
        metrics = {
            'base_loss': base.item(),
            'logical_error_rate': logical_error_rate.item()
        }
        
        return total_loss, metrics


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Convert to binary predictions
        pred_probs = torch.sigmoid(predictions.squeeze())
        pred_binary = (pred_probs > threshold).float()
        targets = targets.float()
        
        # Accuracy
        correct = (pred_binary == targets).float()
        accuracy = correct.mean().item()
        
        # Precision, Recall, F1
        tp = ((pred_binary == 1) & (targets == 1)).sum().item()
        fp = ((pred_binary == 1) & (targets == 0)).sum().item()
        fn = ((pred_binary == 0) & (targets == 1)).sum().item()
        tn = ((pred_binary == 0) & (targets == 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Error weight metrics
        true_weight = targets.sum().item()
        pred_weight = pred_binary.sum().item()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'true_error_weight': true_weight,
            'pred_error_weight': pred_weight
        }
        
        return metrics


class Trainer:
    """Trainer for GNN decoders"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize trainer
        
        Args:
            model: GNN decoder model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, data_list: List) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for data in tqdm(data_list, desc="Training"):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(data)
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), data.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(data.y.detach().cpu())
        
        # Compute epoch metrics
        avg_loss = total_loss / len(data_list)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        self.train_losses.append(avg_loss)
        self.train_metrics.append(metrics)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, data_list: List) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for data in data_list:
            data = data.to(self.device)
            
            # Forward pass
            predictions = self.model(data)
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), data.y)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(data.y.cpu())
        
        # Compute metrics
        avg_loss = total_loss / len(data_list)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        self.val_losses.append(avg_loss)
        self.val_metrics.append(metrics)
        
        return metrics
    
    def train(self,
              train_data: List,
              val_data: List,
              num_epochs: int,
              verbose: bool = True) -> Dict[str, List]:
        """
        Full training loop
        
        Args:
            train_data: List of training Data objects
            val_data: List of validation Data objects
            num_epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            History dictionary
        """
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_data)
            
            # Validate
            val_metrics = self.validate(val_data)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = self.model.state_dict().copy()
            
            epoch_time = time.time() - start_time
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }


class DualTrainer(Trainer):
    """Trainer for dual X/Z decoders"""
    
    def train_epoch(self, data_X_list: List, data_Z_list: List) -> Dict[str, float]:
        """Train epoch for dual decoder"""
        self.model.train()
        
        total_loss = 0
        all_pred_X = []
        all_pred_Z = []
        all_target_X = []
        all_target_Z = []
        
        for data_X, data_Z in tqdm(zip(data_X_list, data_Z_list), desc="Training", total=len(data_X_list)):
            data_X = data_X.to(self.device)
            data_Z = data_Z.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_X, pred_Z = self.model(data_X, data_Z)
            
            # Compute loss
            loss_X = self.criterion(pred_X.squeeze(), data_Z.y)  # Z syndrome predicts X errors
            loss_Z = self.criterion(pred_Z.squeeze(), data_X.y)  # X syndrome predicts Z errors
            loss = loss_X + loss_Z
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_pred_X.append(pred_X.detach().cpu())
            all_pred_Z.append(pred_Z.detach().cpu())
            all_target_X.append(data_Z.y.detach().cpu())
            all_target_Z.append(data_X.y.detach().cpu())
        
        # Compute metrics
        avg_loss = total_loss / len(data_X_list)
        all_pred_X = torch.cat(all_pred_X, dim=0)
        all_target_X = torch.cat(all_target_X, dim=0)
        
        metrics = compute_metrics(all_pred_X, all_target_X)
        metrics['loss'] = avg_loss
        
        self.train_losses.append(avg_loss)
        self.train_metrics.append(metrics)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, data_X_list: List, data_Z_list: List) -> Dict[str, float]:
        """Validate dual decoder"""
        self.model.eval()
        
        total_loss = 0
        all_pred_X = []
        all_target_X = []
        
        for data_X, data_Z in zip(data_X_list, data_Z_list):
            data_X = data_X.to(self.device)
            data_Z = data_Z.to(self.device)
            
            pred_X, pred_Z = self.model(data_X, data_Z)
            
            loss_X = self.criterion(pred_X.squeeze(), data_Z.y)
            loss_Z = self.criterion(pred_Z.squeeze(), data_X.y)
            loss = loss_X + loss_Z
            
            total_loss += loss.item()
            all_pred_X.append(pred_X.cpu())
            all_target_X.append(data_Z.y.cpu())
        
        avg_loss = total_loss / len(data_X_list)
        all_pred_X = torch.cat(all_pred_X, dim=0)
        all_target_X = torch.cat(all_target_X, dim=0)
        
        metrics = compute_metrics(all_pred_X, all_target_X)
        metrics['loss'] = avg_loss
        
        self.val_losses.append(avg_loss)
        self.val_metrics.append(metrics)
        
        return metrics


if __name__ == "__main__":
    from qldpc_codes import ToricCode
    from error_simulation import ErrorSimulator, NoiseModel
    from graph_representation import BatchGraphBuilder
    from gnn_models import GNNDecoder
    
    print("=== Testing Training Module ===")
    
    # Create code and simulator
    toric = ToricCode(L=4)
    simulator = ErrorSimulator(toric.H_X, toric.H_Z, NoiseModel(p_depol=0.05))
    
    # Generate datasets
    print("\nGenerating training data...")
    train_dataset = simulator.generate_dataset(n_samples=100)
    val_dataset = simulator.generate_dataset(n_samples=20)
    
    # Create graphs
    print("Creating graphs...")
    batch_builder = BatchGraphBuilder(toric.H_X, toric.H_Z)
    train_data_X, train_data_Z = batch_builder.create_batch(train_dataset)
    val_data_X, val_data_Z = batch_builder.create_batch(val_dataset)
    
    # Create model
    print("\nInitializing model...")
    model = GNNDecoder(input_dim=3, hidden_dim=32, num_layers=3)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = DecoderLoss(loss_type='bce')
    
    trainer = Trainer(model, optimizer, criterion, device='cpu')
    
    # Train
    print("\nTraining...")
    history = trainer.train(
        train_data_Z,  # Z syndrome detects X errors
        val_data_Z,
        num_epochs=3,
        verbose=True
    )
    
    print("\n=== Training completed! ===")
    print(f"Final train accuracy: {history['train_metrics'][-1]['accuracy']:.4f}")
    print(f"Final val accuracy: {history['val_metrics'][-1]['accuracy']:.4f}")
