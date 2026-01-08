"""Training and validation functions"""

from typing import Dict, Tuple
from tqdm import tqdm
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch
    
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Validate the model
    
    Returns:
        Average loss, accuracy, AUROC score, and F1 score
    """
    try:
        from sklearn.metrics import roc_auc_score, f1_score
    except ImportError:
        roc_auc_score = None
        f1_score = None
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions for metrics
            all_probs.append(probs[:, 1].cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
    
    val_loss = running_loss / total
    val_acc = 100 * correct / total
    
    # Calculate AUROC and F1 score if sklearn is available
    auroc = 0.0
    f1 = 0.0
    if roc_auc_score is not None and f1_score is not None:
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
            f1 = f1_score(all_labels, all_preds)
        except:
            pass
    
    return val_loss, val_acc, auroc, f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    save_path: str = None,
    start_epoch: int = 0
) -> Dict[str, list]:
    """Train the model with validation
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train (not total, but additional epochs)
        scheduler: Optional learning rate scheduler
        save_path: Path to save checkpoints (e.g., 'checkpoints/model.pth')
                   Will save epoch_N.pth for each epoch and best_model.pth for best
                   Also saves results.json with history and metrics
        start_epoch: Starting epoch number (for resuming training)
        
    Returns:
        Dictionary containing training history
    """
    from pathlib import Path
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auroc': [],
        'val_f1': []
    }
    
    best_acc = 0.0
    save_dir = None
    
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch + 1}/{start_epoch + num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_auroc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if val_auroc > 0:
            print(f"Val AUROC: {val_auroc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save checkpoint for every epoch
        if save_dir:
            try:
                from .utils import save_checkpoint
            except ImportError:
                from utils import save_checkpoint
            
            epoch_path = save_dir / f"epoch_{epoch}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                filepath=epoch_path,
                scheduler=scheduler
            )
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            if save_dir:
                try:
                    from .utils import save_checkpoint
                except ImportError:
                    from utils import save_checkpoint
                best_path = save_dir / "best_model.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_acc=best_acc,
                    filepath=best_path,
                    scheduler=scheduler
                )
                print(f"New best model saved! Accuracy: {best_acc:.2f}%")
    
    print(f"\nTraining complete! Best Val Accuracy: {best_acc:.2f}%")
    
    # Save results to JSON
    if save_dir:
        results_path = save_dir / 'results.json'
        results = {
            'best_accuracy': float(best_acc),
            'num_epochs': num_epochs,
            'history': {
                'train_loss': history['train_loss'],
                'train_acc': history['train_acc'],
                'val_loss': history['val_loss'],
                'val_acc': history['val_acc'],
                'val_auroc': history['val_auroc'],
                'val_f1': history['val_f1']
            }
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")
    
    return history
