import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, NumPy, and PyTorch.

    Args:
        seed (int): Random seed to set for reproducibility.

    Raises:
        ValueError: If the seed is negative.
    """
    if seed < 0:
        logger.error(f"Seed must be non-negative, got {seed}")
        raise ValueError("Seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")

def initialize_weights(module: nn.Module) -> None:
    """Initialize weights for Conv2d and BatchNorm2d layers in a neural network.

    Applies Kaiming normal initialization for Conv2d layers (suitable for ReLU) and constant
    initialization for BatchNorm2d layers (weights=1, biases=0).

    Args:
        module (nn.Module): PyTorch module (e.g., Conv2d or BatchNorm2d) to initialize.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> float:
    """Train the model for one epoch and return the average loss.

    Args:
        model (nn.Module): The neural network model to train.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function to compute the training loss.

    Returns:
        float: Average loss per sample over the epoch.

    Raises:
        RuntimeError: If an error occurs during training (e.g., CUDA out of memory).
        ValueError: If the DataLoader is empty or contains invalid data.
    """
    try:
        if not train_loader:
            logger.error("Training DataLoader is empty")
            raise ValueError("Training DataLoader is empty")

        model.train()
        total_loss = 0.0
        num_samples = 0

        for images, masks in train_loader:
            if images.size(0) == 0:
                logger.warning("Empty batch encountered in training DataLoader")
                continue

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)  # Optimize memory usage
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss value in batch: {float(loss)}")
                continue

            loss.backward()
            # Clip gradients to prevent exploding gradients (max_norm=1.0 is a reasonable threshold)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        if num_samples == 0:
            logger.error("No valid samples processed in training epoch")
            raise ValueError("No valid samples processed in training epoch")

        avg_loss = total_loss / num_samples
        logger.info(f"Training epoch completed, average loss: {avg_loss:.4f}")
        return avg_loss

    except RuntimeError as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        raise RuntimeError(f"Training failed: {str(e)}")