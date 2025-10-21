import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

logger = logging.getLogger(__name__)

@torch.no_grad()
def validate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module
) -> float:
    """Evaluate the model on a validation dataset and compute the average loss.

    Runs in evaluation mode without gradient computation for efficiency, using full precision (FP32).

    Args:
        model (nn.Module): Neural network model (e.g., U-Net) to evaluate.
        device (torch.device): Device for computations (e.g., 'cuda' or 'cpu').
        loader (DataLoader): DataLoader providing batches of validation images and masks.
        criterion (nn.Module): Loss function to compute error between predictions and ground truth.

    Returns:
        float: Average validation loss per sample over the dataset.

    Raises:
        ValueError: If the DataLoader is empty or contains invalid data.
        RuntimeError: If an error occurs during model evaluation (e.g., CUDA out of memory).
    """
    try:
        if not loader:
            logger.error("Validation DataLoader is empty")
            raise ValueError("Validation DataLoader is empty")

        model.eval()
        total_loss = 0.0
        num_samples = 0

        for images, masks in loader:
            if images.size(0) == 0:
                logger.warning("Empty batch encountered in validation DataLoader")
                continue

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss value in batch: {float(loss)}")
                continue

            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        if num_samples == 0:
            logger.error("No valid samples processed in validation")
            raise ValueError("No valid samples processed in validation")

        avg_loss = total_loss / num_samples
        logger.info(f"Validation completed, average loss: {avg_loss:.4f}")
        return avg_loss

    except RuntimeError as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}", exc_info=True)
        raise RuntimeError(f"Validation failed: {str(e)}")