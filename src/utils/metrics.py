import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute the Dice coefficient (F1 score) for a batch of predicted and ground truth masks.

    The Dice coefficient measures the similarity between predicted and target masks, averaged over the batch.

    Args:
        pred (torch.Tensor): Predicted logits or probabilities, shape [batch_size, channels, height, width].
            Logits are converted to probabilities via sigmoid.
        target (torch.Tensor): Ground truth binary masks, same shape as pred.
            Values should be binary (0 or 1).
        smooth (float): Smoothing factor to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Mean Dice coefficient across the batch, in [0, 1].

    Raises:
        ValueError: If pred and target have mismatched shapes or invalid values.
    """
    if pred.shape != target.shape:
        logger.error(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        raise ValueError("Pred and target must have the same shape")

    # Convert logits to probabilities
    # pred = torch.sigmoid(pred)

    # Clamp predictions for numerical stability
    pred = torch.clamp(pred, min=smooth, max=1 - smooth)

    # Flatten tensors for element-wise operations
    pred_flat = pred.view(pred.size(0), -1)  # Shape: [batch_size, N]
    target_flat = target.view(target.size(0), -1)  # Shape: [batch_size, N]

    # Compute intersection and denominator
    intersection = (pred_flat * target_flat).sum(dim=1)  # Shape: [batch_size]
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # Shape: [batch_size]

    # Log warning for degenerate cases
    if (denom < smooth).any():
        logger.warning(f"Degenerate Dice: denominator too small, values={denom.detach().cpu().tolist()}")

    # Compute Dice coefficient with smoothing
    dice = (2.0 * intersection + smooth) / (denom + smooth)  # Shape: [batch_size]

    # Check for NaN values
    if torch.isnan(dice).any():
        logger.warning(
            f"NaN detected in Dice computation: "
            f"intersection={intersection.detach().cpu().tolist()}, "
            f"denom={denom.detach().cpu().tolist()}"
        )

    return dice.mean()

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute the Intersection over Union (IoU) score for a batch of predicted and ground truth masks.

    IoU measures the overlap between predicted and ground truth binary masks, averaged over the batch.

    Args:
        pred (torch.Tensor): Predicted masks, typically after thresholding or sigmoid, shape [batch_size, channels, height, width].
            Values should be in [0, 1].
        target (torch.Tensor): Ground truth binary masks, same shape as pred.
            Values should be binary (0 or 1).
        smooth (float): Smoothing factor to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Mean IoU score across the batch, in [0, 1].

    Raises:
        ValueError: If pred and target have mismatched shapes or invalid values.
    """
    if pred.shape != target.shape:
        logger.error(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        raise ValueError("Pred and target must have the same shape")

    # Flatten tensors for element-wise operations
    pred_flat = pred.view(pred.size(0), -1)  # Shape: [batch_size, N]
    target_flat = target.view(target.size(0), -1)  # Shape: [batch_size, N]

    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)  # Shape: [batch_size]
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection  # Shape: [batch_size]

    # Handle degenerate cases
    degenerate = union < smooth
    if degenerate.any():
        logger.warning(f"Degenerate IoU: union too small, values={union[degenerate].detach().cpu().tolist()}")

    # Compute IoU with smoothing
    iou = (intersection + smooth) / (union + smooth)  # Shape: [batch_size]

    # Check for NaN values
    if torch.isnan(iou).any():
        logger.warning(
            f"NaN detected in IoU computation: "
            f"intersection={intersection.detach().cpu().tolist()}, "
            f"union={union.detach().cpu().tolist()}"
        )

    return iou.mean()