import logging
import torch
import torch.nn as nn
from typing import Optional
from src.utils.metrics import dice_score

logger = logging.getLogger(__name__)

class BCEDiceLoss(nn.Module):
    """Combined loss function blending Binary Cross-Entropy (BCE) and Dice loss for segmentation tasks.

    The loss is a weighted sum of BCE and (1 - Dice coefficient), balancing pixel-wise and region-based errors.

    Attributes:
        bce_weight (float): Weight for the BCE component in the combined loss.
        bce (nn.BCEWithLogitsLoss): BCE loss with logits for numerical stability.
    """

    def __init__(self, bce_weight: float = 0.5) -> None:
        """Initialize the BCEDiceLoss module.

        Args:
            bce_weight (float): Weight for the BCE component in the combined loss.
                Must be in [0, 1]. Default: 0.5 (equal weight for BCE and Dice losses).

        Raises:
            ValueError: If bce_weight is not in the range [0, 1].
        """
        super().__init__()
        if not 0 <= bce_weight <= 1:
            logger.error(f"bce_weight must be in [0, 1], got {bce_weight}")
            raise ValueError("bce_weight must be in the range [0, 1]")
        
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined BCE and Dice loss.

        Args:
            inputs (torch.Tensor): Predicted logits from the model, shape [batch_size, channels, height, width].
                Typically, channels=1 for binary segmentation.
            targets (torch.Tensor): Ground truth binary masks, same shape as inputs.
                Values should be binary (0 or 1).

        Returns:
            torch.Tensor: Scalar tensor representing the combined loss (weighted BCE + Dice loss).

        Raises:
            ValueError: If inputs and targets have mismatched shapes or invalid values.
        """
        if inputs.shape != targets.shape:
            logger.error(f"Shape mismatch: inputs {inputs.shape}, targets {targets.shape}")
            raise ValueError("Inputs and targets must have the same shape")

        # Clamp inputs to prevent numerical instability
        inputs = torch.clamp(inputs, min=-100, max=100)

        # Compute BCE loss
        bce_loss = self.bce(inputs, targets)

        # Compute Dice loss
        dice = dice_score(inputs, targets)
        dice_loss = 1 - dice

        # Combine losses
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

        # Log warning if NaN is detected
        if torch.isnan(loss):
            logger.warning(
                f"NaN detected in loss computation: BCE={float(bce_loss.detach().cpu())}, "
                f"Dice={float(dice.detach().cpu())}, "
                f"Target unique values={torch.unique(targets).detach().cpu().tolist()}"
            )

        return loss