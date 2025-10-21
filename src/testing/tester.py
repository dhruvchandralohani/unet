import logging
import torch
import numpy as np
import wandb
from typing import Tuple, Any

from src.utils.losses import BCEDiceLoss
from src.utils.validation import validate
from src.utils.metrics import iou_score, dice_score
from src.utils.test_utils import setup_model, setup_data_loader, setup_wandb, create_visualization_table

logger = logging.getLogger(__name__)

class UNetTester:
    """Class to handle testing of a U-Net model with logging to Weights & Biases.

    Attributes:
        config: Configuration object with testing parameters.
        device: Device for computations (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, config: Any) -> None:
        """Initialize the UNetTester with a configuration object.

        Args:
            config: Configuration object containing testing parameters.

        Raises:
            ValueError: If config is invalid or missing required attributes.
        """
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized UNetTester with device: {self.device}")

    def test_model(self) -> Tuple[float, float, float]:
        """Run model testing, compute metrics, and log to Weights & Biases.

        Computes test loss, average IoU, and average Dice scores, logging them along with visualizations of the first image from each batch.

        Returns:
            Tuple[float, float, float]: Test loss, average IoU score, and average Dice score.

        Raises:
            RuntimeError: If an error occurs during testing (e.g., CUDA out of memory).
            ValueError: If DataLoader or model setup fails.
        """
        try:
            setup_wandb(self.config)
            model = setup_model(self.config, self.device)
            test_loader = setup_data_loader(self.config)
            criterion = BCEDiceLoss(bce_weight=self.config.bce_weight)

            # Compute test loss
            test_loss = validate(model, self.device, test_loader, criterion)
            logger.info(f"Test Loss: {test_loss:.4f}")

            # Initialize lists to store metrics and visualization data
            iou_scores = []
            dice_scores = []
            image_list = []
            mask_list = []
            pred_list = []
            vis_iou_scores = []
            vis_dice_scores = []

            # Evaluate model without gradient computation
            with torch.no_grad():
                for images, masks in test_loader:
                    if images.size(0) == 0:
                        logger.warning("Empty batch encountered in test DataLoader")
                        continue

                    logger.debug(f"DataLoader batch - Images device: {images.device}, Masks device: {masks.device}")
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)

                    # Compute batch metrics
                    batch_iou = iou_score(preds, masks)
                    batch_dice = dice_score(preds, masks)

                    if not (torch.isnan(batch_iou) or torch.isinf(batch_iou)):
                        iou_scores.append(batch_iou.item())
                    if not (torch.isnan(batch_dice) or torch.isinf(batch_dice)):
                        dice_scores.append(batch_dice.item())

                    # Compute metrics for the first sample for visualization
                    if images.size(0) > 0:
                        first_iou = iou_score(preds[0:1], masks[0:1]).item()
                        first_dice = dice_score(preds[0:1], masks[0:1]).item()

                        if not (np.isnan(first_iou) or np.isinf(first_iou) or np.isnan(first_dice) or np.isinf(first_dice)):
                            vis_iou_scores.append(first_iou)
                            vis_dice_scores.append(first_dice)

                        # Collect first image, mask, and prediction from each batch for visualization
                        image_list.append(images[0].cpu())
                        mask_list.append(masks[0].cpu())
                        pred_list.append(preds[0].cpu())

            # Check if valid metrics were computed
            if not iou_scores or not dice_scores:
                logger.error("No valid IoU or Dice scores computed")
                raise ValueError("No valid metrics computed during testing")

            # Compute average metrics
            avg_iou = float(np.mean(iou_scores))
            avg_dice = float(np.mean(dice_scores))
            logger.info(f"Average IoU Score: {avg_iou:.4f}, Average Dice Score: {avg_dice:.4f}")

            # Create and log visualization table
            if image_list and mask_list and pred_list and vis_iou_scores and vis_dice_scores:
                table = create_visualization_table(
                    torch.stack(image_list),
                    torch.stack(mask_list),
                    torch.stack(pred_list),
                    vis_iou_scores,
                    vis_dice_scores
                )
            else:
                logger.warning("No images or metrics available for visualization")
                table = wandb.Table(columns=["Original Image", "Ground Truth Mask", "Predicted Mask",
                                            "IoU Score", "Dice Score"])

            # Log metrics and visualizations
            wandb.log({
                "test_loss": test_loss,
                "avg_iou_score": avg_iou,
                "avg_dice_score": avg_dice,
                "validation_examples": table
            })

            return test_loss, avg_iou, avg_dice

        except RuntimeError as e:
            logger.error(f"Testing error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during testing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Testing failed: {str(e)}")
        finally:
            wandb.finish()