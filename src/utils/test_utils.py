import logging
import numpy as np
import torch
import os
import sys
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from typing import List, Tuple, Any

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import UNet
from src.utils.data import create_test_loader

logger = logging.getLogger(__name__)

def setup_model(config: Any, device: torch.device) -> nn.Module:
    """Initialize and load the U-Net model.

    Args:
        config: Configuration object containing testing parameters.
        device: Device for computations (e.g., 'cuda' or 'cpu').

    Returns:
        nn.Module: Loaded U-Net model in evaluation mode.

    Raises:
        FileNotFoundError: If the model checkpoint file is missing.
        RuntimeError: If model loading fails.
    """
    try:
        model = UNet(
            in_channels=3,
            out_channels=1,
            bilinear=True,
            dropout_prob=config.dropout
        )
        model.load_state_dict(torch.load(config.model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Loaded U-Net model from {config.model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found: {config.model_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {str(e)}")

def setup_data_loader(config: Any) -> DataLoader:
    """Create a DataLoader for the test dataset.

    Args:
        config: Configuration object containing testing parameters.

    Returns:
        DataLoader: Configured test DataLoader.

    Raises:
        ValueError: If DataLoader creation fails.
    """
    try:
        data_loader = create_test_loader(
            test_images_dir=config.test_images_dir,
            test_masks_dir=config.test_masks_dir,
            size=(config.img_size, config.img_size),
            batch_size=config.batch_size
        )
        logger.info("Created test DataLoader")
        return data_loader
    except Exception as e:
        logger.error(f"Failed to create test DataLoader: {str(e)}", exc_info=True)
        raise ValueError(f"Test DataLoader creation failed: {str(e)}")

def setup_wandb(config: Any) -> None:
    """Initialize Weights & Biases for experiment tracking.

    Args:
        config: Configuration object containing project and run details.

    Raises:
        RuntimeError: If Weights & Biases initialization fails.
    """
    try:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=vars(config),
            reinit=True
        )
        logger.info(f"Initialized Weights & Biases for project: {config.project_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {str(e)}", exc_info=True)
        raise RuntimeError(f"Weights & Biases initialization failed: {str(e)}")

def denormalize_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Denormalize images for visualization.

    Args:
        images: Normalized images, shape [batch_size, 3, H, W].
        device: Device for computations (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Denormalized images, shape [batch_size, 3, H, W], values in [0, 255].

    Raises:
        RuntimeError: If tensors are on different devices.
    """
    images = images.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    scale = torch.tensor(255.0, device=device)
    
    logger.debug(f"Images device: {images.device}, Mean device: {mean.device}, Std device: {std.device}, Scale device: {scale.device}")
    
    assert images.device == mean.device == std.device == scale.device, (
        f"Device mismatch: Images on {images.device}, Mean on {mean.device}, "
        f"Std on {std.device}, Scale on {scale.device}"
    )
    
    denorm_imgs = (images * std + mean).clamp(0, 1) * scale
    return denorm_imgs

def prepare_visualizations(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare images, true masks, and predicted masks for visualization.

    Args:
        images: Normalized input images, shape [batch_size, 3, H, W].
        masks: Ground truth masks, shape [batch_size, 1, H, W].
        predictions: Predicted masks after sigmoid, shape [batch_size, 1, H, W].
        device: Device for computations (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Denormalized images, predicted masks, and true masks
            as numpy arrays for visualization.
    """
    logger.debug(f"Visualizations - Images shape: {images.shape}, device: {images.device}")
    logger.debug(f"Visualizations - Masks shape: {masks.shape}, device: {masks.device}")
    logger.debug(f"Visualizations - Predictions shape: {predictions.shape}, device: {predictions.device}")

    denorm_imgs = denormalize_images(images, device).byte().cpu().numpy().transpose(0, 2, 3, 1)
    pred_masks = (predictions > 0.5).float().cpu().numpy() * 255
    true_masks = masks.cpu().numpy() * 255
    return denorm_imgs, pred_masks.astype(np.uint8), true_masks.astype(np.uint8)

def create_visualization_table(
    images: torch.Tensor,
    true_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    iou_scores: List[float],
    dice_scores: List[float]
) -> wandb.Table:
    """Create a Weights & Biases table for visualization of images, masks, and metrics.

    Args:
        images: Input images, shape [num_samples, 3, H, W].
        true_masks: Ground truth masks, shape [num_samples, 1, H, W].
        pred_masks: Predicted masks after sigmoid, shape [num_samples, 1, H, W].
        iou_scores: Per-sample IoU scores for visualized samples.
        dice_scores: Per-sample Dice scores for visualized samples.

    Returns:
        wandb.Table: Table containing images, masks, and per-sample IoU and Dice scores.
    """
    table = wandb.Table(columns=["Original Image", "Ground Truth Mask", "Predicted Mask",
                                "IoU Score", "Dice Score"])

    denorm_imgs, pred_masks_vis, true_masks_vis = prepare_visualizations(images, true_masks, pred_masks, images.device)

    for i in range(len(images)):
        table.add_data(
            wandb.Image(denorm_imgs[i]),
            wandb.Image(true_masks_vis[i, 0]),
            wandb.Image(pred_masks_vis[i, 0]),
            f"{iou_scores[i]:.4f}",
            f"{dice_scores[i]:.4f}"
        )

    return table