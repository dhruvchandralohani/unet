import os
import logging
from typing import Tuple
from torch.utils.data import DataLoader
from src.utils.albumentations_transform import AlbumentationsTransform
from src.utils.segmentation_dataset import SegmentationDataset

logger = logging.getLogger(__name__)

def create_data_loader(
    images_dir: str,
    masks_dir: str,
    size: Tuple[int, int],
    batch_size: int,
    train: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Create a PyTorch DataLoader for segmentation tasks.

    Args:
        images_dir (str): Directory containing the images.
        masks_dir (str): Directory containing the corresponding masks.
        size (Tuple[int, int]): Desired output size (height, width) for images and masks.
        batch_size (int): Number of samples per batch.
        train (bool): If True, creates a training DataLoader with shuffling and augmentations.
            If False, creates a validation/test DataLoader without shuffling. Defaults to False.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        DataLoader: Configured PyTorch DataLoader for the dataset.

    Raises:
        ValueError: If the dataset or transform creation fails.
        FileNotFoundError: If the image or mask directory does not exist.
    """
    try:
        # Validate input directories
        if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
            raise FileNotFoundError(f"Image directory '{images_dir}' or mask directory '{masks_dir}' not found")

        # Initialize transformations
        transform = AlbumentationsTransform(size=size, train=train)

        # Create dataset
        dataset = SegmentationDataset(images_dir=images_dir, masks_dir=masks_dir, transform=transform)

        # Configure DataLoader
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train,  # Shuffle only for training
            num_workers=num_workers,
            pin_memory=True,  # Optimize for CUDA
            drop_last=train  # Drop incomplete batch for training
        )

    except FileNotFoundError as e:
        logger.error(f"Failed to create DataLoader: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating DataLoader for {images_dir}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to create DataLoader: {str(e)}")

def create_train_loader(
    train_images_dir: str,
    train_masks_dir: str,
    size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for training data with augmentations.

    Args:
        train_images_dir (str): Directory containing training images.
        train_masks_dir (str): Directory containing training masks.
        size (Tuple[int, int]): Desired output size (height, width).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        DataLoader: Configured training DataLoader.

    Raises:
        ValueError: If the dataset or transform creation fails.
        FileNotFoundError: If the image or mask directory does not exist.
    """
    return create_data_loader(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        size=size,
        batch_size=batch_size,
        train=True,
        num_workers=num_workers
    )

def create_val_loader(
    val_images_dir: str,
    val_masks_dir: str,
    size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for validation data without augmentations.

    Args:
        val_images_dir (str): Directory containing validation images.
        val_masks_dir (str): Directory containing validation masks.
        size (Tuple[int, int]): Desired output size (height, width).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        DataLoader: Configured validation DataLoader.

    Raises:
        ValueError: If the dataset or transform creation fails.
        FileNotFoundError: If the image or mask directory does not exist.
    """
    return create_data_loader(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        size=size,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers
    )

def create_test_loader(
    test_images_dir: str,
    test_masks_dir: str,
    size: Tuple[int, int],
    batch_size: int,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for test data without augmentations.

    Args:
        test_images_dir (str): Directory containing test images.
        test_masks_dir (str): Directory containing test masks.
        size (Tuple[int, int]): Desired output size (height, width).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        DataLoader: Configured test DataLoader.

    Raises:
        ValueError: If the dataset or transform creation fails.
        FileNotFoundError: If the image or mask directory does not exist.
    """
    return create_data_loader(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        size=size,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers
    )