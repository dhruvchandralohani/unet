import os
import glob
import logging
from typing import List, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SegmentationDataset(Dataset):
    """PyTorch Dataset for loading image-mask pairs for segmentation tasks.

    Assumes images and masks have matching filenames in separate directories. Images are loaded as RGB,
    and masks as single-channel (grayscale). Supports optional transformations (e.g., Albumentations).

    Attributes:
        images_dir (str): Directory containing image files.
        masks_dir (str): Directory containing mask files.
        transform (Optional[Callable]): Transformation function for image-mask pairs.
        filenames (List[str]): List of valid filenames common to both directories.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ) -> None:
        """Initialize the dataset with image and mask directories and optional transformations.

        Args:
            images_dir (str): Path to the directory containing image files (e.g., .jpg, .png).
            masks_dir (str): Path to the directory containing mask files with matching filenames.
            file_list (Optional[List[str]]): Specific list of filenames to include. If None, uses all common files.
                Defaults to None.
            transform (Optional[Callable]): Transformation function (e.g., Albumentations) that takes
                an image and mask and returns transformed tensors. Defaults to None.

        Raises:
            FileNotFoundError: If images_dir or masks_dir does not exist.
            ValueError: If no valid image-mask pairs are found.
        """
        # Validate directories
        if not os.path.isdir(images_dir):
            logger.error(f"Image directory not found: {images_dir}")
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        if not os.path.isdir(masks_dir):
            logger.error(f"Mask directory not found: {masks_dir}")
            raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.filenames = self._get_common_filenames(file_list)

        if not self.filenames:
            logger.error(f"No common files found between {images_dir} and {masks_dir}")
            raise ValueError("No valid image-mask pairs found")

        logger.info(f"Initialized dataset with {len(self.filenames)} image-mask pairs")

    def _get_common_filenames(self, file_list: Optional[List[str]]) -> List[str]:
        """Retrieve common filenames between image and mask directories.

        Args:
            file_list (Optional[List[str]]): Optional list of filenames to filter.

        Returns:
            List[str]: Sorted list of filenames present in both directories.

        Raises:
            ValueError: If no common files are found or file_list filters out all files.
        """
        # Get all filenames from image and mask directories
        image_files = {os.path.basename(p) for p in glob.glob(os.path.join(self.images_dir, '*'))}
        mask_files = {os.path.basename(p) for p in glob.glob(os.path.join(self.masks_dir, '*'))}

        # Find common filenames
        common_files = sorted(image_files.intersection(mask_files))

        # Filter by provided file_list if specified
        if file_list is not None:
            common_files = [f for f in common_files if f in file_list]
            if not common_files:
                logger.error("Provided file_list filtered out all common files")
                raise ValueError("No valid files found after applying file_list filter")

        return common_files

    def __len__(self) -> int:
        """Get the total number of image-mask pairs in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and transform an image-mask pair at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image (shape: [C, H, W]) and mask (shape: [1, H, W]).

        Raises:
            IndexError: If the index is out of range.
            FileNotFoundError: If the image or mask file is missing.
            ValueError: If the transform is None or if image/mask contains invalid values.
        """
        if not 0 <= idx < len(self.filenames):
            logger.error(f"Index {idx} out of range for dataset with {len(self.filenames)} samples")
            raise IndexError(f"Index {idx} out of range")

        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except FileNotFoundError as e:
            logger.error(f"File not found: {img_path} or {mask_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load image or mask {filename}: {str(e)}", exc_info=True)
            raise FileNotFoundError(f"Error loading file {filename}: {str(e)}")

        # Convert to numpy arrays for validation
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Validate for NaN or Inf values
        if np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
            logger.warning(f"Invalid values (NaN or Inf) detected in image: {filename}")
            raise ValueError(f"Invalid image values in {filename}")
        if np.any(np.isnan(mask_np)) or np.any(np.isinf(mask_np)):
            logger.warning(f"Invalid values (NaN or Inf) detected in mask: {filename}")
            raise ValueError(f"Invalid mask values in {filename}")

        # Apply transformations
        if self.transform is None:
            logger.error("Transform is None, but a transform is required for data processing")
            raise ValueError("Transform must be provided for SegmentationDataset")

        try:
            image, mask = self.transform(image, mask)
        except Exception as e:
            logger.error(f"Transform failed for {filename}: {str(e)}", exc_info=True)
            raise ValueError(f"Transform failed for {filename}: {str(e)}")

        return image, mask