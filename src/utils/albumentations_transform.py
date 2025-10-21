import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Union
import torch

class AlbumentationsTransform:
    """Applies Albumentations transformations to image-mask pairs for segmentation tasks.

    Supports data augmentation for training and minimal transformations (resizing and normalization)
    for validation/testing.

    Attributes:
        transform (A.Compose): Configured Albumentations transformation pipeline.
    """

    def __init__(self, size: Tuple[int, int] = (256, 256), train: bool = True) -> None:
        """Initialize the transformation pipeline.

        Args:
            size (Tuple[int, int]): Desired output size (height, width) for images and masks.
                Defaults to (256, 256).
            train (bool): If True, applies augmentation for training; if False, applies only
                resizing and normalization. Defaults to True.
        """
        self.transform = self._build_transform_pipeline(size, train)

    def _build_transform_pipeline(self, size: Tuple[int, int], train: bool) -> A.Compose:
        """Build the Albumentations transformation pipeline.

        Args:
            size (Tuple[int, int]): Desired output size (height, width).
            train (bool): If True, includes data augmentation for training.

        Returns:
            A.Compose: Configured Albumentations transformation pipeline.
        """
        transforms = [
            A.Resize(height=size[0], width=size[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ]

        if train:
            transforms.insert(1, A.HorizontalFlip(p=0.5))
            transforms.insert(2, A.VerticalFlip(p=0.5))
            transforms.insert(3, A.RandomRotate90(p=0.5))
            transforms.insert(4, A.Affine(
                rotate=(-30, 30),
                scale=(0.9, 1.1),
                translate_percent=(0, 0.0625),
                p=0.5
            ))
            transforms.insert(5, A.ElasticTransform(alpha=1, sigma=50, p=0.3))
            transforms.insert(6, A.GridDistortion(p=0.3))
            transforms.insert(7, A.RandomBrightnessContrast(p=0.5))
            transforms.insert(8, A.CLAHE(p=0.3))
            transforms.insert(9, A.GaussNoise(p=0.3))

        return A.Compose(transforms)

    def __call__(self, image: Union[Image.Image, np.ndarray], 
                 mask: Union[Image.Image, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformations to an image-mask pair.

        Args:
            image (Union[PIL.Image.Image, np.ndarray]): Input RGB image.
            mask (Union[PIL.Image.Image, np.ndarray]): Input single-channel mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image (shape: [C, H, W]) and mask (shape: [1, H, W]).

        Raises:
            ValueError: If the input image or mask cannot be converted to a numpy array.
        """
        try:
            # Convert inputs to numpy arrays
            image_np = np.array(image)
            mask_np = np.array(mask)
        except Exception as e:
            raise ValueError(f"Failed to convert inputs to numpy arrays: {str(e)}")

        # Ensure mask is single-channel
        if mask_np.ndim == 3 and mask_np.shape[-1] > 1:
            mask_np = mask_np[..., 0]

        # Binarize mask
        mask_np = (mask_np > 128).astype(np.uint8)

        # Apply transformations
        augmented = self.transform(image=image_np, mask=mask_np)
        image_tensor, mask_tensor = augmented["image"], augmented["mask"]

        # Convert mask to binary tensor and add channel dimension
        mask_tensor = (mask_tensor > 0.5).float()
        mask_tensor = mask_tensor.unsqueeze(0)
        # mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0)

        return image_tensor, mask_tensor