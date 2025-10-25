# UNet Crack Segmentation Project
## Overview
This project implements a UNet-based deep learning pipeline for semantic segmentation, specifically designed for crack detection using the crack500 dataset. It includes training with hyperparameter optimization (using Optuna), testing, and evaluation with metrics such as Intersection over Union (IoU) and Dice coefficient. The pipeline supports data augmentation (via Albumentations), experiment tracking (via Weights & Biases), and robust logging. The architecture is modular, with separate scripts for model definition, data handling, training, testing, and utilities.
## Project Structure
The project is organized as follows:
```
unet_segmentation/
│
├── config/
│   └── config.yaml              # Configuration file for training and testing
├── outputs/
│   ├── study.db                 # SQLite database storing Optuna study results for hyperparameter optimization
│   └── best_model_trail_X.pth   # Saved PyTorch model checkpoint for the best trial from Optuna optimization
├── scripts/
│   ├── test_unet.py             # Main script for testing the trained model
│   └── train_unet.py            # Main script for training with hyperparameter optimization
├── src/
│   ├── models/
│   │   └── unet.py              # UNet model architecture
│   ├── testing/
│   │   └── tester.py            # Testing logic with evaluation and visualization
│   ├── training/
│   │   └── trainer.py           # Training logic with Optuna integration
│   ├── utils/
│   │   ├── albumentations_transform.py  # Data augmentation and preprocessing
│   │   ├── config_utils.py      # Configuration loading and updating
│   │   ├── data.py              # DataLoader creation for training, validation, and testing
│   │   ├── losses.py            # Custom BCEDiceLoss combining BCE and Dice losses
│   │   ├── metrics.py           # IoU and Dice score computation
│   │   ├── optuna_utils.py      # Optuna study setup and best model saving
│   │   ├── segmentation_dataset.py  # Custom PyTorch Dataset for image-mask pairs
│   │   ├── test_utils.py        # Utilities for testing (model, DataLoader, visualizations)
│   │   ├── training.py          # Training utilities (seed setting, weight initialization)
│   │   └── validation.py        # Validation function for computing loss
├── wandb/                       # Directory for Weights & Biases run logs and artifacts
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

### Prerequisites
***Python***: 3.8 or higher
***Hardware***: GPU recommended for faster training (CUDA-compatible for PyTorch)
***Dataset***: The crack500 dataset, with images and masks organized in separate directories for training, validation, and testing (e.g., D:\Deep Learning\Datasets\crack500).
***Dependencies***: Listed in requirements.txt (see Installation).



### Installation
***Clone the Repository***:
```
git clone https://github.com/dhruvchandralohani/unet.git
```
***Set Up a Virtual Environment (recommended)***:
```
python -m venv venv
source venv/bin/activate
venv\Scripts\activate # On Windows: 
```
***Install Dependencies***:
```
pip install -r requirements.txt
```
***Prepare the Dataset***:
Ensure the crack500 dataset is available with the following structure:crack500/
```
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
```
Update config/config.yaml with the correct paths to these directories.

Configure Weights & Biases (optional, for experiment tracking):

Install wandb and log in
```
pip install wandb
wandb login
```
Ensure your Weights & Biases API key is set up.

Configuration

The config/config.yaml file contains parameters for training and testing:

Training Section:

output_dir: Directory for model checkpoints (e.g., D:\Virtual Environments\PyTorch\Projects\unet\outputs).

seed: Random seed (e.g., 42).

study_name: Optuna study name (e.g., unet_optimization).

storage: Optuna storage (e.g., sqlite:///outputs/study.db).

n_trials: Number of Optuna trials (e.g., 2).

epochs, in_channels, out_channels, bilinear, patience: Training hyperparameters.

train_images_dir, train_masks_dir, val_images_dir, val_masks_dir: Dataset paths.

Testing Section:

model_path: Path to the trained model (e.g., D:\Virtual Environments\PyTorch\Projects\unet\outputs\best_model_trial_1.pth).

test_images_dir, test_masks_dir: Test dataset paths.
dropout, bce_weight, img_size, batch_size, project_name, run_name: Testing parameters.

Example config.yaml (with fixed values):
```
training:
  batch_size: 8
  bilinear: true
  epochs: 10
  img_size: 192
  in_channels: 3
  n_trials: 2
  out_channels: 1
  output_dir: D:\Virtual Environments\PyTorch\Projects\unet\outputs
  seed: 42
  patience: 7
  storage: sqlite:///outputs/study.db
  study_name: unet_optimization
  train_images_dir: D:\Deep Learning\Datasets\crack500\train\images
  train_masks_dir: D:\Deep Learning\Datasets\crack500\train\masks
  val_images_dir: D:\Deep Learning\Datasets\crack500\val\images
  val_masks_dir: D:\Deep Learning\Datasets\crack500\val\masks
testing:
  batch_size: 8
  bce_weight: 0.5236582380671724
  dropout: 0.47303267888914247
  img_size: 192
  model_path: D:\Virtual Environments\PyTorch\Projects\unet\outputs\best_model_trial_1.pth
  project_name: unet_optimization
  run_name: test_best_model
  test_images_dir: D:\Deep Learning\Datasets\crack500\test\images
  test_masks_dir: D:\Deep Learning\Datasets\crack500\test\masks
```

Usage

Training

To train the UNet model with hyperparameter optimization:

python train_unet.py

This runs n_trials (e.g., 2) Optuna trials, optimizing hyperparameters like learning rate, batch size, and dropout.

Results are logged to training.log and Weights & Biases.

The best model is saved to output_dir (e.g., best_model_trial_X.pth), and config.yaml is updated with the best hyperparameters.

Testing

To test the trained model:

python test_unet.py

This evaluates the model specified in model_path on the test dataset, computing test loss, average IoU, and Dice scores.

Results and visualizations are logged to testing.log and Weights & Biases.

Key Features

UNet Architecture: Defined in unet.py, with encoder-decoder structure, skip connections, and optional bilinear upsampling.

Data Augmentation: Handled by albumentations_transform.py, with transformations like flips, rotations, and noise for training.

Hyperparameter Optimization: Uses Optuna (optuna_utils.py) to optimize learning rate, batch size, dropout, etc.

Loss Function: Combines BCE and Dice losses (losses.py) with configurable weighting.

Metrics: Computes IoU and Dice scores (metrics.py) for evaluation.

Data Loading: Custom SegmentationDataset (segmentation_dataset.py) and DataLoader utilities (data.py) for efficient data handling.

Experiment Tracking: Logs metrics and visualizations to Weights & Biases (test_utils.py, trainer.py, tester.py).

Reproducibility: Ensures deterministic behavior with seed setting and weight initialization (training.py).

Potential Issues and Fixes

Dataset Setup:
Ensure the crack500 dataset directories contain matching image-mask pairs (same filenames) and that masks are single-channel (grayscale) with binary values (0 or 1).

Memory Usage:

Large img_size (e.g., 192) or batch_size (e.g., 8) may cause CUDA out-of-memory errors. Monitor GPU usage or reduce these values if needed.

Weights & Biases:

Ensure wandb is configured with an API key for logging. If not needed, modify trainer.py, tester.py, and test_utils.py to skip wandb initialization.

Contributing

Contributions are welcome! To contribute:

Fork the repository.

Create a feature branch (git checkout -b feature/your-feature).

Commit changes (git commit -m "Add your feature").

Push to the branch (git push origin feature/your-feature).

Open a pull request.