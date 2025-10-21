import logging
from typing import Dict, Any
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    """Configuration for training a U-Net model.

    Attributes:
        output_dir (str): Directory to save model checkpoints and outputs.
        seed (int): Random seed for reproducibility.
        study_name (str): Name of the Optuna study for hyperparameter optimization.
        storage (str): Storage path for Optuna study persistence.
        n_trials (int): Number of Optuna trials to run.
        epochs (int): Number of training epochs.
        in_channels (int): Number of input channels for the model.
        out_channels (int): Number of output channels for the model.
        bilinear (bool): Whether to use bilinear upsampling in the U-Net.
        train_images_dir (str): Directory containing training images.
        train_masks_dir (str): Directory containing training masks.
        val_images_dir (str): Directory containing validation images.
        val_masks_dir (str): Directory containing validation masks.
        img_size (int): Size to which images are resized (height and width).
        batch_size (int): Number of samples per batch.
        patience (int): Number of epochs to wait before early stopping.
    """
    output_dir: str
    seed: int
    study_name: str
    storage: str
    n_trials: int
    epochs: int
    in_channels: int
    out_channels: int
    bilinear: bool
    train_images_dir: str
    train_masks_dir: str
    val_images_dir: str
    val_masks_dir: str
    img_size: int
    batch_size: int
    patience: int

@dataclass
class TestConfig:
    """Configuration for testing a U-Net model.

    Attributes:
        model_path (str): Path to the trained model checkpoint.
        test_images_dir (str): Directory containing test images.
        test_masks_dir (str): Directory containing test masks.
        dropout (float): Dropout probability for the model.
        bce_weight (float): Weight for BCE loss in the combined loss function.
        img_size (int): Size to which images are resized (height and width).
        batch_size (int): Number of samples per batch.
        project_name (str): Name of the project for experiment tracking.
        run_name (str): Name of the test run for experiment tracking.
    """
    model_path: str
    test_images_dir: str
    test_masks_dir: str
    dropout: float
    bce_weight: float
    img_size: int
    batch_size: int
    project_name: str
    run_name: str

def load_train_config(config_path: str) -> TrainConfig:
    """Load training configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        TrainConfig: Parsed training configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is invalid or missing required fields.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file) or {}
        training_config = config_dict.get('training', {})
        if not training_config:
            raise ValueError("Missing 'training' section in config file")
        return TrainConfig(**training_config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load training config from {config_path}: {str(e)}", exc_info=True)
        raise ValueError(f"Invalid training configuration: {str(e)}")

def load_test_config(config_path: str) -> TestConfig:
    """Load testing configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        TestConfig: Parsed testing configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is invalid or missing required fields.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file) or {}
        testing_config = config_dict.get('testing', {})
        if not testing_config:
            raise ValueError("Missing 'testing' section in config file")
        return TestConfig(**testing_config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load testing config from {config_path}: {str(e)}", exc_info=True)
        raise ValueError(f"Invalid testing configuration: {str(e)}")

def update_config_with_best_params(config_path: str, best_params: Dict[str, Any], model_path: str) -> None:
    """Update the YAML configuration file with the best hyperparameters and model path.

    Args:
        config_path (str): Path to the YAML configuration file.
        best_params (Dict[str, Any]): Dictionary of best hyperparameters.
        model_path (str): Path to the best model checkpoint.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file cannot be updated due to invalid data.
    """
    try:
        # Load existing config or initialize empty dict
        with open(config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file) or {'training': {}, 'testing': {}}

        # Default values for testing section
        default_testing = {
            'dropout': 0.0,
            'bce_weight': 0.5,
            'img_size': 128,
            'batch_size': 4,
            'model_path': ''
        }

        # Update testing section with best parameters
        testing_config = config_dict.get('testing', {})
        testing_config.update({
            'dropout': best_params.get('dropout', testing_config.get('dropout', default_testing['dropout'])),
            'bce_weight': best_params.get('bce_weight', testing_config.get('bce_weight', default_testing['bce_weight'])),
            'img_size': best_params.get('img_size', testing_config.get('img_size', default_testing['img_size'])),
            'batch_size': best_params.get('batch_size', testing_config.get('batch_size', default_testing['batch_size'])),
            'model_path': model_path
        })
        config_dict['testing'] = testing_config

        # Write updated config back to file
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config_dict, file, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated {config_path} with best hyperparameters and model path: {model_path}")

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration file {config_path}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to update configuration: {str(e)}")