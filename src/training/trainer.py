import logging
import gc
import os
from typing import Tuple, Any
import torch
import torch.optim as optim
import optuna
import wandb
from torch.utils.data import DataLoader
from src.models.unet import UNet
from src.utils.losses import BCEDiceLoss
from src.utils.data import create_train_loader, create_val_loader
from src.utils.training import train_one_epoch, initialize_weights
from src.utils.validation import validate

logger = logging.getLogger(__name__)

class Trainer:
    """Manages training and hyperparameter optimization for a U-Net model."""

    def __init__(self, config: Any) -> None:
        """Initialize the trainer with configuration settings.

        Args:
            config: Configuration object containing training parameters.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Trainer initialized with device: {self.device}")

    def _setup_data_loaders(self, img_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create and configure training and validation data loaders.

        Args:
            img_size (int): Size to which images are resized (height and width).
            batch_size (int): Number of samples per batch.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders.

        Raises:
            RuntimeError: If data loader creation fails.
        """
        try:
            train_loader = create_train_loader(
                train_images_dir=self.config.train_images_dir,
                train_masks_dir=self.config.train_masks_dir,
                size=(img_size, img_size),
                batch_size=batch_size
            )
            val_loader = create_val_loader(
                val_images_dir=self.config.val_images_dir,
                val_masks_dir=self.config.val_masks_dir,
                size=(img_size, img_size),
                batch_size=batch_size
            )
            logger.info("Created training and validation data loaders")
            return train_loader, val_loader
        except Exception as e:
            logger.error(f"Failed to create data loaders: {str(e)}", exc_info=True)
            raise RuntimeError(f"Data loader creation failed: {str(e)}")

    def _setup_optimizer(self, model: torch.nn.Module, optimizer_name: str, 
                        lr: float, weight_decay: float) -> optim.Optimizer:
        """Configure the optimizer based on hyperparameter choices.

        Args:
            model (torch.nn.Module): The model whose parameters will be optimized.
            optimizer_name (str): Name of the optimizer ('Adam', 'AdamW', 'RMSprop', or 'SGD').
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 regularization) coefficient.

        Returns:
            optim.Optimizer: Configured optimizer instance.

        Raises:
            ValueError: If an unsupported optimizer name is provided.
        """
        optimizer_map = {
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
            'RMSprop': optim.RMSprop,
            'SGD': lambda params, lr, weight_decay: optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        }
        
        if optimizer_name not in optimizer_map:
            logger.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Optimizer {optimizer_name} is not supported")
        
        try:
            optimizer = optimizer_map[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"Initialized {optimizer_name} optimizer with lr={lr:.2e}, weight_decay={weight_decay:.2e}")
            return optimizer
        except Exception as e:
            logger.error(f"Failed to initialize optimizer {optimizer_name}: {str(e)}", exc_info=True)
            raise

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Suggest hyperparameters for an Optuna trial.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.

        Returns:
            dict: Dictionary containing suggested hyperparameters.
        """
        return {
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
            'bce_weight': trial.suggest_float('bce_weight', 0.2, 0.8),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'optimizer_name': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop']),
            'img_size': trial.suggest_categorical('img_size', [128, 192, 224])
        }

    def _initialize_wandb(self, trial: optuna.Trial, params: dict) -> wandb.Run:
        """Initialize Weights & Biases for experiment tracking.

        Args:
            trial (optuna.Trial): Optuna trial object.
            params (dict): Hyperparameters to log.

        Returns:
            wandb.Run: Initialized Weights & Biases run object.
        """
        return wandb.init(
            project=self.config.study_name,
            config={
                **params,
                'trial_number': trial.number,
                'epochs': self.config.epochs,
                'seed': self.config.seed
            },
            reinit=True,
            resume="allow"
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.

        Returns:
            float: Best validation loss achieved during training.

        Raises:
            optuna.TrialPruned: If the trial is pruned due to poor performance or CUDA OOM.
            RuntimeError: If training fails for reasons other than pruning.
        """
        model = None
        optimizer = None
        run = None
        
        try:
            # Clear CUDA memory
            torch.cuda.empty_cache()

            # Suggest hyperparameters
            params = self._suggest_hyperparameters(trial)
            lr = params['lr']
            weight_decay = params['weight_decay']
            batch_size = params['batch_size']
            bce_weight = params['bce_weight']
            dropout = params['dropout']
            optimizer_name = params['optimizer_name']
            img_size = params['img_size']

            # Initialize Weights & Biases
            run = self._initialize_wandb(trial, params)

            # Setup data loaders
            train_loader, val_loader = self._setup_data_loaders(img_size, batch_size)

            # Initialize model
            model = UNet(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                bilinear=self.config.bilinear,
                dropout_prob=dropout
            ).to(self.device)
            model.apply(initialize_weights)
            logger.info(f"Initialized UNet with dropout={dropout:.2f}")

            # Initialize loss, optimizer, and scheduler
            criterion = BCEDiceLoss(bce_weight=bce_weight)
            optimizer = self._setup_optimizer(model, optimizer_name, lr, weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            temp_model_path = os.path.join(self.config.output_dir, 'best_model_trial_temp.pth')

            for epoch in range(self.config.epochs):
                # Train one epoch
                train_loss = train_one_epoch(model, self.device, train_loader, optimizer, criterion)
                val_loss = validate(model, self.device, val_loader, criterion)
                scheduler.step(val_loss)
                trial.report(val_loss, epoch)

                # Log metrics
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1
                }, step=epoch + 1)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), temp_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.info(
                    f"Trial {trial.number} Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

                # Check for pruning
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()

                # Early stopping
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Save final model
            final_model_path = os.path.join(self.config.output_dir, f"trial_{trial.number}_final.pth")
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Saved final model to: {final_model_path}")

            return best_val_loss

        except optuna.TrialPruned:
            raise
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.warning(f"Trial {trial.number} failed due to CUDA OOM")
                raise optuna.TrialPruned()
            logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up resources
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if run is not None:
                run.finish()
            torch.cuda.empty_cache()
            gc.collect()