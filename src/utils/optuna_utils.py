import os
import logging
from typing import Any
import optuna
from optuna.pruners import HyperbandPruner
from src.utils.config_utils import update_config_with_best_params

logger = logging.getLogger(__name__)

def setup_study(config: Any) -> optuna.Study:
    """Initialize and configure an Optuna study for hyperparameter optimization.

    Args:
        config: Configuration object containing study parameters.

    Returns:
        optuna.Study: Configured Optuna study instance.
    """
    pruner = HyperbandPruner()
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=True
    )
    logger.info(f"Initialized Optuna study: {config.study_name}")
    return study

def save_best_model(config: Any, best_trial: optuna.trial.Trial) -> None:
    """Save the best model and update configuration with best parameters.

    Args:
        config: Configuration object containing output directory and other settings.
        best_trial: Optuna trial containing the best parameters and results.
    """
    temp_path = os.path.join(config.output_dir, 'best_model_trial_temp.pth')
    if not os.path.exists(temp_path):
        logger.warning("No temporary best model found. Use best parameters for retraining.")
        return

    final_path = os.path.join(config.output_dir, f'best_model_trial_{best_trial.number}.pth')
    os.replace(temp_path, final_path)
    logger.info(f"Saved best model to: {final_path}")

    best_params = {
        'dropout': best_trial.params.get('dropout'),
        'bce_weight': best_trial.params.get('bce_weight'),
        'img_size': best_trial.params.get('img_size'),
        'batch_size': best_trial.params.get('batch_size')
    }
    update_config_with_best_params('config/config.yaml', best_params, final_path)

def log_best_trial(study: optuna.Study) -> None:
    """Log the best trial's results and parameters.

    Args:
        study: Optuna study containing trial results.
    """
    best_trial = study.best_trial
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best validation loss: {best_trial.value:.4f}")
    logger.info("Best parameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")