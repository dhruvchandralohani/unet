import os
import logging
from src.utils.training import set_seed
from src.training.trainer import Trainer
from src.utils.config_utils import load_train_config
from src.utils.optuna_utils import setup_study, save_best_model, log_best_trial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Run the U-Net training pipeline with hyperparameter optimization."""
    try:
        # Load configuration
        config = load_train_config('config/config.yaml')
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {config.output_dir}")

        # Set random seed for reproducibility
        set_seed(config.seed)
        logger.debug(f"Random seed set to: {config.seed}")

        # Initialize trainer and study
        trainer = Trainer(config)
        study = setup_study(config)

        # Run optimization
        study.optimize(trainer.objective, n_trials=config.n_trials, gc_after_trial=True)
        logger.info("Hyperparameter optimization completed")

        # Save best model and update config
        save_best_model(config, study.best_trial)
        log_best_trial(study)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Displaying best results.")
        log_best_trial(study)
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()