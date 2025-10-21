import logging
import os
import sys
from typing import Tuple

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_utils import load_test_config
from src.testing.tester import UNetTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('testing.log')
    ]
)
logger = logging.getLogger(__name__)

def main() -> Tuple[float, float, float]:
    """Run the U-Net testing pipeline.

    Returns:
        Tuple[float, float, float]: Test loss, average IoU score, and average Dice score.

    Raises:
        RuntimeError: If the testing pipeline fails.
    """
    try:
        config = load_test_config(os.path.join('config', 'config.yaml'))
        tester = UNetTester(config)
        test_loss, avg_iou, avg_dice = tester.test_model()
        logger.info("Testing completed successfully")
        return test_loss, avg_iou, avg_dice
    except Exception as e:
        logger.error(f"Testing pipeline failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Testing pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()