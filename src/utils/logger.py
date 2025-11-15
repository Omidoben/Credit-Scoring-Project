import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name
    """
    logger = logging.getLogger(name)
    
    # Add console handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# This module configures project-wide logging.
#
# - Creates a "logs/" directory and writes log files with timestamped names.
# - Sets up a global file logger using logging.basicConfig().
# - Provides get_logger(name) to create module-specific loggers that:
#       • include the module name in log messages,
#       • output logs to both file and console,
#       • avoid duplicate handlers across modules.
#
# Usage Example:
#     from src.utils.logger import get_logger
#     logger = get_logger(__name__)
#     e.g - [2025-11-04 13:22:51] demographics - INFO - Starting demographic generation
