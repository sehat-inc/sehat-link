import logging
import os

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logger configuration
def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with console and rotating file handlers.

    Args:
        name (str): Logger name, typically __name__.
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] | [%(levelname)s] | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(
            os.path.join(LOG_DIR, "app.log"),
            mode="a",
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger







