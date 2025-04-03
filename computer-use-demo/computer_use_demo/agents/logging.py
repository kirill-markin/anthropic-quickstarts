"""
Logging module for Computer Use Demo agents.

Configures logging with output to files in the logs/ directory
instead of terminal output.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Global variables to track the logging initialization state
_logging_initialized = False
_current_log_file = None


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for all application modules.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Root logger
    """
    global _logging_initialized, _current_log_file

    # If logging is already initialized, just return the root logger
    if _logging_initialized:
        return logging.getLogger()

    # Convert log level to uppercase and validate
    log_level = log_level.upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        log_level = "INFO"

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with current UTC date/time
    # Using timezone.utc instead of deprecated utcnow()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"agent_{timestamp}_UTC.log"
    _current_log_file = log_file

    # Configure log formatting
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear logger handlers if they were already configured
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure handler for file output
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)

    # Configure loggers for different modules
    agents_logger = logging.getLogger("computer_use_demo.agents")
    agents_logger.setLevel(getattr(logging, log_level))

    streamlit_logger = logging.getLogger("computer_use_demo.streamlit")
    streamlit_logger.setLevel(getattr(logging, log_level))

    # Отключаем детальное логирование HTTP-клиентов, чтобы избежать
    # проблем с рекурсивными вызовами при закрытии сессий
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log startup information
    root_logger.debug("===========================================")
    root_logger.debug(f"Logging initialized at {timestamp} UTC")
    root_logger.debug(f"Log level: {log_level}")
    root_logger.debug(f"Log file: {log_file}")
    root_logger.debug("===========================================")

    # Mark logging as initialized
    _logging_initialized = True

    return root_logger


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger for the specified module.

    Args:
        name: Module name for the logger
        log_level: Optional logging level for this specific logger

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if log_level:
        logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def get_current_log_file() -> Optional[Path]:
    """
    Returns the path to the current log file.

    Returns:
        Path to the current log file or None if logging is not initialized
    """
    return _current_log_file
