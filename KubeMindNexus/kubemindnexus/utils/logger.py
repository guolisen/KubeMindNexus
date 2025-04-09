"""Logging utilities for KubeMindNexus."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from ..config.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE, DATA_DIR


def setup_logger(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    max_size: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> logging.Logger:
    """Set up and configure a logger.
    
    Args:
        name: Logger name. If None, uses the root logger.
        level: Log level. If None, uses LOG_LEVEL from settings.
        log_format: Log format string. If None, uses LOG_FORMAT from settings.
        log_file: Log file path. If None, uses LOG_FILE from settings.
        max_size: Maximum log file size in bytes before rotation.
        backup_count: Number of backup log files to keep.
        
    Returns:
        Configured logger instance.
    """
    logger_name = name or "kubemindnexus"
    logger = logging.getLogger(logger_name)
    
    # Set log level
    log_level_str = level or LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(log_format or LOG_FORMAT)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    log_file_path = log_file or LOG_FILE
    if log_file_path:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_size,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default application logger
app_logger = setup_logger("kubemindnexus")


class LoggerMixin:
    """Mixin to add logging capability to a class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this class.
        
        Returns:
            Logger instance for this class.
        """
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(f"kubemindnexus.{self.__class__.__name__}")
        return self._logger
