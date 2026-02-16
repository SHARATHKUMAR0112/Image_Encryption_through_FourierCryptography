"""
Logging configuration for the Fourier encryption system.

Provides structured logging with configurable levels and output formats.
All terminal output is automatically logged to timestamped files in the logs/ folder.
"""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Default logging configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging.
    
    Adds context information and ensures sensitive data is not logged.
    Uses the Sanitizer class for comprehensive data sanitization.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Import here to avoid circular dependency
        from fourier_encryption.security.sanitizer import Sanitizer
        
        # Sanitize any sensitive data in the record
        if hasattr(record, "args") and isinstance(record.args, dict):
            record.args = Sanitizer.sanitize_dict(record.args)
        
        # Sanitize the message itself
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = Sanitizer.sanitize_exception_message(record.msg)
        
        # Add context information
        if not hasattr(record, "component"):
            record.component = record.name.split(".")[-1]
        
        return super().format(record)


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    log_file: Optional[Path] = None,
    format_string: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    enable_file_logging: bool = True,
) -> None:
    """
    Configure logging for the application.
    
    All terminal output is automatically logged to timestamped files in logs/ folder.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific file path for log output (overrides default timestamped file)
        format_string: Log message format string
        date_format: Date format string
        enable_file_logging: Whether to enable automatic file logging (default: True)
    """
    # Create handlers
    handlers: Dict[str, Dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "structured",
            "stream": sys.stdout,
        }
    }
    
    # Add file handler with timestamp
    if enable_file_logging:
        if log_file is None:
            # Generate timestamped log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOGS_DIR / f"fourier_encryption_{timestamp}.log"
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "structured",
            "filename": str(log_file),
            "mode": "a",
        }
        
        # Also add a latest.log symlink/copy for easy access
        latest_log = LOGS_DIR / "latest.log"
        handlers["latest"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "structured",
            "filename": str(latest_log),
            "mode": "w",  # Overwrite for latest
        }
    
    # Logging configuration dictionary
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
                "format": format_string,
                "datefmt": date_format,
            },
            "simple": {
                "format": "%(levelname)s - %(message)s",
            },
        },
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            "fourier_encryption": {
                "level": level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Initialize default logging on module import with file logging enabled
setup_logging(enable_file_logging=True)
