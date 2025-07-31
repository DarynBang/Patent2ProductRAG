"""
config/logging_config.py

Centralized logging configuration for Patent2ProductRAG system.
This module provides consistent logging setup across all components.

Features:
- File and console logging
- Proper formatting without special characters
- Rotation support for log files
- Different log levels for different components
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    log_filename=None,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5,
    console_output=True
):
    """
    Set up centralized logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: "logs")
        log_filename: Custom log filename (default: auto-generated)
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console as well
    
    Returns:
        logging.Logger: Configured root logger
    """
    
    # Create logs directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"patent2product_rag_{timestamp}.log"
    
    log_file_path = log_dir_path / log_filename
    
    # Create formatter without special characters
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Log the setup
    root_logger.info(f"Logging configured - File: {log_file_path}")
    root_logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return root_logger

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)

def log_system_info():
    """Log basic system information for debugging."""
    logger = get_logger(__name__)
    logger.info("System Information:")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Process ID: {os.getpid()}")

def log_performance(operation_name, start_time, end_time, **kwargs):
    """
    Log performance metrics for an operation.
    
    Args:
        operation_name: Name of the operation
        start_time: Start timestamp
        end_time: End timestamp
        **kwargs: Additional metrics to log
    """
    logger = get_logger("performance")
    duration = end_time - start_time
    
    metrics = [f"Duration: {duration:.2f}s"]
    for key, value in kwargs.items():
        metrics.append(f"{key}: {value}")
    
    logger.info(f"Performance [{operation_name}] - {', '.join(metrics)}")

# Default logging levels for different components
COMPONENT_LOG_LEVELS = {
    'InternshipRAG_pipeline': logging.INFO,
    'agents': logging.INFO,
    'query_generation': logging.INFO,
    'product_suggestion': logging.INFO,
    'utils': logging.WARNING,
    'streamlit': logging.WARNING,
    'transformers': logging.WARNING,
    'langchain': logging.WARNING,
    'chromadb': logging.WARNING,
    'sentence_transformers': logging.WARNING,
}

def configure_component_logging():
    """Configure specific log levels for different components."""
    for component, level in COMPONENT_LOG_LEVELS.items():
        logging.getLogger(component).setLevel(level)
