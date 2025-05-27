"""
Experiment-aware logging system for scryptorum.

This module provides a logging system that automatically directs logs to experiment
run directories while maintaining console output for user feedback.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Global state for current run context
_current_run_dir: Optional[Path] = None
_experiment_logger: Optional[logging.Logger] = None


class ExperimentHandler(logging.Handler):
    """Custom handler that writes logs to experiment run directories."""
    
    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.log_file = run_dir / "experiment.log"
        
        # Ensure parent directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.setFormatter(formatter)
    
    def emit(self, record):
        """Write log record to experiment log file."""
        try:
            msg = self.format(record)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except Exception:
            # Fail silently to avoid breaking the experiment
            pass


def setup_experiment_logging(run_dir: Path, logger_name: str = "scryptorum") -> logging.Logger:
    """
    Set up logging for an experiment run.
    
    Args:
        run_dir: Directory where logs should be written
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    global _current_run_dir, _experiment_logger
    
    _current_run_dir = run_dir
    
    # Create logger if it doesn't exist or if run directory changed
    if _experiment_logger is None or _current_run_dir != run_dir:
        _experiment_logger = logging.getLogger(logger_name)
        
        # Clear existing handlers to avoid duplicates
        _experiment_logger.handlers.clear()
        _experiment_logger.setLevel(logging.INFO)
        
        # Console handler for user feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Experiment file handler
        file_handler = ExperimentHandler(run_dir)
        file_handler.setLevel(logging.DEBUG)
        
        # Add handlers
        _experiment_logger.addHandler(console_handler)
        _experiment_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        _experiment_logger.propagate = False
    
    return _experiment_logger


def get_experiment_logger() -> Optional[logging.Logger]:
    """Get the current experiment logger, if any."""
    return _experiment_logger


def log_info(message: str) -> None:
    """Log an info message. Falls back to print if no experiment logger."""
    if _experiment_logger:
        _experiment_logger.info(message)
    else:
        print(message)


def log_debug(message: str) -> None:
    """Log a debug message. Falls back to nothing if no experiment logger."""
    if _experiment_logger:
        _experiment_logger.debug(message)


def log_warning(message: str) -> None:
    """Log a warning message. Falls back to print if no experiment logger."""
    if _experiment_logger:
        _experiment_logger.warning(message)
    else:
        print(f"WARNING: {message}")


def log_error(message: str) -> None:
    """Log an error message. Falls back to print if no experiment logger."""
    if _experiment_logger:
        _experiment_logger.error(message)
    else:
        print(f"ERROR: {message}")


def cleanup_experiment_logging() -> None:
    """Clean up experiment logging context."""
    global _current_run_dir, _experiment_logger
    
    if _experiment_logger:
        # Close all handlers
        for handler in _experiment_logger.handlers[:]:
            handler.close()
            _experiment_logger.removeHandler(handler)
    
    _current_run_dir = None
    _experiment_logger = None