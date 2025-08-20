"""
Experiment-aware logging system for scryptorum.

This module provides a logging system that automatically directs logs to experiment
run directories while maintaining console output for user feedback.
"""

import logging
import sys
from pathlib import Path

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] <%(filename)s:%(lineno)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ExperimentHandler(logging.Handler):
    """Custom handler that writes logs to experiment run directories."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.log_file = run_dir / "experiment.log"

        # Ensure parent directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.setFormatter(formatter)

    def emit(self, record):
        """Write log record to experiment log file."""
        try:
            msg = self.format(record)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            # Fail silently to avoid breaking the experiment
            pass


def create_experiment_logger(
    run_dir: Path, logger_name: str = "scryptorum"
) -> logging.Logger:
    """
    Create a logger for an experiment run.

    Args:
        run_dir: Directory where logs should be written
        logger_name: Name of the logger

    Returns:
        Configured logger instance
    """
    # Create a unique logger name to avoid conflicts between runs
    unique_logger_name = f"{logger_name}.{run_dir.name}"
    logger = logging.getLogger(unique_logger_name)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Console handler for user feedback (simpler format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)

    # Experiment file handler (detailed format)
    file_handler = ExperimentHandler(run_dir)
    file_handler.setLevel(logging.DEBUG)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def cleanup_logger(logger: logging.Logger) -> None:
    """Clean up a logger's handlers."""
    if logger:
        # Close all handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
