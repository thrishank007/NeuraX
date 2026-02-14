"""
Logging setup for NeuraX Multimodal RAG System.

Provides setup_logging() to configure loguru and get_logger(name) for
component-specific loggers.
"""
import sys


def setup_logging():
    """Initialize logging configuration using loguru."""
    try:
        from loguru import logger
        from config import LOGGING_CONFIG, LOGS_DIR

        # Remove default handler
        logger.remove()

        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Add file handler
        log_file = LOGS_DIR / "neurax.log"
        logger.add(
            log_file,
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"],
            rotation=LOGGING_CONFIG.get("rotation", "10 MB"),
            retention=LOGGING_CONFIG.get("retention", "1 week"),
            compression=LOGGING_CONFIG.get("compression", "gz"),
        )

        # Add console handler
        logger.add(
            sys.stderr,
            level=LOGGING_CONFIG["level"],
            format=LOGGING_CONFIG["format"],
        )

        logger.info("Logging system initialized")
        return True

    except Exception as e:
        print(f"Failed to setup logging: {e}", file=sys.stderr)
        return False


def get_logger(name: str):
    """Return a loguru logger bound with the given component/module name."""
    from loguru import logger
    return logger.bind(name=name)
