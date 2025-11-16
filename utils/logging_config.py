import logging
import sys


def _configure_root_logger() -> None:
    if getattr(_configure_root_logger, "_configured", False):
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    _configure_root_logger._configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)


def console_log(message: str) -> None:
    print(message)
