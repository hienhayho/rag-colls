import logging
from rich.logging import RichHandler

from rag_colls.core.base.loggers.base import BaseLogger


class RichLogger(BaseLogger):
    def __init__(
        self,
        name: str = "rich",
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        handler = RichHandler()
        formatter = logging.Formatter(fmt=format, datefmt="[%X]")
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def __str__(self):
        return "RichLogger"

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self.logger.exception(message, *args, **kwargs)

    def success(self, message, *args, **kwargs):
        # Does not have a built-in success method, using info instead
        self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def add_file_handler(self, file_name: str, *args, **kwargs):
        """
        Add a file handler to the logger.
        Args:
            file_name (str): The name of the file to log to.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
