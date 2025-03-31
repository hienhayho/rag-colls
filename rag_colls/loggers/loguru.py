from loguru import logger

from rag_colls.core.base.loggers.base import BaseLogger


class LoguruLogger(BaseLogger):
    """
    Loguru logger implementation.
    """

    def __init__(self):
        """
        Initialize the logger.
        """
        self.logger = logger

    def __str__(self):
        """
        Return a string representation of the logger.
        """
        return "LoguruLogger"

    def info(self, message: str, *args, **kwargs):
        """
        Log an info message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """
        Log a warning message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.warning(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """
        Log a critical message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """
        Log an exception message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.exception(message, *args, **kwargs)

    def success(self, message: str, *args, **kwargs):
        """
        Log a success message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.success(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """
        Log an error message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """
        Log a debug message.

        Args:
            message (str): The message to log.
            *args: Additional arguments for the logger.
            **kwargs: Additional keyword arguments for the logger.
        """
        self.logger.debug(message, *args, **kwargs)

    def add_file_handler(
        self,
        file_name: str,
        level: str | None = None,
        rotation: str | None = None,
        retention: str | None = None,
        **kwargs,
    ):
        """
        Add a file handler to the logger.

        Args:
            file_name (str): The name of the file to log to.
            level (str | None): The logging level.
            rotation (str | None): The rotation policy.
            retention (str | None): The retention policy.
            **kwargs: Additional arguments for the logger.
        """
        self.logger.add(
            file_name, level=level, rotation=rotation, retention=retention, **kwargs
        )
