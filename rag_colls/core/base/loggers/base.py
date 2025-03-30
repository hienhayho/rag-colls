from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Base class for loggers.
    """

    @abstractmethod
    def info(self, message: str, *args, **kwargs):
        """
        Log an info message.
        """
        raise NotImplementedError("info method not implemented")

    @abstractmethod
    def warning(self, message: str, *args, **kwargs):
        """
        Log a warning message.
        """
        raise NotImplementedError("warning method not implemented")

    @abstractmethod
    def critical(self, message: str, *args, **kwargs):
        """
        Log a critical message.
        """
        raise NotImplementedError("critical method not implemented")

    @abstractmethod
    def exception(self, message: str, *args, **kwargs):
        """
        Log an exception message.
        """
        raise NotImplementedError("exception method not implemented")

    @abstractmethod
    def success(self, message: str, *args, **kwargs):
        """
        Log a success message.
        """
        raise NotImplementedError("success method not implemented")

    @abstractmethod
    def error(self, message: str, *args, **kwargs):
        """
        Log an error message.
        """
        raise NotImplementedError("error method not implemented")

    @abstractmethod
    def debug(self, message: str, *args, **kwargs):
        """
        Log a debug message.
        """
        raise NotImplementedError("debug method not implemented")

    @abstractmethod
    def add_file_handler(self, *args, **kwargs):
        """
        Add a file handler to the logger.
        """
        raise NotImplementedError("add_file_handler method not implemented")
