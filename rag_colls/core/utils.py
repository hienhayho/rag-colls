import re
import time
from typing import Any, Callable, Tuple
from loguru import logger


def check_placeholders(template: str, placeholders: list[str]) -> bool:
    """
    Check if all placeholders are present in the template.

    Args:
        template (str): The template string.
        placeholders (list[str]): A list of placeholders to check for.

    Returns:
        bool: True if all placeholders are present, False otherwise.
    """
    for placeholder in placeholders:
        pattern = r"\{" + re.escape(placeholder) + r"\}"
        if not re.search(pattern, template):
            return False
    return True


def extract_placeholders(template: str) -> list[str]:
    """
    Extract all placeholders from the template.

    Args:
        template (str): The template string.

    Returns:
        list[str]: A list of extracted placeholders.
    """
    return re.findall(r"\{(.*?)\}", template)


def check_torch_device(device: str) -> str:
    """
    Check if the specified device is available in PyTorch.

    Args:
        device (str): The device to check (e.g., "cuda", "cpu").

    Returns:
        str: The available device ("cuda" or "cpu").
    """
    import torch

    try:
        return (
            torch.device(device)
            if device != "auto"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except Exception as e:
        logger.warning(f"Error checking device '{device}': {e}, defaulting to 'cpu'")
        return torch.device("cpu")


def run_fuction_return_time(
    f: Callable, *args: Any, **kwargs: Any
) -> Tuple[float, Any]:
    """
    Run a function and return the execution time and result.

    Args:
        f (Callable): The function to run.
        *args (Any): Positional arguments for the function.
        **kwargs (Any): Keyword arguments for the function.

    Returns:
        Tuple[float, Any]: A tuple containing the execution time in seconds and the result of the function.
    """
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result
