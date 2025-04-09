import re
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
