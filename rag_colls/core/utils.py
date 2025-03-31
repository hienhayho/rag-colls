import re


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
