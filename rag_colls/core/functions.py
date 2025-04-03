import inspect
from loguru import logger
from pydantic import BaseModel
from typing import get_type_hints, Type, get_args, get_origin, Callable, Any

from rag_colls.core.constants import DEBUG_MODE


def log_debug(message: str):
    """
    Logs a debug message if DEBUG_MODE is enabled.

    Args:
        message (str): The message to log.
    """
    if DEBUG_MODE:
        logger.debug(message)


def create_function_schema(func: Callable[..., Any]) -> dict:
    """
    Create a schema for a function based on its parameters and return type.
    Used in function calling LLMs.

    Args:
        func (Callable[..., Any]): The function to create a schema for.

    Returns:
        dict: A dictionary representing the schema of the function.
    """
    type_hints = get_type_hints(func)
    params = inspect.signature(func).parameters

    def get_model_properties(model_class: Type[BaseModel]):
        properties = {}
        for field_name, field in model_class.model_fields.items():
            field_info = {"type": "string"}  # Default type

            # Get field type
            field_type = field.annotation
            if get_origin(field_type) is list:
                field_info["type"] = "array"
                item_type = get_args(field_type)[0]
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    field_info["items"] = {
                        "type": "object",
                        "properties": get_model_properties(item_type),
                    }
                else:
                    field_info["items"] = {"type": "string"}  # Default for array items
            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                field_info = {
                    "type": "object",
                    "properties": get_model_properties(field_type),
                }
            elif field_type is int:
                field_info["type"] = "integer"
            elif field_type is float:
                field_info["type"] = "number"
            elif field_type is bool:
                field_info["type"] = "boolean"

            # Add description if available
            if field.description:
                field_info["description"] = field.description

            properties[field_name] = field_info
        return properties

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Get the {func.__name__}",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    # Process each parameter
    for param_name, param in params.items():
        param_type = type_hints.get(param_name)

        # Handle Pydantic models
        if isinstance(param_type, type) and issubclass(param_type, BaseModel):
            schema["function"]["parameters"]["properties"][param_name] = {
                "type": "object",
                "properties": get_model_properties(param_type),
            }
        else:
            # Handle basic types
            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
            }
            param_info = {
                "type": type_mapping.get(param_type, "string"),
                "description": f"The {param_name} parameter",
            }

            # Handle lists/arrays
            if get_origin(param_type) is list:
                param_info = {
                    "type": "array",
                    "items": {"type": "string"},  # Default type for array items
                }

            schema["function"]["parameters"]["properties"][param_name] = param_info

        # Add required parameters
        if param.default is inspect.Parameter.empty:
            schema["function"]["parameters"]["required"].append(param_name)

    return schema
