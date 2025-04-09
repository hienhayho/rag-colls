# Ref: https://github.com/deepset-ai/haystack/blob/main/haystack/core/serialization.py

# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import torch
from typing import Any, Dict, Type
from types import ModuleType
from loguru import logger
from threading import Lock

_import_lock = Lock()


class DeserializationError(Exception):
    pass


def thread_safe_import(module_name: str) -> ModuleType:
    """
    Import a module in a thread-safe manner.

    Importing modules in a multi-threaded environment can lead to race conditions.
    This function ensures that the module is imported in a thread-safe manner without having impact
    on the performance of the import for single-threaded environments.

    :param module_name: the module to import
    """
    with _import_lock:
        return importlib.import_module(module_name)


def generate_qualified_class_name(cls: Type[object]) -> str:
    """
    Generates a qualified class name for a class.

    :param cls:
        The class whose qualified name is to be generated.
    :returns:
        The qualified name of the class.
    """
    return f"{cls.__module__}.{cls.__name__}"


def default_to_dict(obj: Any, **init_parameters) -> Dict[str, Any]:
    """
    Utility function to serialize an object to a dictionary.

    This is mostly necessary for components but can be used by any object.
    `init_parameters` are parameters passed to the object class `__init__`.
    They must be defined explicitly as they'll be used when creating a new
    instance of `obj` with `from_dict`. Omitting them might cause deserialisation
    errors or unexpected behaviours later, when calling `from_dict`.

    An example usage:

    ```python
    class MyClass:
        def __init__(self, my_param: int = 10):
            self.my_param = my_param

        def to_dict(self):
            return default_to_dict(self, my_param=self.my_param)


    obj = MyClass(my_param=1000)
    data = obj.to_dict()
    assert data == {
        "type": "MyClass",
        "init_parameters": {
            "my_param": 1000,
        },
    }
    ```

    :param obj:
        The object to be serialized.
    :param init_parameters:
        The parameters used to create a new instance of the class.
    :returns:
        A dictionary representation of the instance.
    """
    return {
        "type": generate_qualified_class_name(type(obj)),
        "init_parameters": init_parameters,
    }


def default_from_dict(cls: Type[object], data: Dict[str, Any]) -> Any:
    """
    Utility function to deserialize a dictionary to an object.

    This is mostly necessary for components but can be used by any object.

    The function will raise a `DeserializationError` if the `type` field in `data` is
    missing or it doesn't match the type of `cls`.

    If `data` contains an `init_parameters` field it will be used as parameters to create
    a new instance of `cls`.

    :param cls:
        The class to be used for deserialization.
    :param data:
        The serialized data.
    :returns:
        The deserialized object.

    :raises DeserializationError:
        If the `type` field in `data` is missing or it doesn't match the type of `cls`.
    """
    init_params = data.get("init_parameters", {})
    if "type" not in data:
        raise DeserializationError("Missing 'type' in serialization data")
    if data["type"] != generate_qualified_class_name(cls):
        raise DeserializationError(
            f"Class '{data['type']}' can't be deserialized as '{cls.__name__}'"
        )
    return cls(**init_params)


def import_class_by_name(fully_qualified_name: str) -> Type[object]:
    """
    Utility function to import (load) a class object based on its fully qualified class name.

    This function dynamically imports a class based on its string name.
    It splits the name into module path and class name, imports the module,
    and returns the class object.

    :param fully_qualified_name: the fully qualified class name as a string
    :returns: the class object.
    :raises ImportError: If the class cannot be imported or found.
    """
    try:
        module_path, class_name = fully_qualified_name.rsplit(".", 1)
        logger.debug(
            "Attempting to import class '{cls_name}' from module '{md_path}'",
            cls_name=class_name,
            md_path=module_path,
        )
        module = thread_safe_import(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as error:
        logger.error(
            "Failed to import class '{full_name}'", full_name=fully_qualified_name
        )
        raise ImportError(f"Could not import class '{fully_qualified_name}'") from error


def serialize_hf_model_kwargs(kwargs: Dict[str, Any]):
    """
    Recursively serialize HuggingFace specific model keyword arguments in-place to make them JSON serializable.

    :param kwargs: The keyword arguments to serialize
    """
    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, torch.dtype):
            kwargs[k] = str(v)

        if isinstance(v, dict):
            serialize_hf_model_kwargs(v)


def deserialize_hf_model_kwargs(kwargs: Dict[str, Any]):
    """
    Recursively deserialize HuggingFace specific model keyword arguments in-place to make them JSON serializable.

    :param kwargs: The keyword arguments to deserialize
    """

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, str) and v.startswith("torch."):
            dtype_str = v.split(".")[1]
            dtype = getattr(torch, dtype_str, None)
            if dtype is not None and isinstance(dtype, torch.dtype):
                kwargs[k] = dtype

        if isinstance(v, dict):
            deserialize_hf_model_kwargs(v)
