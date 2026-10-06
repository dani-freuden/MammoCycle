from typing import Annotated, Callable, Tuple, Type, TypeVar, Dict, Any, ParamSpec
from enum import Enum
from functools import partial
import json
from pathlib import Path
from importlib import import_module

from os import PathLike
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, WithJsonSchema


T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')


class ConfigBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @classmethod
    def from_path(cls: Type[T], path: PathLike) -> T:
        with open(path, 'rt') as fp:
            obj = json.load(fp)

        return cls(**obj)

    def to_path(self, path: PathLike) -> None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)

        with open(path, 'wt') as fp:
            fp.write(self.model_dump_json())

    @classmethod
    def from_name(cls: Type[T], name: str) -> T:
        config_filename = Path(__file__).parents[2].joinpath('config', name).with_suffix('.json')

        if not config_filename.is_file():
            raise ValueError(f'Configuration file {config_filename} does not exist.')

        return cls.from_path(config_filename)


def import_and_instantiate(x: Tuple[str, Any, Dict[str, Any]]) -> Tuple[str, Any, Dict[str, Any]]:
    module_name, name, kwargs = x

    if isinstance(name, str):
        module = import_module(module_name)
        cls = getattr(module, name)
        obj = cls(**kwargs)
    else:
        obj = name

    return module_name, obj, kwargs


def import_and_make_partial(x: Tuple[str, Any, Dict[str, Any]]) -> Tuple[str, Any, Dict[str, Any]]:
    module_name, name, kwargs = x

    if isinstance(name, str):
        module = import_module(module_name)
        func = getattr(module, name)
        func = partial(func, **kwargs)
    else:
        func = name

    return module_name, func, kwargs


def replace_object_with_class_name(x: Tuple[str, Any, Dict[str, Any]]):
    module_name, obj, kwargs = x

    return module_name, obj.__class__.__name__, kwargs


def replace_partial_with_function_name(x: Tuple[str, partial, Dict[str, Any]]):
    module_name, obj, kwargs = x

    return module_name, obj.func.__name__, kwargs


DynamicObject = Annotated[
    Tuple[str, T, Dict[str, Any]],
    BeforeValidator(import_and_instantiate),
    PlainSerializer(replace_object_with_class_name, return_type=Tuple[str, str, Dict[str, Any]], when_used='always'),
    WithJsonSchema({
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "string"}, {"type": "object"}],
        "minItems": 3,
        "maxItems": 3,
    })
]


DynamicFunction = Annotated[
    Tuple[str, Callable[P, R], Dict[str, Any]],
    BeforeValidator(import_and_make_partial),
    PlainSerializer(replace_partial_with_function_name, return_type=Tuple[str, str, Dict[str, Any]], when_used='always'),
    WithJsonSchema({
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "string"}, {"type": "object"}],
        "minItems": 3,
        "maxItems": 3,
    })
]


E = TypeVar('E', bound=Enum)
EnumType = Annotated[E, PlainSerializer(lambda e: e.value, return_type=str, when_used='always')]
