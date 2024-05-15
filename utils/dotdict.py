from typing import Any, Dict, Optional
from argparse import Namespace


class DotDict(Namespace):
    """A class that extends `argparse.Namespace` for dynamic attribute access with better IDE support."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        if name not in self.__dict__:
            self.__dict__[name] = DotDict()
        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = DotDict(**value)
        super().__setattr__(name, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, DotDict) else v) for k, v in self.__dict__.items()}
