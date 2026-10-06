from typing import Dict, Mapping, Hashable, Union, Optional, Protocol, Set, Any, List
from pathlib import Path
import torch


class Experiment(Protocol):
    @classmethod
    def new(cls, name: str) -> 'Experiment':
        ...

    @classmethod
    def load(cls, name: str) -> 'Experiment':
        ...

    def flush(self) -> None:
        ...

    def set_scalar(self, name: str, variant: str, epoch: int, value: float) -> None:
        ...

    def set_image(self, image_name: str, sample_name: str, variant: str, epoch: int, image: Union[torch.Tensor, Path], delete_when_done: bool = False) -> None:
        ...

    def set_file(self, name: str, filename: Path, **metadata: Any) -> None:
        ...

    def set_config(self, config: Mapping[str, Hashable]) -> None:
        ...

    def set_model(self, path: Path, epoch: Optional[int] = None, name: str = 'model') -> None:
        ...

    def get_config(self) -> Dict[str, Hashable]:
        ...

    def get_model(self) -> str:
        ...

    def set_cases(self, split: str, cases: Set[Any]) -> None:
        ...

    def add_tag(self, tag: str) -> None:
        ...

    def get_tags(self) -> List[str]:
        ...
