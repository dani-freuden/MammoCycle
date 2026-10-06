from typing import Dict, Mapping, Tuple, Union, Optional, Hashable, Any, Set, List
from os import PathLike
from pathlib import Path
from tempfile import mkdtemp
import shutil
import re
from uuid import uuid4
import json

from dotenv import load_dotenv
import torch
from clearml import Task, OutputModel

from src.utils.io import write_lines


# ClearML URLs and credentials need to be loaded from env variables
load_dotenv(override=True)


# Display names
_PROJECT_NAME = 'Segmentation'
_CONFIG = 'Config'


class ClearMLExperiment:
    def __init__(self, *, _task: Task):
        self._task = _task
        self._logger = self._task.get_logger()
        self._temp_dir = Path(mkdtemp())
        self._model = OutputModel(task=self._task, framework='pytorch')

    def __del__(self):
        shutil.rmtree(self._temp_dir)

    @classmethod
    def new(cls, name: str) -> 'ClearMLExperiment':
        if '/' in name or '\\' in name:
            raise ValueError('Experiment name should not contain slashes.')

        existing_tasks = Task.get_tasks(
            project_name=_PROJECT_NAME,
            task_name=f'^{re.escape(name)}$',
            allow_archived=True,
        )

        if existing_tasks:
            raise ValueError(
                f'Experiment "{name}" already exists (possibly in the archive).'
            )

        task = Task.init(
            project_name=_PROJECT_NAME,
            task_name=name,
            output_uri=True,
            auto_connect_frameworks={'pytorch': False, 'matplotlib': False},
        )

        return cls(_task=task)

    @classmethod
    def load(cls, name: str) -> 'ClearMLExperiment':
        existing_tasks = Task.get_tasks(
            project_name=_PROJECT_NAME,
            task_name=f'^{re.escape(name)}$',
            allow_archived=False,
        )

        if len(existing_tasks) != 1:
            raise ValueError(
                f'There are {len(existing_tasks)} ClearML expermients named "{name}" (maybe it is in the archive).'
            )

        return cls(_task=existing_tasks[0])

    def flush(self) -> None:
        self._task.flush(wait_for_uploads=True)

    def set_scalar(self, name: str, variant: str, epoch: int, value: float) -> None:
        self._logger.report_scalar(name, variant, value, epoch)

    def set_image(
        self,
        image_name: str,
        sample_name: str,
        variant: str,
        epoch: int,
        image: Union[torch.Tensor, Path],
        delete_when_done: bool = False,
    ) -> None:
        kwargs = (
            {'image': image.swapaxes(0, -1).numpy()}
            if isinstance(image, torch.Tensor)
            else {'local_path': image}
        )
        title = ' '.join([variant, image_name]).title()
        self._logger.report_image(
            title, sample_name, epoch, delete_after_upload=delete_when_done, **kwargs
        )

    def set_file(self, name: str, filename: Path, **metadata: Any) -> None:
        metadata = metadata or None
        self._task.upload_artifact(
            name, filename, wait_on_upload=True, metadata=metadata
        )

    def set_config(self, config: Mapping[str, Hashable]) -> None:
        config_json = json.dumps(config, indent=4)
        json_filename = self._temp_dir.joinpath(str(uuid4()), 'config.json')
        json_filename.parent.mkdir(parents=True, exist_ok=True)
        json_filename.write_text(config_json)
        self._task.connect_configuration(json_filename, name=_CONFIG)

    def set_model(
        self, path: Path, epoch: Optional[int] = None, name: str = 'model'
    ) -> None:
        self._model.update_weights(
            path,
            target_filename=name,
            iteration=epoch,
            is_package=False,
            auto_delete_file=False,
            async_enable=False,
        )

    def get_config(self) -> Dict[str, Hashable]:
        config = self._task.get_configuration_object_as_dict(_CONFIG)

        if config is None:
            raise LookupError(
                f'No configuration was set for ClearML experiment {self._task.name}'
            )

        return config

    def get_model(self) -> str:
        models = self._task.get_models()['output']

        if not models:
            raise FileNotFoundError(
                f'There are no saved models for the {self._task.name} ClearML experiment.'
            )

        if len(models) > 1:
            raise TypeError(
                f'The {self._task.name} ClearML experimen contains {len(models)} models.'
            )

        path = models[0].get_local_copy(raise_on_error=True, extract_archive=True)

        return path

    def set_cases(self, split: str, cases: Set[Any]) -> None:
        name = f'{split.title()} Set'
        filename = self._temp_dir.joinpath(name).with_suffix('.txt')
        write_lines(filename, cases, sort=True)
        self._task.upload_artifact(
            name,
            filename,
            metadata={'Number Of Cases': len(cases)},
            delete_after_upload=True,
            preview=filename.read_text(),
        )

    def add_tag(self, tag: str) -> None:
        self._task.add_tags([tag])

    def get_tags(self) -> List[str]:
        return list(self._task.get_tags())


