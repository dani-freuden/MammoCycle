from typing import Dict, Hashable, Optional, Tuple, List
import json
from pathlib import Path

from pydantic import model_validator
import torch

from src.utils.config import ConfigBase, DynamicFunction



class DataConfig(ConfigBase):
    spacing: float = None
    

class NetConfig(ConfigBase):
    pass

class TrainConfig(ConfigBase):
    max_num_train_cases: Optional[int]
    max_epochs: int
    lr: float
    lr_decay: float
    batch_size: int
    normaliztion_method: DynamicFunction[[torch.Tensor], torch.Tensor]
    augmentations: List[Tuple[float, DynamicFunction[[Dict[Hashable, torch.Tensor]], None]]]

    @model_validator(mode='after')
    def filter_augs_and_losses(self):
        self.augmentations = [
            (prob, aug) for (prob, aug) in self.augmentations if prob > 0.0
        ]

        return self
    
    
class Config(ConfigBase):
    data: DataConfig
    net: NetConfig
    train: TrainConfig


# Update JSON schema every time this module gets imported
with Path(__file__).parents[1].joinpath('config', 'schema.json').open('wt') as f:
    json.dump(Config.model_json_schema(), f, indent=2)
