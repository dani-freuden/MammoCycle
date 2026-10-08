from typing import Optional, Tuple
import json
from pathlib import Path

from pydantic import Field, model_validator

from src.utils.config import ConfigBase


STRIDE = 8  # ResNetEncoder.stride; frame sizes must be multiples of it


class DataConfig(ConfigBase):
    height: int = 1024  # 512 x 320 is the fast setting
    width: int = 640  # canvas: ~90% of breasts fill the height at 1024; wider ones shrink to fit, never stretched
    tissue_threshold: float = 0.01  # background and padding are below it; see check_tissue_threshold.py

    @model_validator(mode='after')
    def check_frame_size(self):
        if self.height % STRIDE or self.width % STRIDE:
            raise ValueError(f'{self.height = } and {self.width = } must be multiples of {STRIDE}')

        return self


class AugmentationConfig(ConfigBase):
    """FrameAugmentation(**model_dump())"""
    max_shift: float = 48.0
    max_rotation_degrees: float = 5.0
    scale_range: Tuple[float, float] = (1.0, 1.0)
    reverse_probability: float = 0.5
    intensity: bool = True


class EncoderConfig(ConfigBase):
    """build_encoder(config)"""
    checkpoint: Optional[Path]  # mirai-shira checkpoint; null trains from random weights
    calibrated: Optional[Path] = None  # written by scripts/calibrate_encoder.py; training refuses to start without it
    mean: float = 0.11866  # Mirai's intensity normalization, on the [0, 1] scale
    std: float = 0.18850
    freeze_batch_norm: bool = True  # false only when training from random weights
    dilate: bool = True


class ModuleConfig(ConfigBase):
    """MammoCycleModule(encoder, tissue_threshold=data.tissue_threshold, **model_dump())"""
    patch_size: int = 160
    num_patches: int = 4
    min_tissue_fraction: float = Field(0.75, gt=0, le=1)
    # The largest temperature whose long cycle on identical frames stays under 4 px with the Mirai weights: 3.8 px at
    # 1024 x 640, where 0.05 gives 13.9 px. Re-check with scripts/misc/temperature_sweep.py when the encoder changes.
    temperature: float = 0.03
    max_rotation_degrees: float = 20.0
    cycle_weight: float = 1.0
    similarity_weight: float = 1.0
    huber_delta: Optional[float] = None  # null: squared cycle distance; otherwise Huber, in patch sizes (0.25)
    top_k: int = 5
    propagation_radius: Optional[float] = None  # pixels; null searches the whole frame
    label_threshold: float = 0.5
    cycle_tolerance_px: float = 16.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000


class TrainConfig(ConfigBase):
    max_epochs: int
    batch_size: int = 8
    max_exams: int = Field(4, ge=2)  # exams per training record; 2 trains on pairs only
    precision: str = '16-mixed'
    gradient_clip_val: float = 1.0
    val_check_interval: Optional[int] = None  # training steps between validations; null validates once per epoch
    limit_val_batches: Optional[int] = None  # null validates on every dev pair


class Config(ConfigBase):
    data: DataConfig = Field(default_factory=DataConfig)
    augmentation: Optional[AugmentationConfig] = Field(default_factory=AugmentationConfig)  # null turns it off
    encoder: EncoderConfig
    module: ModuleConfig = Field(default_factory=ModuleConfig)
    train: TrainConfig


# Update JSON schema every time this module gets imported
with Path(__file__).parents[1].joinpath('config', 'schema.json').open('wt') as f:
    json.dump(Config.model_json_schema(), f, indent=2)
