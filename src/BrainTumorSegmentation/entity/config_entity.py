from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreprocessConfig:
    root_dir: Path
    dataset: str
    images: Path
    masks: Path
    splitted_dataset: Path
    dataset_path: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_path: Path
    train_img_dir: str
    train_mask_dir: str
    val_img_dir: str
    val_mask_dir: str
    epochs: int
    batch_size: int
    img_size: int
    lr: float
    num_classes: int
    channels: int
