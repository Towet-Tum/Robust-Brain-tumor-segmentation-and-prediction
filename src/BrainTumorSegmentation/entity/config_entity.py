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
