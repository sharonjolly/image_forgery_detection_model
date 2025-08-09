from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_source: Path
    resaved_path: Path
    pickle_save: Path
    params: dict

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    load_data: Path
    save_model: Path
    params: dict