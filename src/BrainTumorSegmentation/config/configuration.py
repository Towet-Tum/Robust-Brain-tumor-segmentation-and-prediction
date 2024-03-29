from BrainTumorSegmentation.constants import *
from BrainTumorSegmentation.utils.common import read_yaml, create_directories
from BrainTumorSegmentation.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessConfig,
    TrainingConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, param_file_path=PARAM_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.param = read_yaml(param_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_preprocess_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocess
        create_directories(
            [config.root_dir, config.images, config.masks, config.splitted_dataset]
        )
        data_preprocess_config = DataPreprocessConfig(
            root_dir=config.root_dir,
            dataset="artifacts/data_ingestion/BraTs2020/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/",
            images=config.images,
            masks=config.masks,
            splitted_dataset=config.splitted_dataset,
            dataset_path=config.dataset_path,
        )
        return data_preprocess_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params
        create_directories([config.root_dir])
        training_config = TrainingConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            train_img_dir="artifacts/data_preprocess/processed_dataset/train/images/",
            train_mask_dir="artifacts/data_preprocess/processed_dataset/train/masks/",
            val_img_dir="artifacts/data_preprocess/processed_dataset/val/images/",
            val_mask_dir="artifacts/data_preprocess/processed_dataset/val/masks/",
            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            img_size=params.IMG_SIZE,
            num_classes=params.NUM_CLASSES,
            channels=params.CHANNELS,
            lr=params.LEARNING_RATE,
        )
        return training_config
