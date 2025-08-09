
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                DataPreprocessingConfig,
                                                ModelTrainerConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        
        config = self.config.data_preprocessing
        params = self.params.preprocessing
        
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            data_source=Path(config.data_source),
            resaved_path=Path(config.resaved_path),
            pickle_save=Path(config.pickle_save),
            params=params
        )

        return data_preprocessing_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        
        config = self.config.model_trainer
        params = self.params.trainer

        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            load_data=Path(config.load_data),
            save_model=Path(config.save_model),
            params=params
        )

        return model_trainer_config