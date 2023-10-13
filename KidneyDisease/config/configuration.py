from KidneyDisease.constants import *
from KidneyDisease.utils.common import read_yaml, create_directories
from KidneyDisease.entity import DataIngestionConfig, ModelTrainerConfig

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
    
    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config = self.config.model_trainer
        
        create_directories([config.model])
        
        model_trainer_config = ModelTrainerConfig(
            data=config.data,
            model=config.model
        )
        return model_trainer_config