from KidneyDisease.config.configuration import ConfigurationManager
from KidneyDisease.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.trainer()