from KidneyDisease import logger
from KidneyDisease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from KidneyDisease.pipeline.stage_02_model_trainer import ModelTrainerPipeline


STAGE_NAME1 = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME1} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME1} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e
     
STAGE_NAME2 = "Model Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME2} started <<<<<<") 
   model_trainer = ModelTrainerPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME2} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e