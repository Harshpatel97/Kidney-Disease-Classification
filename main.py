from KidneyDisease import logger
from KidneyDisease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline



STAGE_NAME1 = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME1} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME1} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e