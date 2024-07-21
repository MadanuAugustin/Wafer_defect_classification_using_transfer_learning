


import sys
from src.waferDetection.pipeline.stage_01_dataIngestion import DataIngestionTrainingPipeline
from src.waferDetection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.waferDetection.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from src.waferDetection.pipeline.stage_04_model_evaluation import EvaluationPipeline
from src.waferDetection import logger, CustomException



STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f'-----------------{STAGE_NAME} started------------------------')
    dataIngestion = DataIngestionTrainingPipeline()
    dataIngestion.main()
    logger.info(f'-----------------{STAGE_NAME} completed successfully-----------------')
except Exception as e:
    raise(CustomException(e, sys))



# STAGE_NAME = 'Prepare Base Model Stage'

# try:
#     logger.info(f'-----------------{STAGE_NAME} started------------------------')
#     preparebasemodel = PrepareBaseModelTrainingPipeline()
#     preparebasemodel.main()
#     logger.info(f'-----------------{STAGE_NAME} completed successfully-----------------')
# except Exception as e:
#     raise CustomException(e, sys)



# STAGE_NAME = 'Model Training'

# try:
#     logger.info(f'-----------------{STAGE_NAME} started------------------------')
#     modeltraining = ModelTrainingPipeline()
#     modeltraining.main()
# except Exception as e:
#     raise CustomException(e, sys)



# STAGE_NAME = 'Model Evaluation'

# try:
#     logger.info(f'-----------------{STAGE_NAME} started------------------------')
#     modeltraining = EvaluationPipeline()
#     modeltraining.main()
# except Exception as e:
#     raise CustomException(e, sys)