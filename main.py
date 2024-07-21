


import sys
from src.waferDetection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.waferDetection import logger, CustomException


STAGE_NAME = 'Prepare Base Model Stage'

try:
    logger.info(f'-----------------{STAGE_NAME} started------------------------')
    preparebasemodel = PrepareBaseModelTrainingPipeline()
    preparebasemodel.main()
    logger.info(f'-----------------{STAGE_NAME} completed successfully-----------------')
except Exception as e:
    raise CustomException(e, sys)