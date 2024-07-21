
import sys
import os
import mlflow
import mlflow.sklearn
import tensorflow
from pathlib import Path
from src.waferDetection import logger, CustomException
from src.waferDetection.entity.config_entity import EvaluationConfig
from src.waferDetection.utils.common import save_json




class Evaluation:

    def __init__(self, config : EvaluationConfig):
        
        self.config = config

    
    def evaluation(self):

        try:

            logger.info(f'----------started evaluation method--------------------')

            valid_data_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
                rescale = 1./255,
                validation_split = 0.30
            )


            valid_generator = valid_data_generator.flow_from_directory(
                directory = self.config.training_data_path,
                subset = 'validation',
                shuffle = False,
                target_size = self.config.params_image_size[:-1],
                batch_size = self.config.params_batch_size,
                interpolation = 'bilinear'
            )

            logger.info(f'-----------created validatin generator--------------------')


            model = tensorflow.keras.models.load_model(self.config.trained_model_path)

            logger.info(f'-----------successfully loaded trained model------------------')

            logger.info(f'------------valid_generator found {valid_generator.samples} samples---------------')

            score = model.evaluate(valid_generator)

            logger.info(f'----------evaluating model completed-----------------------')

            scores = {'loss' : score[0], 'accuracy' : score[1]}

            save_json(path = Path("score.json", data = scores))

            logger.info(f'--------------the loss of the model is : {score[0]} and accuracy is : {score[1]}')

            os.environ["MLFLOW_TRACKING_URI"]='https://dagshub.com/augustin7766/Wafer_defect_classification_using_transfer_learning.mlflow'
            os.environ["MLFLOW_TRACKING_USERNAME"]="augustin7766"
            os.environ["MLFLOW_TRACKING_PASSWORD"]="8a01ee4bec043666cf3ced22edc7d308526b4b42"


            logger.info(f'------------logging experiment--------------------')

            mlflow.set_experiment('first_exp_01')

            with mlflow.start_run():

                mlflow.log_params(self.config.all_params)

                mlflow.log_metrics(
                    {'loss' : score[0], 'accuracy' : score[1]}
                )

                mlflow.sklearn.log_model(model, 'model', registered_model_name= 'VGG-16')
            
            logger.info(f'----------------params and model successfully logged-------------------')

            logger.info(f'-----------------completed evaluation method-----------------------------')
        
        except Exception as e:
            raise CustomException(e, sys)







    

    
    