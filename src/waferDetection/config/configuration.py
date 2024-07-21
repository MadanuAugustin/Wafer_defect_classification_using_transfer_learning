

from src.waferDetection.constants import *
from src.waferDetection.utils.common import read_yaml , create_directories
from src.waferDetection.entity.config_entity import (PrepareBaseModelConfig,TrainingConfig,EvaluationConfig)
from pathlib import Path




class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])



    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig :

        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weight = self.params.WEIGHTS,
            params_classes= self.params.CLASSES
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:

        config = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir = Path(config.root_dir),
            trained_model_path= Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data_path= Path(config.training_data_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:

        config = self.config.evaluation
        all_params = self.params

        create_directories([config.root_dir])

        evaluation_config = EvaluationConfig(
            root_dir = Path(config.root_dir),
            trained_model_path= Path(config.trained_model_path),
            training_data_path = Path(config.training_data_path),
            params_image_size = all_params.IMAGE_SIZE,
            params_batch_size = all_params.BATCH_SIZE,
            mlflow_uri = "https://dagshub.com/augustin7766/Wafer_defect_classification_using_transfer_learning.mlflow",

        )

        return evaluation_config

    