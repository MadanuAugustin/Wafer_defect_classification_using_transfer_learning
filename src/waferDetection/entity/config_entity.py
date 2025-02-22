


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    local_data_path : Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir : Path
    base_model_path : Path
    updated_base_model_path : Path
    params_image_size : list
    params_learning_rate : float
    params_include_top : bool
    params_weight : str
    params_classes : int



@dataclass(frozen=True)
class TrainingConfig:
    root_dir : Path
    trained_model_path : Path
    training_data_path : Path
    updated_base_model_path : Path
    params_epochs : int
    params_batch_size : int
    params_is_augmentation : bool
    params_image_size : list




@dataclass(frozen=True)
class EvaluationConfig:
    root_dir : Path
    trained_model_path : Path
    training_data_path : Path
    all_params : dict
    mlflow_uri : str
    params_image_size : list
    params_batch_size : int