



from src.waferDetection.config.configuration import ConfigurationManager
from src.waferDetection.components.prepare_base_model import PrepareBaseModel






STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        prepare_base_model_config = config.get_prepare_base_model_config()

        prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)

        prepare_base_model.model_construction()