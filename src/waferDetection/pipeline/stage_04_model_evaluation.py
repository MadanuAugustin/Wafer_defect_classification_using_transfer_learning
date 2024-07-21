


from src.waferDetection.config.configuration import ConfigurationManager
from src.waferDetection.components.model_evaluation import Evaluation





STAGE_NAME = 'Model Evaluation Stage'

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self) : 
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        eval = Evaluation(eval_config)
        eval.evaluation()