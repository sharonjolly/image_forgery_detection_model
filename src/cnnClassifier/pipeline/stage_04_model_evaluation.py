from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import ModelEvaluation
from cnnClassifier import logger
import dagshub
dagshub.init(repo_owner='sharonjolly',repo_name='image_forgery_detection_model',mlflow=True)

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(eval_config)
        evaluation.evaluation()
        # evaluation.log_into_mlflow()
        logger.info("Model evaluation pipeline completed")


if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
