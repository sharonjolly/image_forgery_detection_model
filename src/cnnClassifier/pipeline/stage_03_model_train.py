from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import ModelTrainer
from cnnClassifier import logger

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        trainer = ModelTrainer(config=model_trainer_config)
        history = trainer.run_training_pipeline()
        logger.info("Model training pipeline completed")


if __name__=='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e