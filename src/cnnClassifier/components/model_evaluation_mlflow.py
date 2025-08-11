from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from pathlib import Path
from cnnClassifier.utils.common import save_json
import os
import tempfile
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.X_test = None
        self.y_test = None
        self.score = None

    def load_data(self):
        """Loads test data from joblib files specified in config."""
        logger.info(f"Fetching test dataset from: {self.config.load_data}")
        try:
            x_path = Path(self.config.load_data) / 'X_90.joblib'
            y_path = Path(self.config.load_data) / 'y.joblib'
            X = joblib.load(x_path)
            y = joblib.load(y_path)
            logger.info(f"Data successfully loaded — X: {X.shape}, y: {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Unable to load dataset: {e}")
            raise

    def split_data(self, X, y):
        """Splits data into training and testing sets."""
        logger.info("Separating dataset into training and testing subsets")
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Test subset prepared — X_test: {X_test.shape}, y_test: {y_test.shape}")
            return X_test, y_test
        except Exception as e:
            logger.error(f"Failed during dataset splitting: {e}")
            raise

    def preprocess_data(self, X_test, y_test):
        """Reshapes test data for CNN input."""
        logger.info("Reshaping test data for model compatibility")
        try:
            X_test = X_test.reshape(X_test.shape[0], 128, 128, 3)
            y_test = y_test.reshape(y_test.shape[0], 2)
            logger.info(f"Data reshaped — X_test: {X_test.shape}, y_test: {y_test.shape}")
            self.X_test, self.y_test = X_test, y_test
        except Exception as e:
            logger.error(f"Error while reshaping data: {e}")
            raise

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Loads the trained model."""
        logger.info(f"Retrieving trained model from path: {path}")
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_test_generator(self):
        """Returns a Sequence generator for test data."""
        class TestGenerator(Sequence):
            def __init__(self, X, y, batch_size, **kwargs):
                super().__init__(**kwargs)
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.indexes = np.arange(len(self.X))

            def __len__(self):
                return int(np.floor(len(self.X) / self.batch_size))

            def __getitem__(self, index):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                X = [self.X[k] for k in indexes]
                y = [self.y[k] for k in indexes]
                return np.array(X), np.array(y)

        return TestGenerator(self.X_test, self.y_test, self.config.params['batch_size'])

    def evaluation(self):
        """Evaluates the model and saves scores."""
        logger.info("Starting model evaluation process")
        try:
            X, y = self.load_data()
            X_test, y_test = self.split_data(X, y)
            self.preprocess_data(X_test, y_test)

            model_path = Path(self.config.model_path) / self.config.model
            self.model = self.load_model(model_path)

            test_generator = self.get_test_generator()

            logger.info("Evaluating model performance on test dataset")
            self.score = self.model.evaluate(
                test_generator,
                batch_size=self.config.params['batch_size'],
                return_dict=True
            )
            logger.info(f"Evaluation completed — Scores: {self.score}")

            self.save_score()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def save_score(self):
        """Saves evaluation scores to a JSON file."""
        logger.info("Saving evaluation results to JSON file")
        try:
            f1_score = self.score.get('f1_score', 0.0)
            if isinstance(f1_score, tf.Tensor):
                f1_score = np.mean(f1_score.numpy())
            elif isinstance(f1_score, np.ndarray):
                f1_score = np.mean(f1_score)

            scores = {
                "loss": float(self.score.get('loss', 0.0)),
                "accuracy": float(self.score.get('accuracy', 0.0)),
                "precision": float(self.score.get('precision', 0.0)),
                "recall": float(self.score.get('recall', 0.0)),
                "f1_score": float(f1_score)
            }
            save_json(path=Path("scores.json"), data=scores)
            logger.info("Results successfully saved to scores.json")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def log_into_mlflow(self):
        """Logs parameters, metrics, and model to MLflow."""
        logger.info("Initiating MLflow logging sequence")
        try:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logger.info(f"MLflow tracking URI set to: {self.config.mlflow_uri}")

            with mlflow.start_run():
                mlflow.log_params(self.config.params)

                f1_score = self.score.get('f1_score', 0.0)
                if isinstance(f1_score, tf.Tensor):
                    f1_score = np.mean(f1_score.numpy())
                elif isinstance(f1_score, np.ndarray):
                    f1_score = np.mean(f1_score)

                mlflow.log_metrics({
                    "loss": float(self.score.get('loss', 0.0)),
                    "accuracy": float(self.score.get('accuracy', 0.0)),
                    "precision": float(self.score.get('precision', 0.0)),
                    "recall": float(self.score.get('recall', 0.0)),
                    "f1_score": float(f1_score)
                })

                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_model_path = os.path.join(tmpdirname, "model.keras")
                    logger.info(f"Saving model temporarily at: {temp_model_path}")
                    self.model.save(temp_model_path)
                    if not os.path.exists(temp_model_path):
                        raise FileNotFoundError(f"Model file not found at {temp_model_path}")
                    logger.info(f"Model saved — size: {os.path.getsize(temp_model_path)} bytes")

                    logger.info("Uploading model artifact to MLflow")
                    mlflow.log_artifact(temp_model_path, artifact_path="model")
                    logger.info("Model artifact uploaded successfully")

                if tracking_url_type_store != "file":
                    logger.info("Attempting model registration in MLflow registry")
                    client = MlflowClient()
                    run_id = mlflow.active_run().info.run_id

                    try:
                        client.get_registered_model("image_forgery_detection_model")
                    except RestException:
                        logger.info("Model not found in registry — creating new entry")
                        client.create_registered_model("image_forgery_detection_model")

                    try:
                        source = mlflow.get_artifact_uri("model")
                        logger.info(f"Model source URI: {source}")
                        result = client.create_model_version(
                            name="image_forgery_detection_model",
                            source=source,
                            run_id=run_id
                        )
                        logger.info(f"Model registered successfully — version {result.version}")
                    except RestException as e:
                        logger.error(f"MLflow registry error: {e.__class__.__name__} - {str(e)}")
                        if hasattr(e, "message"):
                            logger.error(f"Message: {e.message}")
                        if hasattr(e, "error_code"):
                            logger.error(f"Error code: {e.error_code}")
                        raise
                    except Exception as e:
                        logger.error(f"Unexpected model registry error: {e}")
                        raise

        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            raise
