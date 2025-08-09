
import joblib
import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import Precision, F1Score, Recall
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import ModelTrainerConfig

class ModelTrainer(Sequence):
    def __init__(self, config: ModelTrainerConfig, **kwargs):
        """
        Initializes the ModelTrainer with configuration.
        """
        super().__init__(**kwargs)  # Initialize Sequence base class
        logger.info("Initializing ModelTrainer")
        self.config = config
        self.params = config.params
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indexes = None
        self.val_indexes = None

    def __len__(self):
        """
        Returns the number of training batches per epoch.
        """
        return int(np.floor(len(self.X_train) / self.params['batch_size']))

    def __getitem__(self, index):
        """
        Retrieves a training batch of data.
        """
        indexes = self.train_indexes[index * self.params['batch_size']:(index + 1) * self.params['batch_size']]
        X = [self.X_train[k] for k in indexes]
        y = [self.y_train[k] for k in indexes]
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        """
        Shuffles training indices at the end of each epoch.
        """
        self.train_indexes = np.arange(len(self.X_train))
        np.random.shuffle(self.train_indexes)

    def get_validation_data(self):
        """
        Returns the validation data generator.
        """
        class ValidationGenerator(Sequence):
            def __init__(self, X, y, batch_size, **kwargs):
                super().__init__(**kwargs)  # Initialize Sequence base class
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

        return ValidationGenerator(self.X_test, self.y_test, self.params['batch_size'])

    def load_data(self):
        """
        Loads X and y data from joblib files specified in config.
        """
        logger.info(f"Loading data from {self.config.load_data}")
        try:
            x_path = self.config.load_data / 'X_90.joblib'
            y_path = self.config.load_data / 'y.joblib'
            X = joblib.load(x_path)
            y = joblib.load(y_path)
            logger.info(f"Loaded X with shape {X.shape} and y with shape {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def split_data(self, X, y):
        """
        Splits data into training and testing sets.
        """
        logger.info("Splitting data into train and test sets")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Train set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            logger.info(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

    def preprocess_data(self, X_train, X_test, y_train, y_test):
        """
        Reshapes data for CNN input and initializes indices.
        """
        logger.info("Preprocessing data")
        try:
            # Reshape for CNN input (128x128x3)
            X_train = X_train.reshape(X_train.shape[0], 128, 128, 3)
            X_test = X_test.reshape(X_test.shape[0], 128, 128, 3)
            y_train = y_train.reshape(y_train.shape[0], 2)
            y_test = y_test.reshape(y_test.shape[0], 2)
            logger.info(f"Reshaped X_train to {X_train.shape}, X_test to {X_test.shape}")
            logger.info(f"Reshaped y_train to {y_train.shape}, y_test to {y_test.shape}")

            # Store reshaped data and initialize indices
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.train_indexes = np.arange(len(self.X_train))
            np.random.shuffle(self.train_indexes)
            logger.info("Data preprocessing completed")
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def build_model(self):
        """
        Builds and returns a CNN model.
        """
        logger.info("Building CNN model")
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            logger.info("Model built successfully")
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def train(self):
        """
        Trains the model with the specified configuration and saves the best model.
        """
        logger.info("Starting model training")
        try:
            # Build model
            self.model = self.build_model()
            
            # Compile model with parameters from params.yaml
            optimizer = self.params['optimizer']
            metrics = [metric.lower() if metric == 'accuracy' else getattr(tf.keras.metrics, metric)() 
                       for metric in self.params['metrics']]
            self.model.compile(optimizer=optimizer, 
                              loss='categorical_crossentropy', 
                              metrics=metrics)
            self.model.summary()

            # Define callbacks
            cal1 = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=self.params['patience'], 
                restore_best_weights=True
            )
            cal2 = tf.keras.callbacks.ModelCheckpoint(
                str(self.config.save_model / 'model.keras'), 
                monitor='val_loss', 
                save_best_only=True
            )
            
            # Train model with default Keras progress bar
            history = self.model.fit(
                self,
                epochs=self.params['epochs'],
                validation_data=self.get_validation_data(),
                callbacks=[cal1, cal2],
                verbose=1  # Use default Keras progress bar
            )
            logger.info("Model training completed")
            return history.history
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def run_training_pipeline(self):
        """
        Runs the complete training pipeline: load, split, preprocess, and train.
        """
        logger.info("Starting training pipeline")
        try:
            # Load data
            X, y = self.load_data()
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Free memory
            del X
            del y
            gc.collect()
            logger.info("Memory cleared after loading and splitting")
            
            # Preprocess data
            self.preprocess_data(X_train, X_test, y_train, y_test)
            
            # Train model
            history = self.train()
            logger.info("Training pipeline completed")
            return history
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise
