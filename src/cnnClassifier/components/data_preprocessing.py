import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError
import os
import joblib
import shutil
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initializes the DataPreprocessing class with configuration
        """
        logger.info("Initializing DataPreprocessing")
        self.config = config
        self.params = config.params
        self.data_label = ['Au', 'Tp']
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        """
        Creates a DataFrame with image paths and labels from the data source
        """
        logger.info(f"Creating DataFrame from data source: {self.config.data_source}")
        label_lst = []
        img_lst = []
        for label in self.data_label:
            label_path = os.path.join(self.config.data_source, label)
            if not os.path.exists(label_path):
                logger.error(f"Directory not found: {label_path}")
                continue
            for img_file in os.listdir(label_path):
                img_lst.append(os.path.join(label_path, img_file))
                label_lst.append(label)
        df = pd.DataFrame({'img': img_lst, 'label': label_lst})
        logger.info(f"DataFrame created with {len(df)} entries")
        return df

    def resave(self):
        """
        Resaves images as JPEGs with specified quality and updates DataFrame
        """
        logger.info(f"Starting resave process with quality: {self.params.quality}")
        resaved_path = self.config.resaved_path
        os.makedirs(resaved_path, exist_ok=True)
        skipped_files = {'non_image': 0, 'error': 0}
        for index, row in self.df.iterrows():
            img_file = row['img']
            if img_file.lower().endswith(tuple(self.params.valid_extensions)):
                try:
                    img = Image.open(img_file).convert('RGB')
                    img_file_name = os.path.basename(img_file)
                    resaved_name = os.path.splitext(img_file_name)[0] + '_resaved.jpg'
                    save_path = os.path.join(self.config.resaved_path, resaved_name)
                    img.save(save_path, 'JPEG', quality=self.params.quality, optimize=True)
                    img.close()  # Close image to prevent WinError 5
                    logger.debug(f"Resaved image: {save_path}")
                except UnidentifiedImageError as e:
                    logger.error(f"Cannot identify image file {img_file}: {e}")
                    skipped_files['error'] += 1
                except Exception as e:
                    logger.error(f"Error resaving {img_file}: {e}")
                    skipped_files['error'] += 1
            else:
                skipped_files['non_image'] += 1
        self.df['img_resaved'] = self.df['img'].apply(
            lambda x: os.path.join(self.config.resaved_path, os.path.splitext(os.path.basename(x))[0] + '_resaved.jpg')
        )
        logger.info("Resaved image paths added to DataFrame")
        if skipped_files['non_image'] > 0:
            print(f"Skipped {skipped_files['non_image']} files in resave due to non-image file extensions")
        if skipped_files['error'] > 0:
            print(f"Skipped {skipped_files['error']} files in resave due to errors (e.g., unidentified image)")

    def img_difference(self, org, resaved):
        """
        Computes the enhanced difference between original and resaved images
        """
        logger.debug(f"Computing difference between {org} and {resaved}")
        try:
            org_img = Image.open(org).convert('RGB')
            resaved_img = Image.open(resaved).convert('RGB')
            diff = ImageChops.difference(org_img, resaved_img)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = self.params.normalization_scale / max_diff
            diff = ImageEnhance.Brightness(diff).enhance(scale)
            enhancer = ImageEnhance.Sharpness(diff)
            diff = enhancer.enhance(self.params.sharpness_factor)
            org_img.close()  # Close image to prevent WinError 5
            resaved_img.close()  # Close image to prevent WinError 5
            logger.debug(f"Difference computed for {org}")
            return diff
        except Exception as e:
            logger.error(f"Error processing {org} and {resaved}: {e}")
            return None

    def prep_dataset(self):
        """
        Prepares dataset by computing image differences and creating feature arrays
        """
        logger.info("Preparing dataset")
        valid_extensions = tuple(self.params.valid_extensions)
        skipped_files = {'invalid_extension': 0, 'failed_diff': 0, 'error': 0}
        # Pre-allocate arrays
        n_samples = len(self.df)
        feature_size = self.params.image_size[0] * self.params.image_size[1] * 3
        X = np.empty((n_samples, feature_size), dtype=np.float32)
        y = np.empty((n_samples, 2), dtype=np.float32)
        idx = 0
        for index, row in self.df.iterrows():
            if (row['img'].lower().endswith(valid_extensions) and 
                row['img_resaved'].lower().endswith(valid_extensions)):
                try:
                    diff = self.img_difference(row['img'], row['img_resaved'])
                    if diff is not None:
                        x = diff.resize(tuple(self.params.image_size))
                        X[idx] = np.array(x, dtype=np.float32).flatten() / self.params.normalization_scale
                        y[idx] = [1, 0] if row['label'] == 'Au' else [0, 1]
                        logger.debug(f"Processed image: {row['img']}")
                        idx += 1
                    else:
                        skipped_files['failed_diff'] += 1
                except (UnidentifiedImageError, FileNotFoundError) as e:
                    logger.error(f"Skipping file {row['img']} due to error: {e}")
                    skipped_files['error'] += 1
            else:
                skipped_files['invalid_extension'] += 1
        # Trim arrays to actual size
        X = X[:idx]
        y = y[:idx]
        logger.info(f"Dataset prepared with {len(X)} samples")
        if skipped_files['invalid_extension'] > 0:
            print(f"Skipped {skipped_files['invalid_extension']} files in prep_dataset due to invalid file extensions")
        if skipped_files['failed_diff'] > 0:
            print(f"Skipped {skipped_files['failed_diff']} files in prep_dataset due to failed difference computation")
        if skipped_files['error'] > 0:
            print(f"Skipped {skipped_files['error']} files in prep_dataset due to errors (e.g., unidentified image or file not found)")
        return X, y

    def delete_resaved(self):
        """
        Deletes the resaved directory and its contents
        """
        logger.info(f"Deleting resaved directory: {self.config.resaved_path}")
        try:
            shutil.rmtree(self.config.resaved_path)
            logger.info(f"Successfully deleted resaved directory: {self.config.resaved_path}")
        except Exception as e:
            logger.error(f"Error deleting resaved directory {self.config.resaved_path}: {e}")

    def save_dataset(self):
        """
        Saves the processed dataset as joblib files and deletes resaved directory
        """
        logger.info("Starting dataset saving process")
        self.resave()
        pickle_save = self.config.pickle_save
        os.makedirs(pickle_save, exist_ok=True)
        X, y = self.prep_dataset()
        x_path = os.path.join(self.config.pickle_save, 'X_90.joblib')
        y_path = os.path.join(self.config.pickle_save, 'y.joblib')
        joblib.dump(X, x_path)
        joblib.dump(y, y_path)
        logger.info(f"Dataset saved to {x_path} and {y_path}")
        self.delete_resaved()
        return X, y