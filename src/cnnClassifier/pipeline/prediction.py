import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        try:
            model_path = os.path.join("model", "model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file is missing at: {model_path}")
            self.model = load_model(model_path)
            logger.info("Model successfully loaded and ready for use")
            logger.info(f"Input shape of the model: {self.model.input_shape}")
            logger.info(f"Output shape of the model: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Unable to load model: {str(e)}")
            raise Exception(f"Unable to load model: {str(e)}")

    def img_difference(self, org_img):
        """Generate a brightness-enhanced difference between the original and re-saved image."""
        try:
            img_io = io.BytesIO()
            org_img.save(img_io, 'JPEG', quality=90, optimize=True)
            resaved_img = Image.open(img_io).convert('RGB')
            
            diff = ImageChops.difference(org_img, resaved_img)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff
            diff = ImageEnhance.Brightness(diff).enhance(scale)
            diff = ImageEnhance.Sharpness(diff).enhance(2.0)
            logger.debug("Image difference calculated")
            return diff
        except Exception as e:
            logger.error(f"Image difference computation error: {str(e)}")
            raise Exception(f"Image difference computation error: {str(e)}")

    def preprocess_image(self, img):
        """Transform a PIL image into a normalized array for prediction."""
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected a PIL.Image object, but received {type(img)}")
        try:
            diff_img = self.img_difference(img)
            diff_img = diff_img.resize((128, 128))
            img_array = np.array(diff_img, dtype=np.float32) / 255.0
            img_array = img_array.reshape(-1, 128, 128, 3)
            logger.info(f"Image preprocessed to shape: {img_array.shape}")
            logger.info(f"Pixel values after preprocessing: min={img_array.min()}, max={img_array.max()}")
            return img_array
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise Exception(f"Preprocessing error: {str(e)}")

    def predict(self, img):
        """Predict whether an image is not forged or forged."""
        try:
            img_array = self.preprocess_image(img)
            pred = self.model.predict(img_array)[0]
            logger.info(f"Model raw output: {pred}")
            if len(pred) != 2:
                logger.error(f"Invalid output shape: {pred.shape}, expected 2 values.")
                raise ValueError(f"Model output length is {len(pred)}, expected 2.")
            confidence = pred[0] * 100 if pred[0] > pred[1] else pred[1] * 100
            return f"Not forged (confidence: {confidence:.2f}%)" if pred[0] > pred[1] else f"Forged (confidence: {confidence:.2f}%)"
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise Exception(f"Prediction error: {str(e)}")

    def get_image_details(self, img, file_size_bytes):
        """Return key information about the image."""
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected a PIL.Image object, but received {type(img)}")
        return {
            "format": img.format if img.format else "Unknown",
            "size": f"{img.size[0]} x {img.size[1]} pixels",
            "file_size": f"{file_size_bytes / 1024:.2f} KB"
        }
