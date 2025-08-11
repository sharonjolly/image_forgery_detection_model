from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from PIL import Image
import io
import base64
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Client app wrapper
class ClientApp:
    def __init__(self):
        self.forgerydetection = PredictionPipeline()

# Home route
@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        # Ensure file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Check allowed file types
        if file.mimetype not in ["image/png", "image/jpeg", "image/jpg"]:
            return jsonify({"error": "Unsupported file type. Please upload PNG or JPEG."}), 400
        
        # Read and process image
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # Model prediction
        result = clApp.forgerydetection.predict(img)
        details = clApp.forgerydetection.get_image_details(img, len(img_data))
        
        # Convert image to Base64
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG")
        image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
        
        # Response payload
        response = {
            "prediction": result,
            "format": details["format"],
            "size": details["size"],
            "file_size": details["file_size"],
            "image_base64": image_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Main entry point
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)  # AWS
