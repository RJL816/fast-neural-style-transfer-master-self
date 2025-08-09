# import os
# import argparse
#
# from PIL import Image
# from flask import Flask, render_template, request, jsonify, send_file
#
# import torch
# import tempfile
# from inference import InferenceProcess
#
# import uuid
#
# app = Flask(__name__)
#
# MODEL_LIST = {
#     "candy": "../weights/candy.onnx",
#     "mosaic": "../weights/mosaic.onnx",
#     "rain-princess": "../weights/rain-princess.onnx",
#     "udnie": "../weights/udnie.onnx",
#     "oil_style": "../weights/oil_style.onnx",
#     "oil_style01": "../weights/oil_style01.onnx",
#     "starry_night": "../weights/starry_night",
# }
#
# # Define command-line arguments
# parser = argparse.ArgumentParser(description="Deployment Arguments")
# parser.add_argument("--port", type=int, default=5000, help="Port number to run the server on")
# parser.add_argument(
#     "--model",
#     type=str,
#     default="mosaic", help="Model name 'candy', 'mosaic', 'rain-princess', 'udnie'")
# args = parser.parse_args()
#
#
# def load_model(model_name: str):
#     assert model_name in MODEL_LIST, f"Model `{model_name}` is not in Model List: {MODEL_LIST.keys()}"
#
#     inference_instance = InferenceProcess(MODEL_LIST[model_name])
#
#     return inference_instance
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'})
#
#     # Read the image file
#     image = request.files['image']
#     # Load the model
#     inference_instance = load_model(args.model)
#
#     filename = image.name
#     image = Image.open(image).convert('RGB')
#
#     # Save the output image to a temporary file
#     temp_dir = tempfile.gettempdir()
#     image_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.jpg")
#
#     # Perform inference
#     with torch.no_grad():
#         output = inference_instance(image)
#
#     # Convert the output tensor to an image and save
#     save_image(image_path, output[0])
#
#     # Return the path to the output image (temporary file path)
#     return jsonify({'output_image_path': image_path})
#
#
# @app.route('/get_image/<image_name>')
# def get_image(image_name):
#     temp_dir = tempfile.gettempdir()
#     image_path = os.path.join(temp_dir, image_name)
#     return send_file(image_path, mimetype='image/png')
#
#
# def save_image(image_path, data):
#     img = data.clone().clamp(0, 255).numpy()
#     img = img.transpose(1, 2, 0).astype("uint8")
#     img = Image.fromarray(img)
#     img.save(image_path)
#
#
# if __name__ == '__main__':
#     app.run(port=args.port, debug=True)
import os
import argparse
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
import torch
import tempfile
from inference import InferenceProcess
import uuid
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

MODEL_LIST = {
    "candy": "../weights/candy.onnx",
    "mosaic": "../weights/mosaic.onnx",
    "rain-princess": "../weights/rain-princess.onnx",
    "udnie": "../weights/udnie.onnx",
    "oil_style": "../weights/oil_style.onnx",
    "oil_style01": "../weights/oil_style01.onnx",
    "starry_night": "../weights/starry_night.onnx",
}

# 模型与期望尺寸的映射
MODEL_SIZES = {
    "candy": (1080, 1080),
    "mosaic": (1080, 1080),
    "rain-princess": (1080, 1080),
    "udnie": (1080, 1080),
    "starry_night": (1707, 1280),
    "oil_style": (1707, 1280),
    "oil_style01": (1707, 1280),
}

parser = argparse.ArgumentParser(description="Deployment Arguments")
parser.add_argument("--port", type=int, default=5000, help="Port number to run the server on")
args = parser.parse_args()

model_cache = {}

def load_model(model_name: str):
    assert model_name in MODEL_LIST, f"Model `{model_name}` is not in Model List: {MODEL_LIST.keys()}"
    if model_name not in model_cache:
        model_cache[model_name] = InferenceProcess(MODEL_LIST[model_name], model_name)
    return model_cache[model_name]

@app.route('/')
def index():
    return render_template('index.html', models=list(MODEL_LIST.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    selected_model = request.form.get('model')
    if not selected_model or selected_model not in MODEL_LIST:
        return jsonify({'error': 'Invalid model selected'})

    image = request.files['image']
    inference_instance = load_model(selected_model)

    img = Image.open(image).convert('RGB')
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.jpg")

    with torch.no_grad():
        output = inference_instance(img)

    save_image(image_path, output[0])
    return jsonify({'output_image_path': image_path})

@app.route('/get_image/<image_name>')
def get_image(image_name):
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, image_name)
    return send_file(image_path, mimetype='image/png')

def save_image(image_path, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(image_path)

if __name__ == '__main__':
    app.run(port=args.port, debug=True)