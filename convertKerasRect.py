import tensorflow as tf
from PIL import Image
import os
import numpy as np
import sys 
import argparse

# === CONFIGURATION ===
IMAGE_SIZE = (176, 144)  # (height, width)


parser = argparse.ArgumentParser(description="Convert model to tflite.")
parser.add_argument(
    "--image_dir",
    "-i",
    type=str,
    required=True,
    help="Path to the image root directory",
)
parser.add_argument(
    "--model",
    "-m",
    required=True,
    type=str,
    help="Input model",
)
args = parser.parse_args()


modelSource = args.model
imagePath = args.image_dir
modelDest = modelSource.split(".keras")[0] + ".tflite"

# === LOAD REPRESENTATIVE IMAGES ===
def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    img = np.array(img).astype(np.float32) / 255.0  # normalize to [0, 1]
    return img

def representative_data_gen():
    image_files = [os.path.join(imagePath, f) 
                   for f in os.listdir(imagePath) 
                   if f.lower().endswith((".jpg", ".png"))]
    for path in image_files[:100]:  # Use max 100 images
        img = load_image(path)
        yield [img[None, ...]]  # shape (1, 160, 160, 3)


# === LOAD MODEL ===
model = tf.keras.models.load_model(modelSource, compile=False)
# Get all layers and used operators
layer_details = []
for layer in model.layers:
    layer_details.append({
        "name": layer.name,
        "type": layer.__class__.__name__,
#        "input_shape": layer.input_shape,
#        "output_shape": layer.output_shape
    })

print("Model Layers:")
for detail in layer_details:
    print(detail)

# Create the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization for full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force int8 model for ESP32-S3 compatibility
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(modelDest, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model (int8) saved to {modelDest}")

interpreter = tf.lite.Interpreter(model_path=modelDest)
interpreter.allocate_tensors()
print("Input:", interpreter.get_input_details())
print("Output:", interpreter.get_output_details())

print("Input index:", interpreter.get_input_details()[0]["index"])
print("Input scale/offset:", interpreter.get_input_details()[0]["quantization"])

# Get used operators in the TFLite model
used_operators = interpreter.get_tensor_details()
print("\nUsed Operators:")
for op in used_operators:
    print(op)

