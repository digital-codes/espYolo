import tensorflow as tf
from PIL import Image
import os
import numpy as np
import sys 

# === CONFIGURATION ===
MODEL_PATH = "best_model_regions.keras"
OUTPUT_TFLITE_PATH = "model_regions_int8.tflite"
REPRESENTATIVE_DATA_DIR = "voc/"  # folder with sample images
IMAGE_SIZE = (160, 160)  # (height, width)

if len(sys.argv) > 1:
    modelSource = sys.argv[1]
else:
    modelSource = MODEL_PATH

modelDest = "".join([modelSource.split(".keras")[0],".tflite"])


# === LOAD REPRESENTATIVE IMAGES ===
def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    img = np.array(img).astype(np.float32) / 255.0  # normalize to [0, 1]
    return img

def representative_data_gen():
    image_files = [os.path.join(REPRESENTATIVE_DATA_DIR, f) 
                   for f in os.listdir(REPRESENTATIVE_DATA_DIR) 
                   if f.lower().endswith((".jpg", ".png"))]
    for path in image_files[:100]:  # Use max 100 images
        img = load_image(path)
        yield [img[None, ...]]  # shape (1, 160, 160, 3)


# === LOAD MODEL ===
model = tf.keras.models.load_model(modelSource, compile=False)

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


