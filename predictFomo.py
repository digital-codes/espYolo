# Re-run necessary imports and code setup after kernel reset
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

import argparse 

# Constants
INCLUDE_EMPTY = True
BATCH_SIZE = 4
NUM_SIZES = 3
NUM_OBJECTS = 5
NUM_CLASSES = NUM_OBJECTS * NUM_SIZES # 5 classes, 3 sizes

# Argument parsing
parser = argparse.ArgumentParser(description="Train FOMO model and export TFLite quantized model.")
parser.add_argument("--image_dir","-i", type=str, required=True, help="Path to the directory containing images.")
parser.add_argument("--output_dir","-o", type=str, help="Path to the directory for results.")
parser.add_argument("--model","-m", type=str, required=True, help="Ouput model name")
args = parser.parse_args()

# Set directories
IMAGE_DIR = args.image_dir


# load best model for quantization
model = tf.keras.models.load_model(f"best_{args.model}.keras", compile=False)
# Get model output shape
output_shape = model.output_shape
print("Model output shape:", output_shape)

dataFiles = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith("labels.json")]
imgFiles = []
for d in dataFiles:
    with open(d, "r") as f:
        data = json.load(f)
    imgFiles.append(os.path.join(IMAGE_DIR, data["img"]))

# === Run and visualize
for idx,img in enumerate(imgFiles[:100]):
    print("Processing image:", img)
    image_ = Image.open(img).convert("RGB")
    width, height = image_.size
    print("Image size:", width, height)
    #image = np.array(image) / 255.0
    image = np.array(image_).astype(np.float32) / 255.0  # normalize to [0, 1]

    print("Img shape:", image.shape)
    # Predict
    pred_vec = model.predict(image[None, ...])[0]
    print("Pred shape:", pred_vec.shape)
    
    save_path = os.path.join(args.output_dir, f"predvec_{idx:04d}.json")
    with open(save_path, "w") as f:
        json.dump(pred_vec.tolist(), f)
    save_path = os.path.join(args.output_dir, f"predvec_{idx:04d}.png")
    
    
    # Draw rectangles on the image based on predictions
    draw = ImageDraw.Draw(image_)
    detected = False
    for r in range(output_shape[1]):
        for c in range(output_shape[2]):
            max_cls = np.argmax(pred_vec[r][c])
            if max_cls > 0:  # Only draw if there's a class prediction
                # print("Max class at grid cell ({},{}): {}".format(r, c, max_cls))
                confidence = pred_vec[r][c][max_cls]
                if confidence > 0.5:  # Threshold for drawing
                    detected = True
                    cls = (max_cls - 1) % NUM_OBJECTS
                    sz = (max_cls - 1) // NUM_OBJECTS + 1
                    col = "red" if sz == 1 else "blue" if sz == 2 else "green"
                    x1 = c * (width // output_shape[2])
                    y1 = r * (height // output_shape[1])
                    x2 = x1 + (width // output_shape[2])
                    y2 = y1 + (height // output_shape[1])
                    draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                    draw.text((x1 + 5, y1 + 5), f"{cls:02d}-{sz}", fill="black")

    # Save the image with rectangles
    if detected:
        image_.save(save_path)
    