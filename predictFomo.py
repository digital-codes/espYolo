# Re-run necessary imports and code setup after kernel reset
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

import argparse 

# Constants
import fomodefs as fomo
OBJ_SIZES = fomo.OBJ_SIZES
NUM_SIZES = len(OBJ_SIZES)  # Number of sizes
NUM_TYPES = fomo.NUM_TYPES
NUM_CLASSES = NUM_TYPES * (NUM_SIZES + 1) # 5 classes, 3 sizes



# Argument parsing
parser = argparse.ArgumentParser(description="Train FOMO model and export TFLite quantized model.")
parser.add_argument("--image_dir","-i", type=str, required=True, help="Path to the directory containing images.")
parser.add_argument("--output_dir","-o", type=str, help="Path to the directory for results.")
parser.add_argument("--model","-m", type=str, required=True, help="Ouput model name")
parser.add_argument("--rgb","-r", type=bool, default=False, help="Image RGB mode")
parser.add_argument("--format","-f", type=str, default="qcif", help="Image format (qcif, qvga)")
args = parser.parse_args()

# Set directories
IMAGE_DIR = args.image_dir
COLORS = 3 if args.rgb else 1  # RGB or grayscale
INPUT_SHAPE = (240, 320, COLORS) if args.format == "qvga" else (144, 176, COLORS)  # HWC format


# load best model for quantization
model = tf.keras.models.load_model(f"best_{args.model}.keras", compile=False)
# Get model output shape
output_shape = model.output_shape
print("Model output shape:", output_shape)

def load_sample(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=COLORS)
    img = tf.image.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


dataFiles = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith("labels.json")]
imgFiles = []
for d in dataFiles:
    with open(d, "r") as f:
        data = json.load(f)
    imgFiles.append(os.path.join(IMAGE_DIR, data["img"]))

# === Run and visualize
for idx,img in enumerate(imgFiles[:100]):
    print("Processing image:", img)
    image = load_sample(img)
    image_ = Image.fromarray((image[..., 0] * 255).astype(np.uint8)) if COLORS == 1 else Image.fromarray((image * 255).astype(np.uint8))
    width = image_.width
    height = image_.height
    print("Img width,height:", width,height)
    print("Img tensor shape:", image.shape)
    # Predict
    pred_vec = model.predict(image[None, ...])[0]
    pred_shape = pred_vec.shape
    print("Pred shape:", pred_shape)
    print("Total activation sum:", np.sum(pred_vec))
    
    save_path = os.path.join(args.output_dir, f"predvec_{idx:04d}.json")
    with open(save_path, "w") as f:
        json.dump(pred_vec.tolist(), f)
    save_path = os.path.join(args.output_dir, f"predvec_{idx:04d}.png")
    
    # Draw rectangles on the image based on predictions
    draw = ImageDraw.Draw(image_)
    #image_.show()
    detected = False
    for r in range(pred_shape[0]):
        for c in range(pred_shape[1]):
            # scale down empty prediction
            # pred_vec[r][c][0] *= .1
            max_cls = np.argmax(pred_vec[r][c])
            if max_cls > 0:  # Only draw if there's a class prediction
                #print("Max class at grid cell ({},{}): {}".format(r, c, max_cls))
                confidence = pred_vec[r][c][max_cls]
                if confidence > 0.5:  # Threshold for drawing
                    detected = True
                    cls = (max_cls - 1) % NUM_TYPES
                    sz = (max_cls - 1) // NUM_TYPES + 1
                    col = "red" if sz == 1 else "blue" if sz == 2 else "green"
                    x1 = c * (width // pred_shape[1])
                    y1 = r * (height // pred_shape[0])
                    x2 = x1 + (width // pred_shape[1])
                    y2 = y1 + (height // pred_shape[0])
                    draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                    draw.text((x1 + 5, y1 + 5), f"{cls:02d}-{sz}", fill="black")

    # Save the image with rectangles
    if detected:
        image_.save(save_path)
    