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
parser.add_argument("--type","-t", type=str, default="custom", help="Model format")
args = parser.parse_args()

os.makedirs(args.output_dir,exist_ok=True)

# Set directories
IMAGE_DIR = args.image_dir
COLORS = 3 if args.rgb else 1  # RGB or grayscale
INPUT_SHAPE = (240, 320, COLORS) if args.format == "qvga" else (144, 176, COLORS)  # HWC format


# load best model for quantization
model = tf.keras.models.load_model(f"best_{args.model}.keras", compile=False)
# Get model output shape
output_shape = model.output_shape
print("Model output shape:", output_shape)

# yuv stuff
def rgb_to_yuv422(image):
    """
    Convert RGB image tensor [H, W, 3] to YUV422 format [H, W*2]
    using YUYV packing (Y0 U Y1 V).
    """
    R = tf.cast(image[:, :, 0], tf.float32)
    G = tf.cast(image[:, :, 1], tf.float32)
    B = tf.cast(image[:, :, 2], tf.float32)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.331 * G + 0.5 * B + 128.0
    V = 0.5 * R - 0.419 * G - 0.081 * B + 128.0

    Y = tf.clip_by_value(Y, 0.0, 255.0)
    U = tf.clip_by_value(U, 0.0, 255.0)
    V = tf.clip_by_value(V, 0.0, 255.0)

    Y = tf.cast(Y, tf.uint8)
    U = tf.cast(U, tf.uint8)
    V = tf.cast(V, tf.uint8)

    # Interleave as YUYV (per 2 pixels)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    assert_op = tf.debugging.assert_equal(width % 2, 0, message="Width must be even for YUV422")

    with tf.control_dependencies([assert_op]):
        Y0 = Y[:, 0::2]
        Y1 = Y[:, 1::2]
        U_pair = (U[:, 0::2] + U[:, 1::2]) // 2
        V_pair = (V[:, 0::2] + V[:, 1::2]) // 2

        # Stack and interleave: [Y0, U, Y1, V]
        yuyv = tf.stack([Y0, U_pair, Y1, V_pair], axis=-1)
        yuyv = tf.reshape(yuyv, (height, width * 2))

    yuyv = tf.expand_dims(yuyv, axis=-1)  # shape becomes [240, 640, 1]
    return yuyv

import numpy as np

def yuv422_to_rgb(yuv: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert YUV422 (YUYV) buffer to RGB888 image.
    yuv: 2D array of shape (height, width * 2), dtype=uint8
    returns: RGB image (height, width, 3), dtype=uint8
    """
    assert yuv.shape == (height, width * 2)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(0, width, 2):
            i = x * 2
            y0 = yuv[y, i + 0].astype(np.int16)
            u  = yuv[y, i + 1].astype(np.int16) - 128
            y1 = yuv[y, i + 2].astype(np.int16)
            v  = yuv[y, i + 3].astype(np.int16) - 128

            def yuv_to_rgb_pixel(y_val, u, v):
                c = y_val - 16
                r = (298 * c + 409 * v + 128) >> 8
                g = (298 * c - 100 * u - 208 * v + 128) >> 8
                b = (298 * c + 516 * u + 128) >> 8
                return np.clip([r, g, b], 0, 255)

            rgb[y, x + 0] = yuv_to_rgb_pixel(y0, u, v)
            rgb[y, x + 1] = yuv_to_rgb_pixel(y1, u, v)

    return rgb


# Re-define helper functions
def load_sample(img_path):
    img = tf.io.read_file(img_path)
    if args.type.endswith("_yuv"):
        img = tf.image.decode_png(img, channels=3)  # Decode as RGB
        #print(f"Initial image shape: {img.shape}")
        img = rgb_to_yuv422(img)
        #print(f"Final image shape: {img.shape}")
        img = tf.image.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]*2))
    else:        
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
    if args.type.endswith("_yuv"):
        #image_ = rgb_to_yuv422(image)
        image_ = Image.fromarray((image[..., 0] * 255).astype(np.uint8))
    else:
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
    #if args.type.endswith("_yuv"):
    #    image_ = Image.fromarray(yuv422_to_rgb(image[..., 0].numpy(), width, height)) 
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
    