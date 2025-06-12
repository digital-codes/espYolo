# Re-run necessary imports and code setup after kernel reset
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import argparse 

# Constants
#INPUT_SHAPE = (144, 176, 3)  # HWC format
INPUT_SHAPE = (240,320, 3)  # HWC format
NUM_SIZES = 3
NUM_OBJECTS = 5
NUM_CLASSES = NUM_OBJECTS * NUM_SIZES # 5 classes, 3 sizes
INCLUDE_EMPTY = True
BATCH_SIZE = 16
# Alpha: 0.35 .. .5  going from .35 to .5 increases size by approx 50%. .35 tflite is around 220kB, .5 around 330kB
OUTPUT_GRID = (INPUT_SHAPE[0]//16,INPUT_SHAPE[1]//16) # (9, 11)
FINAL_CONV_CHANNELS = 128 # maybe use 64 for smaller images
FINAL_CONV_SIZE = 3 # 1 for smalle images. 

# Argument parsing
parser = argparse.ArgumentParser(description="Train FOMO model and export TFLite quantized model.")
parser.add_argument("--image_dir","-i", type=str, required=True, help="Path to the directory containing images.")
parser.add_argument("--label_dir","-l", type=str, help="Path to the directory containing label JSON files. Defaults to image_dir.")
parser.add_argument("--model","-m", type=str, required=True, help="Ouput model name")
parser.add_argument("--convert","-c", type=bool, default=False, help="Convert only (default: False)")
parser.add_argument("--alpha","-a", type=float, default=.5, help="ALpha fraction for Mobilenet(default: .5)")
parser.add_argument("--epochs","-e", type=int, default=30, help="Epochs (default: 30)")
args = parser.parse_args()
args.label_dir = args.label_dir if args.label_dir else args.image_dir

# Set directories
IMAGE_DIR = args.image_dir
LABEL_DIR = args.label_dir
ALPHA = args.alpha
EPOCHS = args.epochs

# Save the best model (lowest validation loss)
checkpoint_cb = ModelCheckpoint(
    filepath=f"best_{args.model}.keras",       # or .keras for TF >= 2.11+
    monitor="val_loss",             # or "val_accuracy"
    save_best_only=True,
    save_weights_only=False,        # True if you only want weights
    mode="min",                     # "min" for loss, "max" for accuracy
    verbose=1
)


# Re-define helper functions
def load_sample(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    img_path = os.path.join(IMAGE_DIR, data["img"])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img = tf.cast(img, tf.float32) / 255.0

    bboxes = data["bboxes"]
    labels = data["labels"]
    return img.numpy(), bboxes, labels

def make_label_grid(bboxes, labels, img_shape, grid_shape, num_classes, include_empty):
    grid_h, grid_w = grid_shape
    img_h, img_w = img_shape
    label = np.zeros((grid_h, grid_w, num_classes + 1 if include_empty else num_classes))
    if include_empty:
        label[..., 0] = 1.0
    cell_h = img_h / grid_h
    cell_w = img_w / grid_w
    cell_area = cell_h * cell_w
    for (bbox, cls) in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx = int(cx / cell_w)
        gy = int(cy / cell_h)
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_size = 1 if bbox_area < cell_area * 2.5 else 2 if bbox_area < 6 * cell_area else 3
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            if include_empty:
                label[gy, gx, 0] = 0
                label[gy, gx, cls + (bbox_size - 1) * num_classes // NUM_SIZES + 1] = 1
            else:
                label[gy, gx, cls + (bbox_size - 1) * num_classes // NUM_SIZES] = 1
    return label

def data_generator(label_dir):
    files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith("labels.json")]
    for json_path in files:
        img, bboxes, labels = load_sample(json_path)
        label_grid = make_label_grid(bboxes, labels, INPUT_SHAPE[:2], OUTPUT_GRID, NUM_CLASSES, INCLUDE_EMPTY)
        yield img, label_grid

def get_tf_dataset(label_dir):
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(label_dir),
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(OUTPUT_GRID[0], OUTPUT_GRID[1], NUM_CLASSES + 1), dtype=tf.float32)
        )
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_mobilenetv2_fomo(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, alpha=ALPHA, include_empty=INCLUDE_EMPTY):
    input_layer = tf.keras.Input(shape=input_shape)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, alpha=alpha, weights=None, input_tensor=input_layer)
    x = base.get_layer("block_13_expand_relu").output


    # Dropout after feature extraction
    x = tf.keras.layers.Dropout(.3, name="dropout_features")(x)

    # Optional conv + activation
    x = tf.keras.layers.Conv2D(FINAL_CONV_CHANNELS, kernel_size=FINAL_CONV_SIZE, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    # Another dropout before final classifier head
    x = tf.keras.layers.Dropout(.3, name="dropout_classhead")(x)

    class_channels = num_classes + 1 if include_empty else num_classes
    class_out = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid" if include_empty else "softmax", name="class_output")(x)
    return tf.keras.Model(inputs=input_layer, outputs=class_out)

# Prepare and train the model
train_ds = get_tf_dataset(LABEL_DIR)

if args.convert == False:
    val_ds = get_tf_dataset(LABEL_DIR)

    model = build_mobilenetv2_fomo()
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb])

    model_path = f"final_{args.model}.keras"
    model.save(model_path)
else:
    print("Skipping training, loading existing model...")

# load best model for quantization
model = tf.keras.models.load_model(f"best_{args.model}.keras", compile=False)

# TFLite Quantization
def representative_data_gen():
    for img, _ in train_ds.take(100):
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

# Save TFLite model
model_path = f"{args.model}.tflite"
with open(model_path, "wb") as f:
    f.write(tflite_quant_model)


