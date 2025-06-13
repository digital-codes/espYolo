# Re-run necessary imports and code setup after kernel reset
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy

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
parser.add_argument("--label_dir","-l", type=str, help="Path to the directory containing label JSON files. Defaults to image_dir.")
parser.add_argument("--model","-m", type=str, required=True, help="Ouput model name")
parser.add_argument("--convert","-c", type=bool, default=False, help="Convert only (default: False)")
parser.add_argument("--alpha","-a", type=float, default=.5, help="ALpha fraction for Mobilenet(default: .5)")
parser.add_argument("--epochs","-e", type=int, default=30, help="Epochs (default: 30)")
parser.add_argument("--format","-f", type=str, default="qcif", help="Image format (qcif, qvga)")
parser.add_argument("--rgb","-r", type=bool, default=False, help="Image RGB mode")
parser.add_argument("--batch_size","-b", type=int, default=4, help="Batch size")
args = parser.parse_args()
args.label_dir = args.label_dir if args.label_dir else args.image_dir

COLORS = 3 if args.rgb else 1  # RGB or grayscale
INPUT_SHAPE = (240, 320, COLORS) if args.format == "qvga" else (144, 176, COLORS)  # HWC format
OUTPUT_GRID = (INPUT_SHAPE[0]//16,INPUT_SHAPE[1]//16) # (9, 11)
FINAL_CONV_SIZE = 3 if args.format != "qcif" else 1 
BATCH_SIZE = args.batch_size

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

# Stop training early if val_loss doesn't improve for 5 epochs
earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=20,         # number of epochs with no improvement before stopping
    restore_best_weights=True,
    verbose=1
)

# weights for loss function penalizing empty classes

def make_weighted_bce_loss(class_weights):
    """
    Returns a binary cross-entropy loss function that applies class weights.
    class_weights: list or tensor of shape (num_classes + 1,)
    """
    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        # Shape: (batch, grid_h, grid_w, num_classes+1)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)  # same shape
        weights = tf.reshape(class_weights, (1, 1, 1, -1))  # make broadcastable
        weighted_bce = bce * weights
        return tf.reduce_mean(weighted_bce)  # scalar
    return weighted_bce

weighted_loss = make_weighted_bce_loss([0.01] + [1.0] * NUM_CLASSES)  # downweight 'empty'


# Re-define helper functions
def load_sample(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    img_path = os.path.join(IMAGE_DIR, data["img"])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=COLORS)
    img = tf.image.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img = tf.cast(img, tf.float32) / 255.0

    bboxes = data["bboxes"]
    labels = data["labels"]
    return img.numpy(), bboxes, labels

def make_label_grid(bboxes, labels, img_shape, grid_shape, num_classes):
    grid_h, grid_w = grid_shape  # HW(c) mode here
    img_h, img_w = img_shape  # HW(c) mode here
    # print(f"Creating label grid of shape {grid_shape} for image shape {img_shape} with {num_classes} classes")
    label = np.zeros((grid_h, grid_w, num_classes + 1))
    label[..., 0] = 1.0
    classSums = np.zeros((num_classes + 1))  # For debugging
    cell_h = img_h / grid_h
    cell_w = img_w / grid_w
    cell_area = cell_h * cell_w
    for (bbox, cls) in zip(bboxes, labels):
        if cls >= NUM_TYPES:
            raise ValueError(f"Class index {cls} exceeds number of objects {NUM_TYPES}.")
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx = int(cx / cell_w)
        gy = int(cy / cell_h)
        if gx >= grid_w or gy >= grid_h:
            raise ValueError(f"Grid cell ({gy}, {gx}) out of bounds for grid shape {grid_shape} with image shape {img_shape}.")
        size = NUM_SIZES # init with max
        if NUM_SIZES > 0:
            # Calculate size based on bbox area
            bbox_area = (x2 - x1) * (y2 - y1)
            for s in range(NUM_SIZES):
                if bbox_area < (cell_area * OBJ_SIZES[s]):
                    size = s
                    break
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            emptyOffs = 1
            #print(f"Assigning class {cls  + (size * NUM_TYPES) + emptyOffs} of size {size} to grid cell ({gy}, {gx})")
            label[gy, gx, cls + (size * NUM_TYPES) + emptyOffs] = 1.0
            label[gy, gx, 0] = 0
            classSums[cls + (size * NUM_TYPES) + emptyOffs] += 1.0
    return label, classSums

def data_generator(label_dir):
    files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith("labels.json")]
    for json_path in files:
        img, bboxes, labels = load_sample(json_path)
        label_grid,class_sums = make_label_grid(bboxes, labels, INPUT_SHAPE[:2], OUTPUT_GRID, NUM_CLASSES)
        #print(f"Loaded {json_path} with {len(bboxes)} bboxes, class sums: {class_sums}")
        #print(list(label_grid))
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


def build_mobilenetv2_fomo(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, alpha=ALPHA):
    print(f"Building MobileNetV2 FOMO model with input shape {input_shape}, num_classes {num_classes}, alpha {alpha}")
    class_channels = num_classes + 1
    input_layer = tf.keras.Input(shape=input_shape)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, alpha=alpha, weights=None, input_tensor=input_layer)
    x = base.get_layer("block_13_expand_relu").output
    print(f"Base layer output shape: {x.shape}")

    # Dropout after feature extraction
    x = tf.keras.layers.Dropout(.3, name="dropout_features")(x)

    if False:
        # Optional conv + activation
        if FINAL_CONV_SIZE == 1:
            x = tf.keras.layers.Conv2D(OUTPUT_GRID[0] * OUTPUT_GRID[1] * class_channels, kernel_size=1, use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            #x = tf.keras.layers.ReLU(max_value=6)(x)
            x = tf.keras.layers.ReLU()(x)
        else:
            x = tf.keras.layers.Conv2D(128, kernel_size=FINAL_CONV_SIZE, padding='same', dilation_rate=2)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU(max_value=6)(x)

        # Another dropout before final classifier head
        x = tf.keras.layers.Dropout(.2, name="dropout_classhead")(x)

    print(f"Layer output shape: {x.shape}")
    class_out = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid", name="class_output")(x)
    print(f"Output layer output shape: {class_out.shape}")
    return tf.keras.Model(inputs=input_layer, outputs=class_out)

def build_custom_fomo(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, alpha=ALPHA):
    print(f"Building custom FOMO model with input shape {input_shape}, num_classes {num_classes}, alpha {alpha}")
    class_channels = num_classes + 1
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    filterSizes = [64, 64, 32, class_channels]  # 128, 128]  # leave out 128, 128
    for f, filters in enumerate(filterSizes):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        if f < len(filterSizes) - 1:
            x = tf.keras.layers.Dropout(0.2)(x)
        print(f"Layer {f}: {filters} filters, output shape: {x.shape}")

    # Print layer size for debugging
    x = tf.keras.layers.Flatten()(x)
    print(f"Layer output shape: {x.shape}")

    # x = tf.keras.layers.Dense(filterSizes[-1], activation="relu")(x)

    #x = tf.keras.layers.Dense(OUTPUT_GRID[0] * OUTPUT_GRID[1] * class_channels, activation="sigmoid", name="class_output")(x)
    x = tf.keras.layers.Reshape((OUTPUT_GRID[0], OUTPUT_GRID[1], class_channels))(x)
    outputs = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid", name="class_output")(x)

    #outputs = tf.keras.layers.Dense(class_channels, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)



# Prepare and train the model
full_ds = get_tf_dataset(LABEL_DIR)
train_size = int(0.8 * len(list(full_ds)))
train_ds = full_ds.take(train_size)
val_ds = full_ds.skip(train_size)

if args.convert == False:

    #model = build_mobilenetv2_fomo()
    model = build_custom_fomo()
    #model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=Adam(1e-3), loss=weighted_loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb,earlystop_cb])

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


