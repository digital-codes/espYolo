# Re-run necessary imports and code setup after kernel reset
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
import argparse 
from PIL import Image


# Constants
import fomodefs as fomo
OBJ_SIZES = fomo.OBJ_SIZES
NUM_SIZES = len(OBJ_SIZES)  # Number of sizes
NUM_TYPES = fomo.NUM_TYPES
NUM_CLASSES = NUM_TYPES * (NUM_SIZES + 1) # 5 classes, 3 sizes
LEVELS = fomo.LEVELS

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
parser.add_argument("--type","-t", type=str, default="custom", help="Model format")
args = parser.parse_args()
args.label_dir = args.label_dir if args.label_dir else args.image_dir

COLORS = 3 if args.rgb else 1  # RGB or grayscale
INPUT_SHAPE = (240, 320, COLORS) if args.format == "qvga" else (144, 176, COLORS)  # HWC format
if args.type == "mobilenet":
    OUTPUT_GRID = (round(INPUT_SHAPE[0]/(2**LEVELS)),round(INPUT_SHAPE[1]/(2**LEVELS))) 
else:
    OUTPUT_GRID = (int(INPUT_SHAPE[0]/(2**LEVELS)),int(INPUT_SHAPE[1]/(2**LEVELS))) 
print(f"Input shape: {INPUT_SHAPE}, Output grid: {OUTPUT_GRID}, Colors: {COLORS}")
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
    patience=10,         # number of epochs with no improvement before stopping
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


# Re-define helper functions
def load_sample(json_path):
    global COLORS, INPUT_SHAPE
    with open(json_path, "r") as f:
        data = json.load(f)
    img_path = os.path.join(IMAGE_DIR, data["img"])
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

def getFiles(label_dir):
    files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith("labels.json")]
    files = sorted(files)
    trainFiles = int(len(files) * 0.8)
    return files[:trainFiles],files[trainFiles:]

def data_generator(files):
    for json_path in files:
        img, bboxes, labels = load_sample(json_path)
        label_grid,class_sums = make_label_grid(bboxes, labels, INPUT_SHAPE[:2], OUTPUT_GRID, NUM_CLASSES)
        #print(f"Loaded {json_path} with {len(bboxes)} bboxes, class sums: {class_sums}")
        #print(list(label_grid))
        yield img, label_grid

def get_tf_dataset(files,yuv=False):
    columns = INPUT_SHAPE[1] if not yuv else INPUT_SHAPE[1] * 2
    colors = COLORS if not yuv else 1  # YUV422 has only one channel
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(files),
        output_signature=(
            tf.TensorSpec(shape=(INPUT_SHAPE[0],columns,colors), dtype=tf.float32),
            tf.TensorSpec(shape=(OUTPUT_GRID[0], OUTPUT_GRID[1], NUM_CLASSES + 1), dtype=tf.float32)
        )
    )
    return ds.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)


def build_mobilenetv2_fomo(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, alpha=ALPHA):
    print(f"Building MobileNetV2 FOMO model with input shape {input_shape}, num_classes {num_classes}, alpha {alpha}")
    class_channels = num_classes + 1
    input_layer = tf.keras.Input(shape=input_shape)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, alpha=alpha, weights=None, input_tensor=input_layer)
    if LEVELS == 5:
        x = base.output
    else:
        x = base.get_layer("block_13_expand_relu").output
    print(f"Base layer output shape: {x.shape}")

    # Dropout after feature extraction
    x = tf.keras.layers.Dropout(.2, name="dropout_features")(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', dilation_rate=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    print(f"Layer output shape: {x.shape}")
    class_out = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid", name="class_output")(x)
    print(f"Output layer output shape: {class_out.shape}")
    return tf.keras.Model(inputs=input_layer, outputs=class_out)

def build_custom_fomo(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    print(f"Building custom FOMO model with input shape {input_shape}, num_classes {num_classes}")
    class_channels = num_classes + 1
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if LEVELS == 5:
        filterSizes = [16,32, 64, 128, 4*class_channels]  # 128, 128]  # leave out 128, 128
    else:
        filterSizes = [16, 32, 64, 4*class_channels]
    for f, filters in enumerate(filterSizes):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        if f < len(filterSizes) - 1:
            x = tf.keras.layers.Dropout(0.2)(x)
        print(f"Layer {f}: {filters} filters, output shape: {x.shape}")

    outputs = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid", name="class_output")(x)
    print(f"Output Layer: {outputs.shape}")

    return tf.keras.Model(inputs, outputs)

def build_custom_fomo_yuv(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    print(f"Building custom FOMO model with input shape {input_shape}, num_classes {num_classes}")
    class_channels = num_classes + 1
    print("Input shape ", INPUT_SHAPE)
    # inputs = tf.keras.Input(shape=input_shape + (1,), name="yuv422_input")
    inputs = tf.keras.Input(shape=input_shape, name="yuv422_input")
    print(f"Input Layer: {inputs.shape}")

    # Step 1: Reshape to [H, W, 2] — group each pixel as 2 bytes
    reshaped = tf.keras.layers.Reshape((input_shape[0], input_shape[1] // 2, 2))(inputs)
    print(f"Reshaped Layer: {reshaped.shape}")

    # Step 2: Take only the first byte (Y) from each pair: [:, :, :, 0]
    def channel_selector_conv2d():
        #conv = tf.keras.layers.Conv2D(1, 1, use_bias=False, trainable=False)
        conv = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=1,
        padding='valid',
        use_bias=False,
        trainable=False
    )
        conv.build((None, None, None, 2))
        weights = tf.constant([[[[1.0], [0.0]]]], dtype=tf.float32)  # shape (1, 1, 2, 1)
        #weights = tf.constant([[[[1.0]], [[0.0]]]])  # select channel 0
        conv.set_weights([weights.numpy()])
        return conv
    x = channel_selector_conv2d()(reshaped)          # extract Y channel
    # x = tf.keras.layers.Lambda(lambda x: x[..., 0:1])(reshaped)  # shape: [H, W, 1]
    print(f"Y only Layer: {x.shape}")

    if LEVELS == 5:
        filterSizes = [16,32, 64, 128, 4*class_channels]  # 128, 128]  # leave out 128, 128
    else:
        filterSizes = [16, 32, 64, 4*class_channels]
    for f, filters in enumerate(filterSizes):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        if f < len(filterSizes) - 1:
            x = tf.keras.layers.Dropout(0.2)(x)
        print(f"Layer {f}: {filters} filters, output shape: {x.shape}")

    outputs = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid", name="class_output")(x)
    print(f"Output Layer: {outputs.shape}")

    return tf.keras.Model(inputs, outputs)


def build_custom_fomo2(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    print(f"Building custom FOMO model 2 with input shape {input_shape}, num_classes {num_classes}")
    class_channels = num_classes + 1
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    fexSizes = []
    for f in range(LEVELS):
        fexSizes.append(2**(4+f))
    for f, filters in enumerate(fexSizes):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        print(f"FEX Layer {f}: {filters} filters, output shape: {x.shape}")

    x = tf.keras.layers.Dropout(0.2)(x) 
    
    detSizes = fexSizes[-2::-1]
    print(f"DET Layer sizes: {detSizes}")
    for f in detSizes:
        if f == detSizes[-1]:
            kernel = 1
            x = tf.keras.layers.Conv2D(f, kernel, activation='relu')(x)
        else:
            kernel = 3
            x = tf.keras.layers.Conv2D(f, kernel, padding='same', activation='relu')(x)
            
        if f == detSizes[0]:
            x = tf.keras.layers.Dropout(0.2)(x) 

        print(f"DET Layer: {f} filters, {kernel}, output shape: {x.shape}")


    outputs = tf.keras.layers.Conv2D(class_channels, kernel_size=1, activation="sigmoid", name="class_output")(x)
    print(f"Output Layer: {outputs.shape}")

    return tf.keras.Model(inputs, outputs)
    

# Prepare and train the model
trainFiles, valFiles = getFiles(LABEL_DIR)
train_ds = get_tf_dataset(trainFiles,yuv=True if args.type.endswith("_yuv") else False)
val_ds = get_tf_dataset(valFiles,yuv=True if args.type.endswith("_yuv") else False)
trainSteps = len(trainFiles) // BATCH_SIZE
valSteps = len(valFiles) // BATCH_SIZE
print(f"Training on {len(trainFiles)} samples, validation on {len(valFiles)} samples.")
print("Val files from: ", valFiles[0])

if args.convert == False:

    if args.type == "mobilenet":
        model = build_mobilenetv2_fomo()
        model.compile(optimizer=Adam(1e-3), loss=weighted_loss, metrics=["accuracy"])
    elif args.type == "custom":
        model = build_custom_fomo()
        #model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer=Adam(1e-3), loss=weighted_loss, metrics=["accuracy"])
    elif args.type == "custom_yuv":
        model = build_custom_fomo_yuv(input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1]*2, 1), num_classes=NUM_CLASSES)
        #model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer=Adam(1e-3), loss=weighted_loss, metrics=["accuracy"])
    elif args.type == "custom2":
        model = build_custom_fomo2()
        #model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer=Adam(1e-3), loss=weighted_loss, metrics=["accuracy"])
    else:
        raise ValueError(f"Unknown model type: {args.type}")
    #model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=Adam(1e-3), loss=weighted_loss, metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, 
            epochs=EPOCHS, 
            steps_per_epoch=trainSteps,
            validation_steps=valSteps,
            callbacks=[checkpoint_cb,earlystop_cb])

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


