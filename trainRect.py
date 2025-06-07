import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, force CPU

import json
import argparse
import sys

import re
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

#import matplotlib
#matplotlib.use("Agg")  # Use Agg backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Target Metrics (for a well-trained model)
#Metric	Value (expected)	Notes
#Binary Accuracy	≥ 0.85 (ideal), ≥ 0.75 (good)	Per-label correctness
#Binary Crossentropy Loss	≤ 0.25 (ideal), ≤ 0.35 (usable)	For sigmoid outputs

import layoutUtilsRect as layout

GRID = (7,5)
IMAGE_SIZE = (176,144) # QCIF
REG_ITEMS = 4


checkpoint_cb = ModelCheckpoint(
    "best_model_regions_wide.keras",              # recommended Keras format
    monitor="val_f1",            # <- use F1 score now
    mode="max",                  # <- maximize F1
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

lr_cb = ReduceLROnPlateau(
    monitor='val_f1',
    factor=0.5,
    patience=20,
    min_lr=1e-6,
    mode='max',
    verbose=1
)


stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1', patience=50, mode='max', restore_best_weights=True
)


# --- TinyissimoYOLO-like Model ---
def build_quadrant_model(input_shape, output_dim):

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # filterSizes = [16, 16, 32, 32, 64, 64, 128] # 128, 128]  # leave out 128, 128
    # filterSizes = [16, 32, 32, 64, 128 ] # 128, 128]  # leave out 128, 128
    filterSizes = [16, 16, 32, 64 ] # 128, 128]  # leave out 128, 128
    for f, filters in enumerate(filterSizes): 
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        if f < len(filterSizes) - 1:
            x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Flatten()(x)
    # dropout extra
    x = tf.keras.layers.Dropout(0.3)(x)  # ✅ Dropout before output layer
    x = tf.keras.layers.Dense(filterSizes[-1], activation='relu')(x)


    #x = tf.keras.layers.Dense(filter_pairs[-1][-1], activation='relu')(x)
    # outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)


    return tf.keras.Model(inputs, outputs)


def load_dataset(image_dir, classes, cells, regions, grid, output_size=None):
    label_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(image_dir)
        for f in filenames if f.endswith("labels.json")
    ]

    if not label_files:
        raise ValueError("No 'labels.json' files found in the provided directory.")

    def generator():
        for label_file in label_files:
            with open(label_file, "r") as f:
                data = json.load(f)
                
            image_path = os.path.join(image_dir, data["img"])
            if not os.path.exists(image_path):
                print(f"[WARN] Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
            image = np.array(image) / 255.0

            bboxes = np.array(data["bboxes"], dtype=np.float32) # read as float32 for further processing
            labels = np.array(data["labels"], dtype=np.int32)
            if len(bboxes) == 0 or len(labels) == 0:
                labelVector = np.zeros(output_size, dtype=np.float32)
            else:
                labelVector = layout.create_label_vector(cells, regions, grid, bboxes, labels, 
                                                         len(classes.keys()),output_size, REG_ITEMS)
                

            yield image, labelVector # (bboxes, labels)
                

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(output_size), dtype=tf.float32)
        )
    )
    return ds


def split_dataset(ds, train_size, val_size, test_size):
    total_size = train_size + val_size + test_size
    ds = ds.shuffle(buffer_size=total_size, reshuffle_each_iteration=False).cache()
    
    indexed_ds = ds.enumerate()

    train_ds = indexed_ds.filter(lambda i, _: i < train_size).map(lambda _, x: x)
    val_ds   = indexed_ds.filter(lambda i, _: (i >= train_size) & (i < train_size + val_size)).map(lambda _, x: x)
    test_ds  = indexed_ds.filter(lambda i, _: i >= train_size + val_size).map(lambda _, x: x)

    return train_ds, val_ds, test_ds


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + \
                 (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        return tf.reduce_mean(weight * ce)
    return loss

def sparse_f1_score(threshold=0.5):
    def f1(y_true, y_pred):
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred, axis=1)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=1)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1)
    return f1



# --- Main ---
def main():
    global regions
    parser = argparse.ArgumentParser(description="Train a quadrant prediction model.")
    parser.add_argument(
        "--image_dir", "-i", type=str, required=True,
        help="Path to the image root directory"
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    if not os.path.exists(image_dir):
        raise ValueError(f"Provided image directory does not exist: {image_dir}")

    with open(os.path.join(image_dir, "label_map.json"), "r") as f:
        classes = json.load(f)
    print(f"Loaded {len(classes)} classes from label_map.json")

    cells = layout.define_cells(IMAGE_SIZE, GRID)
    regions = layout.define_regions(cells,GRID)
    print(f"Defined {len(regions)} regions for image size {IMAGE_SIZE}.")
    item = "class,prob,x0,y1,x1,y1"
    output_size = len(regions) * (len(item.split(","))) * REG_ITEMS 
    print(f"Output vector size: {output_size}.")
    
    ds = load_dataset(image_dir, classes, cells, regions,GRID, output_size=output_size)
    print(f"Loaded dataset")
    
    if ds is None:
        print("No data loaded. Please check the image directory and label files.")
        print("Exiting...")
        sys.exit()
    else:
        print("Dataset loaded. Sample data:")
        for image, label in ds.shuffle(buffer_size=100).take(5):
            print(f"Image shape: {image.shape}, Label shape: {label.shape}")
            print(f"Label vector: {label.numpy()}")
        if ds is None:
            print("No data loaded. Please check the image directory and label files.")
            print("Exiting...")

    input_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    print(f"Input shape: {input_shape}, Output dimension: {output_size}")

    model = build_quadrant_model(input_shape = input_shape, output_dim = output_size)

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Start with 1e-3 (default), go higher or lower based on behavior
    # lr = 1e-3  # default
    # lr = 3e-4  # slower but safer
    # lr = 1e-4  # for fine-tuning or very stable training
    lr = 1e-3
    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(), # 'binary_crossentropy',
        metrics=[sparse_f1_score(threshold=0.2)]  # use lower threshold if needed
        # metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    # Split dataset into train, validation, and test sets
    dataset_size = len(list(ds))

    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size
    batch_size = 16
    steps = train_size // batch_size
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples.")

    # Split into 3 disjoint parts
    train_ds, val_ds, test_ds = split_dataset(ds, train_size, val_size, test_size)

    # Batch and prefetch
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2000,
        batch_size=batch_size,
        steps_per_epoch=steps,
        verbose=1,
        callbacks=[TqdmCallback(verbose=1), checkpoint_cb, lr_cb, stop_cb]
    )

    class_labels = list(classes.keys())

    model.save("final_model_regions_wide.keras")  # Native format

    print("\nEvaluating model:")
    test_steps = test_size // batch_size
    results = model.evaluate(test_ds, steps=test_steps)
    for metric, value in zip(model.metrics_names, results):
        print(f"{metric}: {value:.4f}")
    
    

if __name__ == '__main__':
    main()
