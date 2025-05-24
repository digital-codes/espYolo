import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, force CPU


import re
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use("Agg")  # Use Agg backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Target Metrics (for a well-trained model)
#Metric	Value (expected)	Notes
#Binary Accuracy	≥ 0.85 (ideal), ≥ 0.75 (good)	Per-label correctness
#Binary Crossentropy Loss	≤ 0.25 (ideal), ≤ 0.35 (usable)	For sigmoid outputs

# --- Label Mapping ---
label_map = {'bike': 0, 'car': 1, 'none': 2, 'person': 3}
class_names = list(label_map.keys())
quadrant_names = ['Q1', 'Q2', 'Q3', 'Q4', 'Center']

def visualize_quadrant_predictions(model, dataset, num_samples=4):
    for images, labels in dataset.unbatch().take(num_samples):
        preds = model(tf.expand_dims(images, 0), training=False)[0].numpy()

        class_pred = preds[:4]
        quad_pred = preds[4:]

        class_str = ", ".join(
            f"{class_names[i]} ({class_pred[i]:.2f})"
            for i in range(4) if class_pred[i] > 0.5
        )
        quad_str = ", ".join(
            f"{quadrant_names[i]} ({quad_pred[i]:.2f})"
            for i in range(4) if quad_pred[i] > 0.5
        )

        fig, ax = plt.subplots(1)
        ax.imshow(images.numpy())
        ax.set_title(f"Classes: {class_str}\nQuadrants: {quad_str}", fontsize=10)
        ax.axis('off')

        # Draw quadrant lines
        h, w = 160, 160
        ax.plot([80, 80], [0, 160], color='white', linestyle='--', linewidth=1)
        ax.plot([0, 160], [80, 80], color='white', linestyle='--', linewidth=1)
        plt.axis("off")
        plt.savefig(f"image_{class_str}_{quad_str}.png")
        #plt.show()


# Re-imports due to code execution state reset
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Evaluate the model and collect predictions
def evaluate_with_confusion(model, dataset, class_names, quadrant_names):
    y_true = []
    y_pred = []

    for x_batch, y_batch in dataset.unbatch():
        preds = model(tf.expand_dims(x_batch, axis=0), training=False)[0].numpy()
        y_true.append(y_batch.numpy())
        y_pred.append(preds > 0.5)  # Thresholding

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = class_names + quadrant_names
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        cm = conf_matrices[i]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(labels[i])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        for j in range(2):
            for k in range(2):
                ax.text(k, j, cm[j, k], ha="center", va="center", color="red")

    plt.tight_layout()
    plt.show()


checkpoint_cb = ModelCheckpoint(
    "best_model.keras",              # recommended Keras format
    monitor="val_loss",              # or "val_binary_accuracy"
    save_best_only=True,
    save_weights_only=False,
    mode="min",                      # use "max" if tracking accuracy
    verbose=1
)


# --- Parse TU Graz Annotation ---
def parse_tugraz_annotation(txt_path, label_map, tugraz_root):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    img_line = next(line for line in lines if "Image filename" in line)
    rel_path = re.search(r'"(.+?)"', img_line).group(1)
    img_path = os.path.join(tugraz_root, os.path.normpath(rel_path))

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB").resize((160, 160))
    orig_width, orig_height = 640, 480

    bboxes = []
    labels = []
    current_label = None

    for i, line in enumerate(lines):
        if "Original label for object" in line:
            match = re.search(r'"(.+?)"\s*:\s*"(.+?)"', line)
            if match:
                current_label = match.group(2)
        elif "Bounding box for object" in line and current_label:
            coords = re.findall(r"\((\d+),\s*(\d+)\)", line)
            if len(coords) == 2:
                xmin, ymin = map(int, coords[0])
                xmax, ymax = map(int, coords[1])
                xmin, xmax = xmin / orig_width, xmax / orig_width
                ymin, ymax = ymin / orig_height, ymax / orig_height
                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(label_map[current_label])
                current_label = None

    return np.array(img) / 255.0, (np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int32))

# --- Quadrant Logic ---
def bbox_to_quadrant(ymin, xmin, ymax, xmax):
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Check if box spans across center boundaries
    if xmin < 0.5 and xmax > 0.5 and ymin < 0.5 and ymax > 0.5:
        return 4  # Center

    if cy < 0.5 and cx < 0.5:
        return 0
    elif cy < 0.5 and cx >= 0.5:
        return 1
    elif cy >= 0.5 and cx < 0.5:
        return 2
    else:
        return 3
    
# --- Flatten Label Function ---
def flatten_with_quadrant(image, labels_tuple):
    bboxes, labels = labels_tuple
    class_vec = tf.reduce_max(tf.one_hot(labels, depth=4), axis=0)

    quadrant_ids = tf.map_fn(
        lambda box: tf.cast(bbox_to_quadrant(box[0], box[1], box[2], box[3]), tf.int32),
        bboxes,
        fn_output_signature=tf.int32
    )
    quadrant_vec = tf.reduce_max(tf.one_hot(quadrant_ids, depth=5), axis=0)

    combined = tf.concat([class_vec, quadrant_vec], axis=0)
    return image, combined

# --- Dataset Loader ---
def load_dataset(root_dir):
    ann_root = os.path.join(root_dir, "Annotations")
    txt_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(ann_root)
        for f in filenames if f.endswith(".txt")
    ]

    if not txt_files:
        raise ValueError("No annotation files found.")

    image_root = root_dir.split("/TUGraz")[0]
    
    def generator():
        for txt in txt_files:
            try:
                yield parse_tugraz_annotation(txt, label_map, image_root)
            except Exception as e:
                print(f"[WARN] Skipping {txt}: {e}")

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(160, 160, 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
    )
    return ds.map(flatten_with_quadrant, num_parallel_calls=tf.data.AUTOTUNE)

# --- Tiny Model ---
def build_quadrant_model_1(input_shape=(160, 160, 3), output_dim=9):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

def build_quadrant_model_2(input_shape=(160, 160, 3), num_classes=4, num_quadrants=5):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(8, 3, strides=2, activation='relu', padding='same')(inputs)  # 80x80x8
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(16, 1, activation='relu')(x)  # 80x80x16

    x = tf.keras.layers.MaxPooling2D()(x)  # 40x40
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, 1, activation='relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)  # ✅ Dropout before output layer

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes * num_quadrants, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)

# --- TinyissimoYOLO-like Model ---
def build_quadrant_model(input_shape=(160, 160, 3), grid_size=5, num_boxes=1, num_classes=4, num_quadrants=5):
    S = grid_size
    B = num_boxes
    C = num_classes
    Q = num_quadrants
    output_dim = S * S * (B * 5 + C + Q)

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    filterSizes = [16, 16, 32, 32, 64, 64, 128] # 128, 128]  # leave out 128, 128
    for filters in filterSizes: 
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    #filter_pairs = [(16, 16), (16, 32), (32, 64), (64, 64)] # , (128, 128)]
    #for f1, f2 in filter_pairs:
    #    x = tf.keras.layers.Conv2D(f1, 3, padding='same', activation='relu')(x)
    #    x = tf.keras.layers.Conv2D(f2, 3, padding='same', activation='relu')(x)
    #    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)


    x = tf.keras.layers.Flatten()(x)
    # dropout extra
    x = tf.keras.layers.Dropout(0.3)(x)  # ✅ Dropout before output layer

    x = tf.keras.layers.Dense(filterSizes[-1], activation='relu')(x)
    #x = tf.keras.layers.Dense(filter_pairs[-1][-1], activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes + num_quadrants, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)


# --- Main ---
def main():
    ds = load_dataset("./datasets/TUGraz")
    ds = ds.batch(4).prefetch(tf.data.AUTOTUNE)

    count = 0
    class_counter = np.zeros(4)
    for _, label in ds.unbatch():
        count += 1
        class_counter[:4] += label[:4].numpy()

    print(f"Total samples loaded: {count}")
    for idx, name in enumerate(class_names):
        print(f"Class '{name}': {int(class_counter[idx])} occurrences")

    if count == 0:
        raise ValueError("Dataset is empty. Check annotation files.")

    model = build_quadrant_model()
    visualize_quadrant_predictions(model, ds)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    def weighted_loss(y_true, y_pred):
        weights = tf.constant([2.0]*4 + [1.0]*5)  # upweight class part
        return tf.reduce_mean(weights * loss_fn(y_true, y_pred))


    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(
        optimizer='adam',
        loss=weighted_loss, # 'binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.fit(ds, epochs=200,    callbacks=[TqdmCallback(verbose=1),checkpoint_cb])
    print("\nEvaluating model:")
    model.evaluate(ds)
    model.save("model_5quadrant.h5")
    
    #evaluate_with_confusion(model, ds.batch(1), class_names, quadrant_names)
    
    visualize_quadrant_predictions(model, ds)

if __name__ == '__main__':
    main()
