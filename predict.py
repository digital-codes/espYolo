import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import json
import math

import layoutUtils as layout

# CONFIG
MODEL_PATH = "best_model_regions.keras"
IMAGE_SIZE = 160
OUTPUT_DIR = "verify"
THRESHOLD = 0.35  # Prediction confidence threshold
MAX_IMAGES = 50

GRID = 3
INPUT_PATH = "./output"
IMAGE_SIZE = 160

def load_dataset(image_dir, classes, cells, regions):
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

            image = Image.open(image_path).convert("RGB") # .resize((imageSize, imageSize))
            image = np.array(image) / 255.0

            bboxes = np.array(data["bboxes"], dtype=np.float32) # read as float32 for further processing
            labels = np.array(data["labels"], dtype=np.int32)
            if len(bboxes) == 0 or len(labels) == 0:
                labelVector = np.zeros(len(regions) * len(classes.keys()), dtype=np.float32)
            else:
                labelVector = layout.create_label_vector(cells, regions, bboxes, labels, len(classes.keys()))

            yield image, labelVector # (bboxes, labels)
                

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(regions)*len(classes.keys())), dtype=tf.float32)
        )
    )
    return ds


with open(os.path.join(INPUT_PATH, "label_map.json"), "r") as f:
    classes = json.load(f)


# === Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

cells = layout.define_cells(IMAGE_SIZE, GRID)
regions = layout.define_regions(GRID)
NUM_REGIONS = len(regions)



# === Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(INPUT_PATH, classes, cells, regions)


class_counts = [0] * len(list(classes.keys()))
for _, label in dataset:
    for class_id in range(len(list(classes.keys()))):
        start = class_id * NUM_REGIONS
        end = start + NUM_REGIONS
        class_counts[class_id] += tf.reduce_sum(label[start:end]).numpy()
print("Label counts per class:", class_counts)



# === Run and visualize
for idx, (img, lbl) in enumerate(dataset):
    img_np = img.numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)

    # Predict
    pred_vec = model.predict(img[None, ...])[0]

    # Draw boxes
    fig, ax = plt.subplots(1)
    ax.imshow(img_uint8)

    for class_id in range(len(list(classes.keys()))):
        print(f"Processing class {class_id} ({list(classes.keys())[class_id]})")
        for region_id in range(NUM_REGIONS):
            index = class_id * NUM_REGIONS + region_id
            score = pred_vec[index]
            if score > THRESHOLD:
                (sc, ec) = regions[region_id]
                x1 = cells[sc][0]
                y1 = cells[sc][1]
                x2 = cells[ec][2]
                y2 = cells[ec][3]
                label = f"{list(classes.keys())[class_id]} ({score:.2f})"
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, label, color='lime', fontsize=8, weight='bold')
            # add ground thruth
            if lbl[index] > 0.5:
                # print(f"  Found GT for class {class_id} in region {region_id},{regions[region_id]}")
                (sc, ec) = regions[region_id]
                x1 = cells[sc][0]
                y1 = cells[sc][1]
                x2 = cells[ec][2]
                y2 = cells[ec][3]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor='red', linestyle='--', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y2 + 3, f"{list(classes.keys())[class_id]} [GT]", fontsize=8, color='red')


    ax.axis('off')
    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx:03d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"✅ Saved {idx+1} prediction images to '{OUTPUT_DIR}/'")
