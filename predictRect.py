import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import json
import math

import layoutUtils as layout

import sys 



# CONFIG
MODEL_PATH = "best_model_regions_wide.keras"
IMAGE_SIZE = 160
OUTPUT_DIR = "verify"
THRESHOLD = 0.4  # Prediction confidence threshold

GRID = 5
INPUT_PATH = "./voc"
IMAGE_SIZE = 160

if len(sys.argv) > 1:
    modelSource = sys.argv[1]
else:
    modelSource = MODEL_PATH

INPUT_PATH = "./voc"
if len(sys.argv) > 2:
    imgSource = sys.argv[2]
else:
    imgSource = INPUT_PATH


def load_dataset(image_dir, classes, cells, regions):
    label_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(image_dir)
        for f in filenames if f.endswith("labels.json")
    ]
    label_files.sort()
    
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

            print(f"Processing image: {image_path}")
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


with open(os.path.join(imgSource, "label_map.json"), "r") as f:
    classes = json.load(f)


# === Load model
model = tf.keras.models.load_model(modelSource, compile=False)

cells = layout.define_cells(IMAGE_SIZE, GRID)
regions = layout.define_regions(GRID)
NUM_REGIONS = len(regions)



# === Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(imgSource, classes, cells, regions)


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

    print("Img shape:", img_np.shape)
    # Predict
    pred_vec = model.predict(img[None, ...])[0]

    # Draw boxes
    fig = plt.figure(figsize=(IMAGE_SIZE/100, IMAGE_SIZE/100), dpi=100)  # 1.6 * 100 = 160 pixels
    ax = fig.add_axes([0, 0, 1, 1])  # full canvas, no padding
    ax.imshow(img_uint8)
    ax.axis('off')
    
    #fig, ax = plt.subplots(1, figsize=(IMAGE_SIZE / 100, IMAGE_SIZE / 100))
    #ax.set_xlim(0, IMAGE_SIZE)
    #ax.set_ylim(IMAGE_SIZE, 0)
    #ax.imshow(img_uint8)
    objects= []
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
                lblx = x1 + 10 if x1 < IMAGE_SIZE - 50 else x1 - 60
                lbly = y1 + 10 if y1 < IMAGE_SIZE - 20 else y1 - 20
                ax.text(lblx, lbly, label, color='lime', fontsize=6)
                objects.append((region_id, class_id))
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
                lblx = x1 + 15 if x1 < IMAGE_SIZE - 50 else x1 - 70
                lbly = y1 + 15 if y1 < IMAGE_SIZE - 20 else y1 - 30
                ax.text(lblx, lbly, f"{list(classes.keys())[class_id]} [GT]", fontsize=6, color='red')

    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx:04d}.json")
    with open(save_path, "w") as f:
        json.dump(objects, f)

    #ax.axis('off')
    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx:04d}.png")
    #fig.set_size_inches(IMAGE_SIZE / fig.dpi, IMAGE_SIZE / fig.dpi)  # Match input image dimensions
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    

print(f"✅ Saved {idx+1} prediction images to '{OUTPUT_DIR}/'")
