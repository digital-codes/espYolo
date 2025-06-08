import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import json
import math

import layoutUtilsRect as layout

import sys 



# CONFIG
MODEL_PATH = "best_model_regions_rect.keras"
GRID = (7,5)
IMAGE_SIZE = (176,144) # QCIF
REG_ITEMS = 4

OUTPUT_DIR = "verify"
THRESHOLD = 0.4  # Prediction confidence threshold
INPUT_PATH = "./voc"

if len(sys.argv) > 1:
    modelSource = sys.argv[1]
else:
    modelSource = MODEL_PATH

INPUT_PATH = "./voc"
if len(sys.argv) > 2:
    imgSource = sys.argv[2]
else:
    imgSource = INPUT_PATH



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
                                                         len(classes.keys()),REG_ITEMS, 6)
                

            yield image, labelVector # (bboxes, labels)
                

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(output_size), dtype=tf.float32)
        )
    )
    return ds


with open(os.path.join(imgSource, "label_map.json"), "r") as f:
    classes = json.load(f)


# === Load model
model = tf.keras.models.load_model(modelSource, compile=False)

cells = layout.define_cells(IMAGE_SIZE, GRID)
regions = layout.define_regions(cells,GRID)
NUM_REGIONS = len(regions)
print(f"Defined {len(regions)} regions for image size {IMAGE_SIZE}.")
item = "class,prob,x0,y1,x1,y1"
output_size = len(regions) * (len(item.split(","))) * REG_ITEMS 
print(f"Output vector size: {output_size}.")

# ds = load_dataset(image_dir, classes, cells, regions, GRID, output_size=output_size)



# === Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(imgSource, classes, cells, regions,GRID, output_size=output_size)


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
    
    items = layout.decode_label_vector(pred_vec, cells, regions, GRID, REG_ITEMS)

    if len(items) == 0:
        print(f"No items found in prediction vector for image {idx}. Skipping...")
        continue
    
                
    # Convert prediction vector to class and bounding boxes

    # Draw boxes
    fig = plt.figure(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)  # 1.6 * 100 = 160 pixels
    ax = fig.add_axes([0, 0, 1, 1])  # full canvas, no padding
    ax.imshow(img_uint8)
    ax.axis('off')

    for item in items:
        class_id = item[1]
        label = f"{list(classes.keys())[class_id]} ({item[0]:.2f})"
        rect = plt.Rectangle((item[2], item[3]), item[4] - item[2], item[5] - item[3],
                                linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        lblx = item[2] + 10 if item[2] < IMAGE_SIZE[0] - 50 else item[2] - 60
        lbly = item[3] + 10 if item[3] < IMAGE_SIZE[1] - 20 else item[3] - 20
        ax.text(lblx, lbly, label, color='lime', fontsize=6)


    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx:04d}.json")
    with open(save_path, "w") as f:
        # json.dump([item.tolist() for item in items], f)
        json.dump([item for item in items], f)

    #ax.axis('off')
    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx:04d}.png")
    #fig.set_size_inches(IMAGE_SIZE / fig.dpi, IMAGE_SIZE / fig.dpi)  # Match input image dimensions
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    

print(f"✅ Saved {idx+1} prediction images to '{OUTPUT_DIR}/'")
