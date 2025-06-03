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
MODEL_PATH = "best_model_regions_softmax.keras"
IMAGE_SIZE = 160
OUTPUT_DIR = "verify"
THRESHOLD = 0.4  # Prediction confidence threshold

if len(sys.argv) > 1:
    modelSource = sys.argv[1]
else:
    modelSource = MODEL_PATH

INPUT_PATH = "./voc"
if len(sys.argv) > 2:
    imgSource = sys.argv[2]
else:
    imgSource = INPUT_PATH


GRID = 5
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

            labelVector = layout.convert_to_softmax_labels(labelVector, len(classes.keys()), len(regions))
            yield image, labelVector # (bboxes, labels)
                

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(len(regions)), dtype=tf.float32)
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
print("Classes: ", len(classes))



# === Run and visualize
for idx, (img, lbl) in enumerate(dataset):
    img_np = img.numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)

    # Ground thruth
    gt_vec = np.array(lbl)
    # print("GT vector:",gt_vec)
    ground_truth = []
    for region_id in range(NUM_REGIONS):
            class_id = gt_vec[region_id]
            if class_id > 0.5:  # Only consider regions with a class
                gt = {"region":region_id, "class":class_id - 1, "conf":1}
                print(f"GT: Region {gt["region"]}: class {gt["class"]}, confidence {gt["conf"]:.2f}")
                ground_truth.append(gt)

    # Predict
    pred_vec = model.predict(img[None, ...])[0]
    #print(f"Prediction vector for image {idx}: {pred_vec.shape}")
    # Get class IDs (0 = background, 1 = class 0, etc.)
    class_ids = np.argmax(pred_vec, axis=-1)     # shape: (36,)
    confidences = np.max(pred_vec, axis=-1)      # shape: (36,)
    detections = []
    for region_id, (class_id, conf) in enumerate(zip(class_ids, confidences)):
        if class_id != 0:
            detection = {"region":region_id, "class":class_id - 1, "conf":conf}
            print(f"Predict: Region {detection["region"]}: class {detection["class"]}, confidence {detection["conf"]:.2f}")
            detections.append(detection)    

    # Draw boxes
    fig, ax = plt.subplots(1)
    ax.imshow(img_uint8)


    for item in ground_truth:
        # print(f"  Found GT for class {class_id} in region {region_id},{regions[region_id]}")
        region_id = regions[item["region"]]
        class_id = item["class"].astype(int)
        (sc, ec) = region_id
        x1 = cells[sc][0]
        y1 = cells[sc][1]
        x2 = cells[ec][2]
        y2 = cells[ec][3]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor='red', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y2 + 3, f"{list(classes.keys())[class_id]} [GT]", fontsize=8, color='red')


    for item in detections:
        region_id = regions[item["region"]]
        class_id = item["class"].astype(int)
        conf = item["conf"]
        if conf < THRESHOLD:
            continue
        (sc, ec) = region_id
        x1 = cells[sc][0]
        y1 = cells[sc][1]
        x2 = cells[ec][2]
        y2 = cells[ec][3]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor='green', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f"{list(classes.keys())[class_id]} [{conf:0.2f}]", fontsize=8, color='green')



    ax.axis('off')
    save_path = os.path.join(OUTPUT_DIR, f"pred_{idx:03d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


print(f"✅ Saved {idx+1} prediction images to '{OUTPUT_DIR}/'")
