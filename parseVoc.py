import os

import re
import numpy as np
from PIL import Image
import json

import matplotlib
matplotlib.use("Agg")  # Use Agg backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Label Mapping ---
label_map = {'bike': 0, 'car': 1, 'person': 2}
class_names = list(label_map.keys())
# cell names is start x, start y, size
cell_names = [f"{x:01d}_{y:01d}_1" for y  in range(0,3) for x in range(0,3)]
cell_names += [f"{x:01d}_{y:01d}_2" for y  in range(0,2) for x in range(0,2)]
cell_names += [f"{x:01d}_{y:01d}_3" for y  in range(0,1) for x in range(0,1)]

imgSize = 160


# --- Parse TU Graz Annotation ---
def parse_tugraz_annotation(txt_path, label_map, tugraz_root):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    img_line = next(line for line in lines if "Image filename" in line)
    rel_path = re.search(r'"(.+?)"', img_line).group(1)
    img_path = os.path.join(tugraz_root, os.path.normpath(rel_path))

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # print(f"Processing image: {img_path}")
    img = Image.open(img_path)
    img_name = os.path.basename(img_path)
    orig_width, orig_height = img.width, img.height
    img = Image.open(img_path).convert("RGB").resize((imgSize, imgSize))

    bboxes = []
    labels = []
    current_label = None

    for i, line in enumerate(lines):
        if "Original label for object" in line:
            match = re.search(r'"(.+?)"\s*:\s*"(.+?)"', line)
            if match:
                current_label = match.group(2)
                # print(f"Found label: {current_label}")
                if current_label == "none":
                    break # empty image
        elif "Bounding box for object" in line and current_label:
            coords = re.findall(r"\((\d+),\s*(\d+)\)", line)
            # print(f"Found coordinates: {coords} for label: {current_label}")
            if len(coords) == 2:
                xmin, ymin = map(int, coords[0])
                xmax, ymax = map(int, coords[1])
                xmin, xmax = xmin / orig_width, xmax / orig_width
                ymin, ymax = ymin / orig_height, ymax / orig_height
                bboxes.append([int(ymin*imgSize), int(xmin*imgSize), int(ymax*imgSize), int(xmax*imgSize)])
                labels.append(label_map[current_label])
                current_label = None

    return img_name, np.array(img) / 255.0, (np.array(bboxes, dtype=np.int32), np.array(labels, dtype=np.int32))


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
    annotated_imgs = []
    for txt in txt_files:
        try:
            annotated_imgs.append(parse_tugraz_annotation(txt, label_map, image_root))
        except Exception as e:
            print(f"[WARN] Skipping {txt}: {e}")


    return annotated_imgs


# --- Main ---
def main():
    sources = load_dataset("./datasets/TUGraz")
    print(f"Total images: {len(sources)}")
    outDir = "./output"
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    print(f"Output directory: {outDir}")
    

    for i, (name, img, (bboxes, labels)) in enumerate(sources):
        if i < 10:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img)
            for bbox, label in zip(bboxes, labels):
                ymin, xmin, ymax, xmax = bbox
                rect = patches.Rectangle((xmin, ymin),
                                        (xmax - xmin),
                                        (ymax - ymin),
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, class_names[label], color='white', fontsize=12)
            plt.axis('off')
            plt.savefig(f"output_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"Processed image {i + 1}, {name}")
            print(f"Image shape: {img.shape}")
            print(f"Bounding boxes: {bboxes}")
            print(f"Labels: {labels}")
            print("===" * 10)
            print(f"\n") 
        # Save the image
        img_save_path = os.path.join(outDir, name)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(img_save_path)
        print(f"Saved image to {img_save_path}")
        # Save the labels
        label_save_path = os.path.join(outDir, f"{name.split('.')[0]}_labels.json")
        with open(label_save_path, 'w') as f:
            json.dump({"img":name, "bboxes": bboxes.tolist(), "labels": labels.tolist()}, f)
        print(f"Saved labels to {label_save_path}")
        print("=" * 40)

    # Save the label names map
    label_map_save_path = os.path.join(outDir, "label_map.json")
    with open(label_map_save_path, 'w') as f:
        json.dump(label_map, f)
    print(f"Saved label map to {label_map_save_path}")

if __name__ == '__main__':
    main()
