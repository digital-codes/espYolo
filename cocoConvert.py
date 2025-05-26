import json
import os
from PIL import Image
from collections import defaultdict

def coco_to_custom_cropped_format(
    coco_json_path,
    image_dir,
    cropped_image_dir,
    crop_size=(160, 160)
):
    os.makedirs(cropped_image_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Category mapping
    
    category_ids = {cat["id"]: cat["id"]-1 for _, cat in enumerate(coco["categories"])}
    category_names = {cat["name"]: cat["name"] for _, cat in enumerate(coco["categories"])}
    print(f"Found {len(category_ids)} categories in COCO JSON.")    
    print(f"Categories: {list(category_ids.values())}")
    print(f"Category names: {list(category_names.values())}")
    
    label_map = {c: category_ids[list(category_ids.keys())[i]] for i, c in enumerate(category_names)}
    print("Category mapping:",label_map)
    # Image metadata
    image_id_to_info = {
        img["id"]: {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        }
        for img in coco["images"]
    }

    # Group annotations per image
    image_annotations = defaultdict(lambda: {"bboxes": [], "labels": []})
    for ann in coco["annotations"]:
        image_annotations[ann["image_id"]]["raw_annots"] = image_annotations[ann["image_id"]].get("raw_annots", []) + [ann]

    results = []

    for image_id, ann_data in image_annotations.items():
        info = image_id_to_info[image_id]
        original_path = os.path.join(image_dir, info["file_name"])

        if not os.path.exists(original_path):
            print(f"Warning: image not found: {original_path}")
            continue

        img = Image.open(original_path)
        width, height = img.size
        crop_w, crop_h = crop_size

        # Compute crop box (center crop)
        cx, cy = width // 2, height // 2
        left = max(0, cx - crop_w // 2)
        top = max(0, cy - crop_h // 2)
        right = left + crop_w
        bottom = top + crop_h

        # Crop image
        cropped_img = img.crop((left, top, right, bottom))

        # Adjust annotations
        bboxes = []
        labels = []

        for ann in ann_data["raw_annots"]:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Check full inclusion in crop
            if x1 >= left and y1 >= top and x2 <= right and y2 <= bottom:
                # Adjust to crop
                adj_x1 = x1 - left
                adj_y1 = y1 - top
                adj_x2 = x2 - left
                adj_y2 = y2 - top

                bboxes.append([adj_x1, adj_y1, adj_x2, adj_y2])
                labels.append(category_ids[ann["category_id"]])

        if bboxes:
                    # Prepare filenames
                    base, ext = os.path.splitext(info["file_name"])
                    cropped_name = f"{base}_cropped{ext}"
                    label_name = f"{base}_labels.json"

                    # Save image
                    cropped_path = os.path.join(cropped_image_dir, cropped_name)
                    cropped_img.save(cropped_path)

                    # Save per-image JSON
                    label_data = {
                        "img": cropped_name,
                        "bboxes": bboxes,
                        "labels": labels
                    }

                    label_path = os.path.join(cropped_image_dir, label_name)
                    with open(label_path, "w") as f:
                        json.dump(label_data, f, indent=4)

                    print(f"Saved: {cropped_name}, Labels: {label_name}")
            

    print(f"Done. {len(results)} cropped images saved.")
    with open(os.path.join(cropped_image_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    print(f"Label map saved to {os.path.join(cropped_image_dir, 'label_map.json')}")    

if __name__ == "__main__":
    for d in ["train", "verify", "test"]:
        coco_to_custom_cropped_format(
            coco_json_path=f"/mnt_ai/data/fiftyone_traffic_objects_output/{d}/labels.json",
            image_dir=f"/mnt_ai/data/fiftyone_traffic_objects_output/{d}/data",
            cropped_image_dir=f"coco/{d}",
            crop_size=(160, 160)
        )
    