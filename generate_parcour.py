import os
import random
import json
from PIL import Image

# CONFIG
OUTPUT_DIR = "synthetic_dataset"
BACKGROUND_DIR = "backgrounds"
OBJECT_DIR = "objects"
CLASSES = ["cube", "rectangle", "ball", "cone", "line", "floor_marking"]
IMAGE_SIZE = (160, 160)
NUM_IMAGES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

def load_object_images():
    obj_images = {}
    for class_name in CLASSES:
        class_dir = os.path.join(OBJECT_DIR, class_name)
        obj_images[class_name] = [Image.open(os.path.join(class_dir, f)).convert("RGBA")
                                  for f in os.listdir(class_dir) if f.endswith(".png")]
    return obj_images

def random_bbox(pasted_size, image_size):
    w, h = pasted_size
    img_w, img_h = image_size
    x1 = random.randint(0, img_w - w)
    y1 = random.randint(0, img_h - h)
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def generate_images():
    backgrounds = [Image.open(os.path.join(BACKGROUND_DIR, f)).convert("RGB")
                   for f in os.listdir(BACKGROUND_DIR) if f.endswith(".jpg") or f.endswith(".png")]
    objects = load_object_images()

    for i in range(NUM_IMAGES):
        bg = random.choice(backgrounds).resize(IMAGE_SIZE)
        label_data = []

        for _ in range(random.randint(1, 5)):  # number of objects per image
            class_name = random.choice(CLASSES)
            class_id = CLASSES.index(class_name)
            obj_img = random.choice(objects[class_name])
            scale = random.uniform(0.2, 0.6)
            obj_resized = obj_img.resize((int(obj_img.width * scale), int(obj_img.height * scale)))

            x1, y1, x2, y2 = random_bbox(obj_resized.size, IMAGE_SIZE)
            bg.paste(obj_resized, (x1, y1), obj_resized)

            # Save bbox in normalized format
            label_data.append({
                "class_id": class_id,
                "bbox": [x1 / IMAGE_SIZE[0], y1 / IMAGE_SIZE[1], x2 / IMAGE_SIZE[0], y2 / IMAGE_SIZE[1]]
            })

        # Save image and label
        img_name = f"synthetic_{i:04d}.png"
        label_name = f"synthetic_{i:04d}.json"
        bg.save(os.path.join(OUTPUT_DIR, "images", img_name))
        with open(os.path.join(OUTPUT_DIR, "labels", label_name), "w") as f:
            json.dump(label_data, f, indent=2)

    print(f"✅ Generated {NUM_IMAGES} synthetic images in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    generate_images()


