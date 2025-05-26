import random
import os

# ⚠️ Set ALL relevant environment variables BEFORE importing fiftyone
os.environ["FIFTYONE_TEMP_DIR"] = "/mnt_ai/data/fiftyone_config"
os.environ["FIFTYONE_TMP_DIR"] = "/mnt_ai/data/fiftyone_config"
os.environ["FIFTYONE_CONFIG_DIR"] = "/mnt_ai/data/fiftyone_config"
os.environ["FIFTYONE_DATABASE_DIR"] = "/mnt_ai/data/fiftyone_db"
os.environ["FIFTYONE_APP_DIR"] = "/mnt_ai/data/fiftyone_app"
os.environ["FIFTYONE_CACHE_DIR"] = "/mnt_ai/data/fiftyone_cache"
os.environ["FIFTYONE_ZOO_DIR"] = "/mnt_ai/data/fiftyone_zoo"   # <-- crucial for dataset download path
os.environ["FIFTYONE_DISABLE_ZOO_CACHE"] = "0"  # keep dataset if you want
os.environ["FIFTYONE_DO_NOT_TRACK"] = "1"


import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# CONFIG
selected_classes = ["car", "bicycle", "truck", "traffic light", "stop sign"]
output_dir = "/mnt_ai/data/fiftyone_traffic_objects_output"

# Step 1: Load COCO validation split
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",  # can change to "train" for more samples
    label_types=["detections"],
    max_samples=10000,
    shuffle=True
)




# Step 2: Filter labels to selected classes
filtered = dataset.filter_labels(
    "ground_truth", 
    F("label").is_in(selected_classes)
)

# 3. Keep only images that have at least one label left
filtered = filtered.match(F("ground_truth.detections").length() > 0)



# 4. Then split
samples = list(filtered)
random.shuffle(samples)

num_train = int(len(samples) * 0.7)
num_test = int(len(samples) * .1)
num_verify = len(samples) - num_train - num_test


def split_samples(samples, train_n, test_n, verify_n):
    return (
        samples[:train_n],
        samples[train_n:train_n + test_n],
        samples[train_n + test_n:train_n + test_n + verify_n]
    )

train_samples, test_samples, verify_samples = split_samples(samples, num_train, num_test, num_verify)


# Step 5: Create new datasets
train_ds = fo.Dataset(name="traffic_train")
train_ds.add_samples(train_samples)

test_ds = fo.Dataset(name="traffic_test")
test_ds.add_samples(test_samples)

verify_ds = fo.Dataset(name="traffic_verify")
verify_ds.add_samples(verify_samples)

# Step 6: Export with bounding boxes
train_ds.export(
    export_dir=os.path.join(output_dir, "train"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth"
)

test_ds.export(
    export_dir=os.path.join(output_dir, "test"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth"
)

verify_ds.export(
    export_dir=os.path.join(output_dir, "verify"),
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth"
)

print(f"Export completed to directory: {output_dir}")

