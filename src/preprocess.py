import os
import json
import cv2
import pandas as pd
from shapely import wkt
from tqdm import tqdm

# Define paths for inpyut and output data
RAW_DIR = "data/raw/train"
IMAGE_DIR = os.path.join(RAW_DIR, "images")
LABEL_DIR = os.path.join(RAW_DIR, "labels")

OUTPUT_DIR_TRAIN = "data/processed/train"
PRE_DIR = os.path.join(OUTPUT_DIR_TRAIN, "pre")
POST_DIR = os.path.join(OUTPUT_DIR_TRAIN, "post")

OUTPUT_DIR_HOLD = "data/processed/hold"
PRE_DIR_HOLD = os.path.join(OUTPUT_DIR_HOLD, "pre")
POST_DIR_HOLD = os.path.join(OUTPUT_DIR_HOLD, "post")

CROP_SIZE = 224

# Define damage label mapping
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

# Function to crop building from image based on polygon and resize to crop size
def crop_building(image, polygon, crop_size=224):
    # Get bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate center of the bounding box
    cx = int((minx + maxx) / 2)
    cy = int((miny + maxy) / 2)

    half = crop_size // 2
    
    # Ensure the crop is within image boundaries
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, image.shape[1])
    y2 = min(cy + half, image.shape[0])

    # Crop the image
    crop = image[y1:y2, x1:x2]

    # If the crop is empty (e.g., if the building is too close to the edge), return None
    if crop.size == 0:
        return None

    # Resize the crop to the desired size
    return cv2.resize(crop, (crop_size, crop_size))
def process_dataset(dataset_name):
    raw_dir = os.path.join("data/raw", dataset_name)
    image_dir = os.path.join(raw_dir, "images")
    label_dir = os.path.join(raw_dir, "labels")

    output_dir = os.path.join("data/processed", dataset_name)
    pre_dir = os.path.join(output_dir, "pre")
    post_dir = os.path.join(output_dir, "post")

    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)

    records = []

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]

    print(f"\nProcessing {dataset_name} dataset...")

    for label_file in tqdm(label_files):
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r") as f:
            label_data = json.load(f)

        post_image_name = label_data["metadata"]["img_name"]
        pre_image_name = post_image_name.replace("post_disaster", "pre_disaster")

        post_image_path = os.path.join(image_dir, post_image_name)
        pre_image_path = os.path.join(image_dir, pre_image_name)

        if not os.path.exists(pre_image_path) or not os.path.exists(post_image_path):
            continue

        pre_image = cv2.imread(pre_image_path)
        post_image = cv2.imread(post_image_path)

        if pre_image is None or post_image is None:
            continue

        buildings = label_data["features"]["xy"]

        for i, building in enumerate(buildings):
            subtype = building["properties"].get("subtype")

            if subtype not in DAMAGE_MAP:
                continue

            label = DAMAGE_MAP[subtype]

            polygon = wkt.loads(building["wkt"])

            pre_crop = crop_building(pre_image, polygon, CROP_SIZE)
            post_crop = crop_building(post_image, polygon, CROP_SIZE)

            if pre_crop is None or post_crop is None:
                continue

            base_name = label_file.replace(".json", "")
            crop_id = f"{dataset_name}_{base_name}_building_{i}"

            pre_filename = f"{crop_id}_pre.png"
            post_filename = f"{crop_id}_post.png"

            pre_save_path = os.path.join(pre_dir, pre_filename)
            post_save_path = os.path.join(post_dir, post_filename)

            cv2.imwrite(pre_save_path, pre_crop)
            cv2.imwrite(post_save_path, post_crop)

            records.append({
                "id": crop_id,
                "pre_image": pre_save_path,
                "post_image": post_save_path,
                "damage_label": label,
                "damage_type": subtype,
                "dataset": dataset_name
            })

    df = pd.DataFrame(records)
    labels_path = os.path.join(output_dir, "labels.csv")
    df.to_csv(labels_path, index=False)

    print(f"Finished processing {dataset_name}.")
    print(f"Processed {len(df)} building pairs.")
    print(f"Saved pre crops to: {pre_dir}")
    print(f"Saved post crops to: {post_dir}")
    print(f"Saved labels to: {labels_path}")

def main():
    process_dataset("train")
    process_dataset("hold")


if __name__ == "__main__":
    main()