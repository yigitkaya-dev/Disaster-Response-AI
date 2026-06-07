import os
import json
import cv2
import pandas as pd
from shapely import wkt
from tqdm import tqdm

RAW_DIR = "data/raw/train"
IMAGE_DIR = os.path.join(RAW_DIR, "images")
LABEL_DIR = os.path.join(RAW_DIR, "labels")

OUTPUT_DIR = "data/processed"
PRE_DIR = os.path.join(OUTPUT_DIR, "pre")
POST_DIR = os.path.join(OUTPUT_DIR, "post")

CROP_SIZE = 128

DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}


def crop_building(image, polygon, crop_size=128):
    minx, miny, maxx, maxy = polygon.bounds

    cx = int((minx + maxx) / 2)
    cy = int((miny + maxy) / 2)

    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, image.shape[1])
    y2 = min(cy + half, image.shape[0])

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (crop_size, crop_size))


def main():
    os.makedirs(PRE_DIR, exist_ok=True)
    os.makedirs(POST_DIR, exist_ok=True)

    records = []

    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".json")]

    for label_file in tqdm(label_files):
        label_path = os.path.join(LABEL_DIR, label_file)

        with open(label_path, "r") as f:
            label_data = json.load(f)

        post_image_name = label_data["metadata"]["img_name"]
        pre_image_name = post_image_name.replace("post_disaster", "pre_disaster")

        post_image_path = os.path.join(IMAGE_DIR, post_image_name)
        pre_image_path = os.path.join(IMAGE_DIR, pre_image_name)

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
            crop_id = f"{base_name}_building_{i}"

            pre_filename = f"{crop_id}_pre.png"
            post_filename = f"{crop_id}_post.png"

            pre_save_path = os.path.join(PRE_DIR, pre_filename)
            post_save_path = os.path.join(POST_DIR, post_filename)

            cv2.imwrite(pre_save_path, pre_crop)
            cv2.imwrite(post_save_path, post_crop)

            records.append({
                "id": crop_id,
                "pre_image": pre_save_path,
                "post_image": post_save_path,
                "damage_label": label,
                "damage_type": subtype
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)

    print(f"Processed {len(df)} building pairs.")
    print(f"Saved pre crops to: {PRE_DIR}")
    print(f"Saved post crops to: {POST_DIR}")
    print(f"Saved labels to: {os.path.join(OUTPUT_DIR, 'labels.csv')}")


if __name__ == "__main__":
    main()