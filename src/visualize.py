import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shapely import wkt

### -- This is just a simple way to confirm that the polygons in files match the house locations -- ###

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "raw" / "train"

IMAGES_DIR = TRAIN_DIR / "images"
LABELS_DIR = TRAIN_DIR / "labels"

# Pick one post-disaster label file
label_path = next(LABELS_DIR.glob("*post_disaster.json"))

# Get matching image
image_path = IMAGES_DIR / label_path.name.replace(".json", ".png")

# Open image
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Open label JSON
with open(label_path, "r") as f:
    label_data = json.load(f)

# Damage colors
colors = {
    "no-damage": "green",
    "minor-damage": "yellow",
    "major-damage": "orange",
    "destroyed": "red",
    "un-classified": "blue",
}

# Draw polygons
for feature in label_data["features"]["xy"]:
    damage_class = feature["properties"].get("subtype", "un-classified")
    polygon = wkt.loads(feature["wkt"])

    coords = list(polygon.exterior.coords)
    color = colors.get(damage_class, "blue")

    draw.line(coords + [coords[0]], fill=color, width=3)

print(f"Image used: {image_path}")
print(f"Label used: {label_path}")

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()