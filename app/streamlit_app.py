import streamlit as st
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import re

st.set_page_config(page_title="xBD Damage Heatmap", layout="wide")
st.title("xBD Building Damage Heatmap (0–3 scale)")

# Hide error messages if desired
st.markdown("""
    <style>
        [data-testid="stAlert"] {display: none !important;}
        .stAlert {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# ====================== HELPER FUNCTIONS ======================
def load_image(path):
    return np.array(Image.open(path))

def load_labels(label_path):
    with open(label_path) as f:
        data = json.load(f)
    features = []
    if isinstance(data, dict):
        feat_container = data.get("features", data)
        if isinstance(feat_container, dict):
            lng_lat = feat_container.get("lng_lat")
            if isinstance(lng_lat, list):
                features = lng_lat
            elif isinstance(feat_container.get("features"), list):
                features = feat_container.get("features", [])
            else:
                for v in feat_container.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        features = v
                        break
        elif isinstance(feat_container, list):
            features = feat_container
    return features

def normalize_damage(text):
    if not text:
        return "unclassified"
    return str(text).strip().lower().replace(" ", "-").replace("_", "-")

def parse_polygon(poly_string):
    match = re.search(r'POLYGON\s*\(\s*\(\s*(.*?)\s*\)\s*\)', str(poly_string), re.IGNORECASE)
    if not match:
        return []
    points = []
    for pair in match.group(1).split(','):
        pair = pair.strip()
        if pair:
            try:
                x, y = map(float, pair.split())
                points.append([x, y])
            except:
                continue
    return points

def get_all_pixel_coords(features, img_width=1024, img_height=1024, flip_y=True, y_offset=0, x_offset=0):
    all_coords = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        poly_data = feature.get("xy") or feature.get("wkt") or feature.get("lng_lat")
        if poly_data:
            coords = parse_polygon(poly_data)
            if len(coords) >= 3:
                all_coords.append(coords)
   
    if not all_coords:
        return {}
   
    all_x = [p[0] for coords in all_coords for p in coords]
    all_y = [p[1] for coords in all_coords for p in coords]
    minx, maxx = min(all_x), max(all_x)
    miny, maxy = min(all_y), max(all_y)
   
    if maxx == minx or maxy == miny:
        return {i: coords for i, coords in enumerate(all_coords)}
   
    scaled_dict = {}
    for i, coords in enumerate(all_coords):
        scaled = []
        for x, y in coords:
            sx = (x - minx) / (maxx - minx) * (img_width - 1) + x_offset
            sy = (y - miny) / (maxy - miny) * (img_height - 1)
            if flip_y:
                sy = img_height - 1 - sy
            sy += y_offset
            scaled.append([sx, sy])
        scaled_dict[i] = scaled
    return scaled_dict

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Dataset Paths")
    xbd_root = st.text_input("Path to xbd folder", value="./data/xbd")
    
    if os.path.exists(xbd_root):
        available_splits = [d for d in os.listdir(xbd_root) if os.path.isdir(os.path.join(xbd_root, d))]
    else:
        available_splits = ["tier1", "hold", "test", "tier3"]
    split = st.selectbox("Select split", available_splits)
  
    st.header("Visualization Controls")
    damage_threshold = st.slider("Minimum damage level to show", 0, 3, 0)
    show_polygons = st.checkbox("Show building polygons", True)
    show_mask = st.checkbox("Overlay raster mask", False)
    alpha = st.slider("Polygon opacity", 0.3, 1.0, 0.65)
    use_rasterized = st.checkbox("Use rasterized overlay (recommended)", True)
    flip_y = st.checkbox("Flip Y coordinates", True)
    y_offset = st.slider("Y offset", -100, 100, 0)
    x_offset = st.slider("X offset", -100, 100, 0)

# ====================== PATHS ======================
images_dir = os.path.join(xbd_root, split, "images")
labels_dir = os.path.join(xbd_root, split, "labels")
masks_dir = os.path.join(xbd_root, split, "masks")

if not os.path.exists(images_dir):
    st.error(f"Images folder not found: {images_dir}")
    st.stop()

image_files = [f for f in os.listdir(images_dir) if f.endswith("_post_disaster.png")]
st.success(f"✅ Found {split}/images with {len(image_files)} image chips!")

selected = st.selectbox("Choose an image chip", sorted(image_files), index=0)

pre_path = os.path.join(images_dir, selected.replace("post", "pre"))
post_path = os.path.join(images_dir, selected)
label_path = os.path.join(labels_dir, selected.replace(".png", ".json"))
mask_path = os.path.join(masks_dir, selected.replace("_post_disaster.png", "_mask.png"))

# ====================== PRE & POST IMAGES ======================
col_pre, col_post = st.columns(2)
with col_pre:
    if os.path.exists(pre_path):
        st.image(load_image(pre_path), caption="🟢 Pre-disaster", use_container_width=True)
    else:
        st.warning("Pre-disaster image not found")
with col_post:
    if os.path.exists(post_path):
        st.image(load_image(post_path), caption="🔴 Post-disaster", use_container_width=True)
    else:
        st.warning("Post-disaster image not found")

# ====================== HEATMAP ======================
if st.button("Load Damage Heatmap & Mask", type="primary"):
    try:
        features = load_labels(label_path) if os.path.exists(label_path) else []
       
        if show_mask and os.path.exists(mask_path):
            st.image(load_image(mask_path), caption="🟦 Damage Mask", use_container_width=True)
        elif show_mask:
            st.warning(f"Mask not found: {mask_path}")
        
        st.subheader("Damage Heatmap Overlay")
        post_img = load_image(post_path)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(post_img)
        ax.axis("off")
        
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        drawn = 0
        level_map = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
        colors = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]
        
        scaled_coords = get_all_pixel_coords(features, post_img.shape[1], post_img.shape[0],
                                           flip_y=flip_y, y_offset=y_offset, x_offset=x_offset)
        
        if show_polygons and use_rasterized:
            overlay = Image.new('RGBA', (post_img.shape[1], post_img.shape[0]), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
        
        for idx, feature in enumerate(features):
            if not isinstance(feature, dict):
                continue
               
            props = feature.get("properties", {})
            raw = props.get("subtype") or props.get("damage") or props.get("damage_level", "")
            damage_str = normalize_damage(raw)
            if damage_str == "unclassified":
                continue
            level = level_map.get(damage_str, 0)
            if level < damage_threshold:
                continue
            counts[level] += 1
            
            if show_polygons and idx in scaled_coords:
                try:
                    pixel_coords = scaled_coords[idx]
                    if len(pixel_coords) < 3:
                        continue
                   
                    color = colors[level]
                    if use_rasterized:
                        draw.polygon([(int(round(x)), int(round(y))) for x, y in pixel_coords],
                                   fill=(*color, int(255 * alpha)))
                    drawn += 1
                except:
                    pass
        
        if show_polygons and use_rasterized:
            ax.imshow(overlay)
        
        st.pyplot(fig)
        
        # Summary
        st.subheader("Damage Summary")
        total = sum(counts.values())
        labels_text = ["No damage", "Minor", "Major", "Destroyed"]
        emoji = ["🟢", "🟡", "🟠", "🔴"]
        for lvl in range(4):
            st.write(f"{emoji[lvl]} **Level {lvl} — {labels_text[lvl]}**: {counts[lvl]} buildings")
       
        st.caption(f"**Total shown**: {total} | **Drawn**: {drawn} | Split: {split}")
        
    except Exception as e:
        st.error(f"Error generating heatmap: {str(e)}")