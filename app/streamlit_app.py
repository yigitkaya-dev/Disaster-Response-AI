import os
import re
import sys
import json
import tempfile

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.building_mask_prediction import predict_single_image
from src.predict import predict


# ============================================================
# CONSTANTS
# ============================================================

LOCALIZATION_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "unetformer_best_localization.pth"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_model.pth"
)

CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

CLASS_LEVELS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

CLASS_COLORS = {
    "no-damage": (0, 255, 0),
    "minor-damage": (255, 255, 0),
    "major-damage": (255, 165, 0),
    "destroyed": (255, 0, 0)
}

CLASS_DISPLAY_NAMES = {
    "no-damage": "No damage",
    "minor-damage": "Minor damage",
    "major-damage": "Major damage",
    "destroyed": "Destroyed"
}


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Disaster Response AI",
    page_icon="🛰️",
    layout="wide"
)

st.title("Disaster Response AI")

st.write(
    "Locate buildings using UNetFormer and classify their damage "
    "using the trained pre/post ResNet18 model."
)


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def load_labels(label_path):
    with open(label_path, "r") as file:
        data = json.load(file)

    feature_container = data.get("features", data)

    if isinstance(feature_container, dict):
        xy_features = feature_container.get("xy")

        if isinstance(xy_features, list):
            return xy_features

        longitude_latitude = feature_container.get("lng_lat")

        if isinstance(longitude_latitude, list):
            return longitude_latitude

    elif isinstance(feature_container, list):
        return feature_container

    return []


def normalize_damage(text):
    if not text:
        return "unclassified"

    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
    )


def parse_polygon(poly_string):
    match = re.search(
        r"POLYGON\s*\(\s*\(\s*(.*?)\s*\)\s*\)",
        str(poly_string),
        re.IGNORECASE
    )

    if not match:
        return []

    points = []

    for pair in match.group(1).split(","):
        pair = pair.strip()

        if not pair:
            continue

        try:
            x, y = map(float, pair.split())
            points.append([x, y])

        except (ValueError, TypeError):
            continue

    return points


def get_all_pixel_coords(features):
    coordinate_dictionary = {}

    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue

        polygon_data = feature.get("xy") or feature.get("wkt")

        if not polygon_data:
            continue

        coordinates = parse_polygon(polygon_data)

        if len(coordinates) >= 3:
            coordinate_dictionary[index] = coordinates

    return coordinate_dictionary


def validate_scene_pair(pre_image, post_image):
    if pre_image.size != post_image.size:
        raise ValueError(
            "Pre- and post-disaster scenes must have matching dimensions. "
            f"Pre: {pre_image.size}; Post: {post_image.size}"
        )


def expand_box(
    box,
    image_width,
    image_height,
    padding=16
):
    return {
        "minx": max(0, int(box["minx"]) - padding),
        "miny": max(0, int(box["miny"]) - padding),
        "maxx": min(image_width, int(box["maxx"]) + padding),
        "maxy": min(image_height, int(box["maxy"]) + padding)
    }


def crop_from_box(image, box):
    return image.crop(
        (
            box["minx"],
            box["miny"],
            box["maxx"],
            box["maxy"]
        )
    )


def draw_prediction_overlay(post_image, predictions):
    overlay = post_image.convert("RGBA")

    drawing_layer = Image.new(
        "RGBA",
        overlay.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(drawing_layer)

    for result in predictions:
        box = result["box"]
        predicted_class = result["prediction"]
        confidence = result["confidence"]

        red, green, blue = CLASS_COLORS[predicted_class]

        draw.rectangle(
            (
                box["minx"],
                box["miny"],
                box["maxx"],
                box["maxy"]
            ),
            outline=(red, green, blue, 255),
            fill=(red, green, blue, 70),
            width=3
        )

        label_text = (
            f'{result["building_id"]}: '
            f'{CLASS_DISPLAY_NAMES[predicted_class]} '
            f'{confidence * 100:.0f}%'
        )

        text_x = box["minx"]
        text_y = max(0, box["miny"] - 14)

        draw.text(
            (text_x, text_y),
            label_text,
            fill=(red, green, blue, 255)
        )

    return Image.alpha_composite(
        overlay,
        drawing_layer
    ).convert("RGB")


# ============================================================
# FULL AI SCENE ANALYSIS
# ============================================================

def analyze_full_scene(
    pre_image,
    post_image,
    padding,
    minimum_confidence
):
    validate_scene_pair(pre_image, post_image)

    predictions = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        pre_scene_path = os.path.join(
            temporary_directory,
            "pre_scene.png"
        )

        post_scene_path = os.path.join(
            temporary_directory,
            "post_scene.png"
        )

        mask_path = os.path.join(
            temporary_directory,
            "predicted_building_mask.png"
        )

        pre_image.save(pre_scene_path)
        post_image.save(post_scene_path)

        mask_visual, building_boxes = predict_single_image(
            image_path=pre_scene_path,
            weight_path=LOCALIZATION_MODEL_PATH,
            output_path=mask_path
        )

        image_width, image_height = pre_image.size

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_buildings = len(building_boxes)

        for building_index, original_box in enumerate(building_boxes):
            status_text.write(
                f"Classifying building "
                f"{building_index + 1} of {total_buildings}"
            )

            box = expand_box(
                original_box,
                image_width=image_width,
                image_height=image_height,
                padding=padding
            )

            if (
                box["maxx"] <= box["minx"]
                or box["maxy"] <= box["miny"]
            ):
                continue

            pre_crop = crop_from_box(
                pre_image,
                box
            )

            post_crop = crop_from_box(
                post_image,
                box
            )

            pre_crop_path = os.path.join(
                temporary_directory,
                f"building_{building_index}_pre.png"
            )

            post_crop_path = os.path.join(
                temporary_directory,
                f"building_{building_index}_post.png"
            )

            pre_crop.save(pre_crop_path)
            post_crop.save(post_crop_path)

            predicted_class, confidence, probabilities = predict(
                pre_crop_path,
                post_crop_path
            )

            if confidence >= minimum_confidence:
                predictions.append({
                    "building_id": building_index,
                    "box": box,
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities
                })

            progress_bar.progress(
                (building_index + 1)
                / max(total_buildings, 1)
            )

        progress_bar.empty()
        status_text.empty()

    prediction_overlay = draw_prediction_overlay(
        post_image,
        predictions
    )

    return mask_visual, building_boxes, predictions, prediction_overlay


# ============================================================
# APPLICATION TABS
# ============================================================

ai_tab, dataset_tab = st.tabs(
    [
        "AI Scene Analysis",
        "Ground-Truth Dataset Viewer"
    ]
)


# ============================================================
# TAB 1 — AI SCENE ANALYSIS
# ============================================================

with ai_tab:
    st.header("Full Satellite Scene Analysis")

    st.write(
        "Upload matching pre- and post-disaster satellite scenes. "
        "The localization model detects buildings from the pre-disaster "
        "scene, and the classification model predicts damage from each "
        "matching pre/post building pair."
    )

    upload_column_pre, upload_column_post = st.columns(2)

    with upload_column_pre:
        uploaded_pre_scene = st.file_uploader(
            "Upload pre-disaster scene",
            type=["png", "jpg", "jpeg"],
            key="ai_pre_scene"
        )

    with upload_column_post:
        uploaded_post_scene = st.file_uploader(
            "Upload post-disaster scene",
            type=["png", "jpg", "jpeg"],
            key="ai_post_scene"
        )

    settings_column_one, settings_column_two = st.columns(2)

    with settings_column_one:
        crop_padding = st.slider(
            "Building crop padding",
            min_value=0,
            max_value=50,
            value=16,
            help=(
                "Adds surrounding context around each detected "
                "building before classification."
            )
        )

    with settings_column_two:
        minimum_confidence = st.slider(
            "Minimum prediction confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    if (
        uploaded_pre_scene is not None
        and uploaded_post_scene is not None
    ):
        pre_scene = Image.open(
            uploaded_pre_scene
        ).convert("RGB")

        post_scene = Image.open(
            uploaded_post_scene
        ).convert("RGB")

        scene_column_pre, scene_column_post = st.columns(2)

        with scene_column_pre:
            st.image(
                pre_scene,
                caption="Pre-disaster scene",
                use_container_width=True
            )

        with scene_column_post:
            st.image(
                post_scene,
                caption="Post-disaster scene",
                use_container_width=True
            )

        if st.button(
            "Run Full AI Analysis",
            type="primary",
            key="run_ai_analysis"
        ):
            try:
                if not os.path.exists(LOCALIZATION_MODEL_PATH):
                    raise FileNotFoundError(
                        "Localization model not found at: "
                        f"{LOCALIZATION_MODEL_PATH}"
                    )

                if not os.path.exists(CLASSIFICATION_MODEL_PATH):
                    raise FileNotFoundError(
                        "Classification model not found at: "
                        f"{CLASSIFICATION_MODEL_PATH}"
                    )

                with st.spinner(
                    "Locating and classifying buildings..."
                ):
                    (
                        predicted_mask,
                        building_boxes,
                        predictions,
                        prediction_overlay
                    ) = analyze_full_scene(
                        pre_image=pre_scene,
                        post_image=post_scene,
                        padding=crop_padding,
                        minimum_confidence=minimum_confidence
                    )

                result_column_mask, result_column_overlay = st.columns(2)

                with result_column_mask:
                    st.subheader("Predicted Building Mask")

                    st.image(
                        predicted_mask,
                        caption=(
                            f"UNetFormer detected "
                            f"{len(building_boxes)} building regions"
                        ),
                        use_container_width=True,
                        clamp=True
                    )

                with result_column_overlay:
                    st.subheader("Predicted Damage Overlay")

                    st.image(
                        prediction_overlay,
                        caption=(
                            f"{len(predictions)} classified buildings"
                        ),
                        use_container_width=True
                    )

                counts = {
                    class_name: 0
                    for class_name in CLASS_NAMES
                }

                for result in predictions:
                    counts[result["prediction"]] += 1

                st.subheader("Predicted Damage Summary")

                summary_columns = st.columns(4)

                summary_emojis = {
                    "no-damage": "🟢",
                    "minor-damage": "🟡",
                    "major-damage": "🟠",
                    "destroyed": "🔴"
                }

                for column, class_name in zip(
                    summary_columns,
                    CLASS_NAMES
                ):
                    with column:
                        st.metric(
                            (
                                f"{summary_emojis[class_name]} "
                                f"{CLASS_DISPLAY_NAMES[class_name]}"
                            ),
                            counts[class_name]
                        )

                st.write(
                    f"**Buildings localized:** {len(building_boxes)}"
                )

                st.write(
                    f"**Buildings classified above confidence "
                    f"threshold:** {len(predictions)}"
                )

                if predictions:
                    st.subheader("Building-Level Predictions")

                    result_rows = []

                    for result in predictions:
                        result_rows.append({
                            "Building": result["building_id"],
                            "Prediction": CLASS_DISPLAY_NAMES[
                                result["prediction"]
                            ],
                            "Confidence": (
                                f'{result["confidence"] * 100:.2f}%'
                            ),
                            "Min X": result["box"]["minx"],
                            "Min Y": result["box"]["miny"],
                            "Max X": result["box"]["maxx"],
                            "Max Y": result["box"]["maxy"]
                        })

                    st.dataframe(
                        result_rows,
                        use_container_width=True
                    )

            except Exception as error:
                st.error(
                    f"AI scene analysis failed: {error}"
                )


# ============================================================
# TAB 2 — GROUND-TRUTH DATASET VIEWER
# ============================================================

with dataset_tab:
    st.header("xBD Ground-Truth Damage Heatmap")

    st.caption(
        "This tab reads known damage labels from the xBD JSON files. "
        "It does not use model predictions."
    )

    with st.sidebar:
        st.header("Dataset Viewer")

        xbd_root = st.text_input(
            "Path to xBD folder",
            value="./data/raw"
        )

        damage_threshold = st.slider(
            "Minimum ground-truth damage level",
            min_value=0,
            max_value=3,
            value=0
        )

        polygon_opacity = st.slider(
            "Ground-truth polygon opacity",
            min_value=0.3,
            max_value=1.0,
            value=0.65
        )

    if os.path.exists(xbd_root):
        available_splits = [
            directory
            for directory in os.listdir(xbd_root)
            if os.path.isdir(
                os.path.join(xbd_root, directory)
            )
        ]
    else:
        available_splits = []

    if not available_splits:
        st.warning(
            f"No dataset splits were found under {xbd_root}."
        )

    else:
        selected_split = st.selectbox(
            "Select dataset split",
            sorted(available_splits)
        )

        images_directory = os.path.join(
            xbd_root,
            selected_split,
            "images"
        )

        labels_directory = os.path.join(
            xbd_root,
            selected_split,
            "labels"
        )

        if not os.path.exists(images_directory):
            st.error(
                f"Images directory not found: {images_directory}"
            )

        else:
            post_image_files = sorted(
                filename
                for filename in os.listdir(images_directory)
                if filename.endswith("_post_disaster.png")
            )

            if not post_image_files:
                st.warning(
                    "No post-disaster image chips were found."
                )

            else:
                selected_filename = st.selectbox(
                    "Choose an image chip",
                    post_image_files
                )

                pre_filename = selected_filename.replace(
                    "_post_disaster.png",
                    "_pre_disaster.png"
                )

                selected_pre_path = os.path.join(
                    images_directory,
                    pre_filename
                )

                selected_post_path = os.path.join(
                    images_directory,
                    selected_filename
                )

                selected_label_path = os.path.join(
                    labels_directory,
                    selected_filename.replace(".png", ".json")
                )

                viewer_column_pre, viewer_column_post = st.columns(2)

                with viewer_column_pre:
                    if os.path.exists(selected_pre_path):
                        st.image(
                            load_image(selected_pre_path),
                            caption="Pre-disaster",
                            use_container_width=True
                        )

                with viewer_column_post:
                    if os.path.exists(selected_post_path):
                        st.image(
                            load_image(selected_post_path),
                            caption="Post-disaster",
                            use_container_width=True
                        )

                if st.button(
                    "Load Ground-Truth Heatmap",
                    key="load_ground_truth"
                ):
                    try:
                        features = load_labels(
                            selected_label_path
                        )

                        post_image_array = load_image(
                            selected_post_path
                        )

                        figure, axis = plt.subplots(
                            figsize=(12, 12)
                        )

                        axis.imshow(post_image_array)
                        axis.axis("off")

                        overlay = Image.new(
                            "RGBA",
                            (
                                post_image_array.shape[1],
                                post_image_array.shape[0]
                            ),
                            (0, 0, 0, 0)
                        )

                        draw = ImageDraw.Draw(overlay)

                        coordinates = get_all_pixel_coords(
                            features
                        )

                        counts = {
                            0: 0,
                            1: 0,
                            2: 0,
                            3: 0
                        }

                        for index, feature in enumerate(features):
                            properties = feature.get(
                                "properties",
                                {}
                            )

                            raw_damage = (
                                properties.get("subtype")
                                or properties.get("damage")
                                or properties.get(
                                    "damage_level",
                                    ""
                                )
                            )

                            damage_class = normalize_damage(
                                raw_damage
                            )

                            if damage_class not in CLASS_LEVELS:
                                continue

                            damage_level = CLASS_LEVELS[
                                damage_class
                            ]

                            if damage_level < damage_threshold:
                                continue

                            counts[damage_level] += 1

                            if index not in coordinates:
                                continue

                            red, green, blue = CLASS_COLORS[
                                damage_class
                            ]

                            draw.polygon(
                                [
                                    (
                                        int(round(x)),
                                        int(round(y))
                                    )
                                    for x, y in coordinates[index]
                                ],
                                fill=(
                                    red,
                                    green,
                                    blue,
                                    int(255 * polygon_opacity)
                                )
                            )

                        axis.imshow(overlay)

                        st.pyplot(figure)
                        plt.close(figure)

                        summary_columns = st.columns(4)

                        level_to_class = {
                            0: "no-damage",
                            1: "minor-damage",
                            2: "major-damage",
                            3: "destroyed"
                        }

                        for level, column in enumerate(summary_columns):
                            class_name = level_to_class[level]

                            with column:
                                st.metric(
                                    CLASS_DISPLAY_NAMES[class_name],
                                    counts[level]
                                )

                    except Exception as error:
                        st.error(
                            f"Ground-truth visualization failed: {error}"
                        )