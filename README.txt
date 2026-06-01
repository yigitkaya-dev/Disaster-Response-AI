# DisasterResponseAI

AI-powered disaster damage assessment and emergency response recommendation system using satellite imagery and the xBD (xView2) dataset.

## Project Overview

Natural disasters such as hurricanes, floods, wildfires, and earthquakes can cause widespread infrastructure damage. Manually assessing affected areas using satellite imagery is time-consuming and difficult during emergency situations.

This project uses machine learning and computer vision to automatically analyze before-and-after satellite images, classify building damage severity, and generate recommendations for emergency response resources.

The system is built using the xBD dataset, which provides pre-disaster and post-disaster satellite imagery along with building footprint annotations and damage labels.

---

## Project Architecture

```text
xBD Dataset
     ↓
Data Pipeline & Preprocessing
     ↓
Building Crop Generation
     ↓
Damage Classification Model
     ↓
Reliability & Validation
     ↓
Visualization & Decision Support
     ↓
Emergency Response Recommendations
```

---

## Damage Classes

The model predicts one of four damage categories:

| Class        | Description                               |
| ------------ | ----------------------------------------- |
| No Damage    | Building appears intact                   |
| Minor Damage | Small visible damage                      |
| Major Damage | Significant structural damage             |
| Destroyed    | Building is severely damaged or destroyed |

---

## Project Structure

```text
DisasterResponseAI/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── hold/
│   │
│   └── processed/
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   └── exploration.ipynb
│
├── outputs/
│   ├── predictions/
│   └── visualizations/
│
├── src/
│   ├── visualize.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── dataset_statistics.py
│   ├── predict.py
│   ├── overlay.py
│   └── recommend.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## Main Components

### Part 1: Data Pipeline & Preprocessing

Goal:
Convert raw xBD imagery and labels into a machine-learning-ready dataset.

Tasks:

* Read xBD JSON labels
* Extract building polygons
* Crop buildings from satellite imagery
* Generate processed dataset

Outputs:

* preprocess.py
* data/processed/
* dataset statistics

---

### Part 2: Damage Classification Model

Goal:
Train a deep learning model to classify building damage severity.

Tasks:

* Train CNN / ResNet model
* Learn damage patterns from cropped buildings
* Save trained model weights

Outputs:

* model.py
* train.py
* best_model.pth

---

### Part 3: Reliability & Validation

Goal:
Evaluate model performance and identify weaknesses.

Tasks:

* Calculate accuracy
* Generate precision and recall metrics
* Build confusion matrix
* Analyze incorrect predictions

Outputs:

* evaluate.py
* error_analysis.py
* dataset_statistics.py

---

### Part 4: Visualization & Decision Support

Goal:
Provide useful information for emergency responders.

Tasks:

* Predict damage on new images
* Generate map overlays
* Produce emergency response recommendations
* Display results through a Streamlit dashboard

Outputs:

* predict.py
* overlay.py
* recommend.py
* streamlit_app.py

---

## Example Workflow

```text
Before Disaster Image
+
After Disaster Image
          ↓
Building Detection (Provided by xBD)
          ↓
Damage Classification
          ↓
Map Overlay
          ↓
Resource Recommendation
```

Example Output:

```text
Area A

Destroyed Buildings: 200
Major Damage: 500
Minor Damage: 800

Recommended Resources:
- 10 Utility Repair Crews
- 5 Search & Rescue Teams
- 3 Debris Removal Teams
```

---

## Technologies

* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Shapely
* Scikit-Learn
* Streamlit

---

## Future Improvements

* Automatic building detection using YOLO or SpaceNet models
* Disaster type classification
* GIS integration
* Real-time satellite image ingestion
* FEMA-style response planning dashboard

---

## Dataset

Dataset: xBD (xView2 Building Damage Assessment Dataset)

Contents:

* Pre-disaster satellite imagery
* Post-disaster satellite imagery
* Building footprint polygons
* Building damage labels

Reference:
https://xview2.org/

```
```
