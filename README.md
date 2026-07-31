# Disaster Response AI

AI-powered pipeline for post-disaster building damage assessment and emergency resource allocation, built on satellite imagery from the xBD dataset.

The system takes pre- and post-disaster satellite images, locates individual buildings, classifies how badly each one was damaged, and displays the results in an interactive Streamlit dashboard with rule-based recommendations for emergency responders (search-and-rescue teams, medical units, shelter units, inspection teams, etc.).

This was built as a team project (Team 9, CAP 6951, Florida Atlantic University) by Shefatha Rabbany, Taylor Belyeu, Nicolas Hernandez, and Yigit Kaya.

---

## How It Works

The pipeline has three stages:

1. **Building Localization (UNetFormer)** — A transformer-based segmentation model with a ResNet-18 encoder scans post-disaster imagery and outputs a pixel-wise probability map of where buildings are located.
2. **Post-Processing (Contour Detection + Watershed)** — The raw segmentation output isn't perfectly clean, so contour detection and watershed segmentation are applied to separate clusters of predicted building pixels into distinct, individual buildings. This step is what lets us reliably crop each building out as its own image, even when the pixel-level mask overlaps or touches neighboring buildings.
3. **Damage Classification (ResNet-18)** — Each cropped building image is classified into one of four damage tiers:

   | Score | Label | Description |
   |---|---|---|
   | 0 | No damage | Undisturbed structure |
   | 1 | Minor damage | Partial roof/structure issues, nearby water, minor cracks |
   | 2 | Major damage | Partial wall/roof collapse, structure surrounded by water/mud |
   | 3 | Destroyed | Structure collapsed, scorched, or no longer present |

4. **Visualization & Recommendations (Streamlit)** — Results are rendered as a color-coded overlay (green → red) on the original imagery, with aggregated statistics and rule-based recommendations (e.g., clusters of destroyed buildings trigger search-and-rescue and heavy machinery recommendations).

```
Pre/Post Disaster Images
        ↓
UNetFormer Localization (building probability map)
        ↓
Contour Detection + Watershed (separate individual buildings)
        ↓
ResNet-18 Classification (per-building damage tier)
        ↓
Streamlit Dashboard (heatmap + resource recommendations)
```

---

## Results

Evaluated on the official xBD held-out test set:

| Model | Metric | Score |
|---|---|---|
| UNetFormer (localization) | Precision | 0.76 |
| UNetFormer (localization) | IoU | 0.53 |
| UNetFormer (localization) | F1 | 0.64 |
| ResNet-18 (classification) | Accuracy | 83.99% |
| ResNet-18 (classification) | Macro F1 | 83.40% |

The raw pixel-overlap (IoU) for localization is modest at 0.53. This metric alone isn't the best measure of success for our use case, since our goal isn't a pixel-perfect mask — it's identifying *where* a building is so it can be cropped and classified. Because localization precision is high (0.76), the model is reliable at flagging real buildings; the contour detection + watershed post-processing step described above is what turns those approximate pixel clusters into clean, separated building crops for the classifier, even when the segmentation mask itself is imprecise.

The classification model performs best on the two extreme classes (no damage and destroyed), which have the most visually distinct signatures. Minor damage is the hardest class to classify, since it's visually closer to both "no damage" and "major damage."

---

## Project Structure

```
Disaster-Response-AI/
│
├── app/
│   └── streamlit_app.py       # Main dashboard application
│
├── src/                        # Preprocessing, training, evaluation, inference scripts
├── notebooks/                  # Exploration / experimentation notebooks
├── models/                     # Model weights go here (not included in repo, see below)
├── requirements.txt
└── SETUP.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yigitkaya-dev/Disaster-Response-AI.git
cd Disaster-Response-AI
```

### 2. Create a virtual environment and install dependencies

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

Then install requirements:
```bash
pip install -r requirements.txt
```

### 3. Download the model weights

Model weights are too large to be stored in this GitHub repository, so they need to be downloaded manually from Kaggle before running the app.

- **Localization model** — `unetformer_best_localization.pth`
  Download from: https://www.kaggle.com/code/nicolashernandez1307/xbd-unetformer/notebook

- **Classification model** — `best_model.pth`
  Download from: https://www.kaggle.com/models/yigitkayadev/classification-model-natural-disaster/

Once downloaded, create a `models` folder in the root of the repository and place both files inside it:

```
Disaster-Response-AI/
└── models/
    ├── unetformer_best_localization.pth
    └── best_model.pth
```

### 4. Run the Streamlit app

From the root of the repository:

```bash
streamlit run app/streamlit_app.py
```

This will open the dashboard in your browser, where you can upload satellite imagery, view the localization and classification results, and see the generated damage overlay and resource recommendations.

---

## Dataset

This project uses the [xBD dataset](https://xview2.org/), which provides pre- and post-disaster satellite image pairs with expert-annotated building polygons and damage labels across six disaster types (hurricanes, floods, wildfires, volcanic eruptions, earthquakes, and tsunamis).

The dataset is not included in this repository due to its size. See `SETUP.md` for instructions on setting up the data folder structure if you want to retrain or re-evaluate the models.

---

## Technologies

Python, PyTorch, OpenCV, NumPy, Pandas, Streamlit, Scikit-learn, Shapely

---

## Known Limitations

- **Minor damage class** is harder to classify accurately due to subtle visual differences from the adjacent classes.
- **Localization IoU** is moderate (0.53); post-processing helps compensate, but imprecise masks can still propagate errors downstream in rare cases.
- **Domain gap**: the models were trained and evaluated on the curated xBD dataset. Performance on real-time, unlabeled operational satellite imagery (clouds, different resolutions/sensors, etc.) has not yet been tested.

## Future Work

- Improve minor-damage classification with focal loss or ensemble methods
- Evaluate the full pipeline end-to-end on real-world, unlabeled disaster imagery
- Extend the recommendation engine with logistics-aware optimization (e.g., road accessibility, available response assets)

---

## Acknowledgments

Built as part of CAP 6951 at Florida Atlantic University. Thanks to the creators of the [xBD dataset](https://xview2.org/) for making annotated satellite imagery publicly available for disaster response research.
