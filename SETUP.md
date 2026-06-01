# DisasterResponseAI Setup Guide

## 1. Clone the Repository

Open a terminal and run:

```bash
git clone <REPOSITORY_URL>
cd DisasterResponseAI
```

---

## 2. Install Dependencies

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 3. Download the xBD Dataset

Download:

* Challenge Training Set
* Challenge Holdout Set

from the xBD website.

Extract the downloaded archives.

---

## 4. Create the Data Folder Structure

The dataset is NOT stored in GitHub because it is too large.

Each team member must create the following structure locally:

```text
data/
└── raw/
    ├── train/
    │   ├── images/
    │   ├── labels/
    │   └── targets/
    │
    └── hold/
        ├── images/
        ├── labels/
        └── targets/
```

Example:

```text
data/raw/train/images
data/raw/train/labels
data/raw/train/targets

data/raw/hold/images
data/raw/hold/labels
data/raw/hold/targets
```

Move the extracted xBD files into the corresponding folders.

---

## 5. Verify Setup

Run:

```bash
python src/visualize.py
```

If setup is correct, a satellite image should appear with building polygons drawn on top.

---

# Working with Git

## Pull Latest Changes

Before starting work:

```bash
git pull origin main
```

---

## Create a Branch

Create a branch for your feature:

```bash
git checkout -b feature-name
```

Examples:

```bash
git checkout -b preprocess
git checkout -b training
git checkout -b dashboard
```

---

## Save Changes

Check modified files:

```bash
git status
```

Stage files:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Added preprocessing pipeline"
```

---

## Push Changes

Push your branch:

```bash
git push origin feature-name
```

Example:

```bash
git push origin preprocess
```

---

## Update Local Repository

If someone else pushed code:

```bash
git pull origin main
```

before continuing work.

---

# Important Notes

DO NOT commit:

```text
data/raw/
data/processed/
outputs/
*.pth
```

These files are already ignored by `.gitignore`.

Only commit:

```text
src/
app/
notebooks/
README.md
requirements.txt
```

The dataset, trained models, and generated outputs should remain local.
