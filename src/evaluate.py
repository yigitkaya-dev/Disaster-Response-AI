import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


LABELS_CSV = "data/processed/hold/labels.csv"
MODEL_PATH = "models/best_model.pth"

OUTPUT_DIR = "outputs"
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "hold_confusion_matrix.png")

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 4

CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]


class XBDPrePostDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pre_img = Image.open(row["pre_image"]).convert("RGB")
        post_img = Image.open(row["post_image"]).convert("RGB")
        label = int(row["damage_label"])

        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)

        combined_img = torch.cat([pre_img, post_img], dim=0)

        return combined_img, label


def create_model():
    model = models.resnet18(weights=None)

    old_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = XBDPrePostDataset(LABELS_CSV, transform=transform)

    print(f"Hold/test samples: {len(test_dataset)}")

    pin_memory = device.type == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )

    model = create_model().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating hold set"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("\nHold Set Evaluation Results")
    print("---------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Hold Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print(f"\nSaved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate()