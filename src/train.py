import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import f1_score

# Constants
LABELS_CSV = "data/processed/train/labels.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 4
RANDOM_SEED = 42


class XBDPrePostDataset(Dataset):
    # Dataset class for loading pre and post disaster images along with their damage labels
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load pre and post images, convert to RGB, and apply transformations if provided
        pre_img = Image.open(row["pre_image"]).convert("RGB")
        post_img = Image.open(row["post_image"]).convert("RGB")
        label = int(row["damage_label"])

        # Concatenate pre and post images along the channel dimension to create a 6-channel input
        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)

        combined_img = torch.cat([pre_img, post_img], dim=0)

        return combined_img, label


def create_model():
    # Create a ResNet-18 model and modify the first convolutional layer to accept 6-channel input (pre and post images)
    # Load the pretrained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Save the original weights of the first convolutional layer
    old_conv = model.conv1

    # Replace the first convolutional layer to accept 6 input channels instead of 3
    model.conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # Initialize the new convolutional layer's weights by copying the
    # pretrained weights for the first 3 channels and duplicating them for the next 3 channels
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = old_conv.weight
        model.conv1.weight[:, 3:, :, :] = old_conv.weight

    # Replace the final fully connected layer to output the correct number of classes for damage classification
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

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

    dataset = XBDPrePostDataset(LABELS_CSV, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Total train dataset samples: {len(dataset)}")
    print(f"Training split samples: {len(train_dataset)}")
    print(f"Validation split samples: {len(val_dataset)}")

    pin_memory = device.type == "cuda"

    # --- Compute class counts / per-sample weights for the training split ---
    train_labels = [dataset.df.iloc[i]["damage_label"] for i in train_dataset.indices]
    train_labels = pd.Series(train_labels).astype(int)

    class_counts = train_labels.value_counts().sort_index()
    class_counts = class_counts.reindex(range(NUM_CLASSES), fill_value=0)

    print("\nClass counts in training split:")
    print(class_counts)

    # Inverse frequency per class -> used only for the sampler now, not the loss
    sample_weight_per_class = 1.0 / class_counts.replace(0, 1)  # avoid div-by-zero

    print("\nSampling weight per class:")
    print(sample_weight_per_class)

    # Map each training sample to its class's sampling weight
    sample_weights = train_labels.map(sample_weight_per_class).values
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,   # replaces shuffle=True — balances classes seen per batch
        num_workers=4,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,      # keep val set in its natural (imbalanced) distribution
        num_workers=4,
        pin_memory=pin_memory
    )

    model = create_model().to(device)

    # Loss is unweighted now — balancing is handled by the sampler instead
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total

        # Macro-F1 weights every class equally, regardless of how many
        # samples it has — unlike accuracy, it won't be dominated by no-damage.
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val Macro-F1: {val_macro_f1:.4f}")
        print(f"Per-class F1 [no-damage, minor-damage, major-damage, destroyed]: "
              f"{[round(f, 4) for f in per_class_f1]}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1

            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_macro_f1": best_val_f1,
                "class_names": [
                    "no-damage",
                    "minor-damage",
                    "major-damage",
                    "destroyed"
                ],
                "img_size": IMG_SIZE
            }, MODEL_PATH)

            print(f"Saved new best model: {MODEL_PATH}")

    print("\nTraining complete.")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()