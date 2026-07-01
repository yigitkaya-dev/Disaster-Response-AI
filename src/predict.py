import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 224
NUM_CLASSES = 4

CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]


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


def load_image_pair(pre_image_path, post_image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    pre_img = Image.open(pre_image_path).convert("RGB")
    post_img = Image.open(post_image_path).convert("RGB")

    pre_tensor = transform(pre_img)
    post_tensor = transform(post_img)

    combined_tensor = torch.cat([pre_tensor, post_tensor], dim=0)

    # Add batch dimension: [6, 224, 224] -> [1, 6, 224, 224]
    combined_tensor = combined_tensor.unsqueeze(0)

    return combined_tensor


def predict(pre_image_path, post_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor = load_image_pair(pre_image_path, post_image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = confidence.item()

    return predicted_label, confidence_score, probabilities.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(
        description="Predict building damage from pre- and post-disaster image crops."
    )

    parser.add_argument("--pre", required=True, help="Path to pre-disaster building crop")
    parser.add_argument("--post", required=True, help="Path to post-disaster building crop")

    args = parser.parse_args()

    prediction, confidence, probabilities = predict(args.pre, args.post)

    print("\nPrediction Result")
    print("-----------------")
    print(f"Pre image:  {args.pre}")
    print(f"Post image: {args.post}")
    print(f"Predicted damage: {prediction}")
    print(f"Confidence: {confidence:.4f}")

    print("\nClass probabilities:")
    for class_name, prob in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()