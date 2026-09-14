"""
Scene Classifier — image-level disaster type classification using ResNet18.

Classifies full images into disaster categories BEFORE object detection runs.
This gives a scene-level "what kind of disaster is this?" alongside YOLO's
object-level "what specific things are in this image?"

Classes: fire, flood, earthquake, storm, normal

Training: python scene_classifier.py --train --data-dir datasets/disaster_scenes
Inference: used automatically in the upload pipeline via classify_scene()

Dataset structure expected:
    datasets/disaster_scenes/
        train/
            fire/        (images of fire scenes)
            flood/       (images of flood scenes)
            earthquake/  (images of earthquake/collapse scenes)
            storm/       (images of storm damage)
            normal/      (images of normal/undamaged areas)
        val/
            fire/
            flood/
            earthquake/
            storm/
            normal/
"""

import os
import json
import numpy as np

SCENE_CLASSES = ["earthquake", "fire", "flood", "normal", "storm"]

SCENE_CLASS_META = {
    "fire": {"color": "#ef4444", "icon": "F", "label": "FIRE EVENT"},
    "flood": {"color": "#3b82f6", "icon": "W", "label": "FLOOD EVENT"},
    "earthquake": {"color": "#a855f7", "icon": "E", "label": "EARTHQUAKE"},
    "storm": {"color": "#6366f1", "icon": "S", "label": "STORM DAMAGE"},
    "normal": {"color": "#22c55e", "icon": "N", "label": "NORMAL"},
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene_classifier_model.pth")


def _load_model():
    """Load the trained ResNet18 scene classifier."""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(SCENE_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def _preprocess_image(image_path):
    """Load and preprocess image for ResNet18 (224x224, normalized)."""
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def _preprocess_cv2_frame(frame):
    """Preprocess an OpenCV BGR frame for ResNet18."""
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        import cv2
    except ImportError:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(pil_img).unsqueeze(0)


def classify_scene(image_path=None, cv2_frame=None):
    """
    Classify a scene from an image file or OpenCV frame.

    Returns:
        dict with scene_class, confidence, label, color, icon, all_scores
        or None if model not available
    """
    model = _load_model()

    if model is None:
        return _classify_scene_fallback(image_path, cv2_frame)

    try:
        import torch
    except ImportError:
        return _classify_scene_fallback(image_path, cv2_frame)

    if image_path:
        tensor = _preprocess_image(image_path)
    elif cv2_frame is not None:
        tensor = _preprocess_cv2_frame(cv2_frame)
    else:
        return None

    if tensor is None:
        return _classify_scene_fallback(image_path, cv2_frame)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    scores = {cls: round(probs[i].item(), 4) for i, cls in enumerate(SCENE_CLASSES)}
    top_idx = probs.argmax().item()
    top_class = SCENE_CLASSES[top_idx]
    top_conf = probs[top_idx].item()

    meta = SCENE_CLASS_META.get(top_class, {})

    return {
        "scene_class": top_class,
        "confidence": round(top_conf, 3),
        "label": meta.get("label", top_class.upper()),
        "color": meta.get("color", "#64748b"),
        "icon": meta.get("icon", "?"),
        "all_scores": scores,
        "method": "resnet18",
    }


def _classify_scene_fallback(image_path=None, cv2_frame=None):
    """
    Fallback scene classification using color histogram analysis.
    Used when PyTorch/trained model is not available.
    Analyzes dominant colors to guess disaster type.
    """
    try:
        import cv2
    except ImportError:
        return {
            "scene_class": "normal",
            "confidence": 0.0,
            "label": "NORMAL",
            "color": "#22c55e",
            "icon": "N",
            "all_scores": {},
            "method": "unavailable",
        }

    if cv2_frame is not None:
        frame = cv2_frame
    elif image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            return None
    else:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total_pixels = frame.shape[0] * frame.shape[1]

    red_mask = ((h < 10) | (h > 170)) & (s > 80) & (v > 80)
    orange_mask = ((h >= 10) & (h < 25)) & (s > 80) & (v > 80)
    fire_pixels = (np.count_nonzero(red_mask) + np.count_nonzero(orange_mask)) / total_pixels

    blue_mask = ((h >= 90) & (h < 130)) & (s > 40)
    brown_mask = ((h >= 10) & (h < 30)) & (s > 30) & (v < 150)
    water_pixels = np.count_nonzero(blue_mask) / total_pixels

    gray_mask = (s < 30) & (v > 50) & (v < 200)
    gray_pixels = np.count_nonzero(gray_mask) / total_pixels

    green_mask = ((h >= 35) & (h < 85)) & (s > 40) & (v > 40)
    green_pixels = np.count_nonzero(green_mask) / total_pixels

    brown_pixels = np.count_nonzero(brown_mask) / total_pixels

    scores = {
        "fire": min(1.0, fire_pixels * 5),
        "flood": min(1.0, water_pixels * 3),
        "earthquake": min(1.0, (gray_pixels * 2 + brown_pixels * 3) * 0.8),
        "storm": min(1.0, (gray_pixels * 1.5 + green_pixels * 0.5) * 0.6),
        "normal": min(1.0, green_pixels * 2 + (1 - fire_pixels - water_pixels) * 0.3),
    }

    top_class = max(scores, key=scores.get)
    top_conf = scores[top_class]

    meta = SCENE_CLASS_META.get(top_class, {})

    return {
        "scene_class": top_class,
        "confidence": round(top_conf, 3),
        "label": meta.get("label", top_class.upper()),
        "color": meta.get("color", "#64748b"),
        "icon": meta.get("icon", "?"),
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
        "method": "color_histogram",
    }


def train_model(data_dir, epochs=15, batch_size=32, lr=0.001):
    """
    Fine-tune ResNet18 on disaster scene images.

    Args:
        data_dir: path containing train/ and val/ subdirectories
                  each with fire/, flood/, earthquake/, storm/, normal/ folders
        epochs: training epochs (default 15)
        batch_size: batch size (default 32)
        lr: learning rate (default 0.001)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"Training classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(SCENE_CLASSES))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs} — "
              f"Loss: {running_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model (val acc: {val_acc:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scene Classifier for Disaster Images")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data-dir", type=str, default="datasets/disaster_scenes",
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--classify", type=str, help="Classify a single image")

    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir, args.epochs, args.batch_size, args.lr)
    elif args.classify:
        result = classify_scene(image_path=args.classify)
        if result:
            print(f"Scene: {result['label']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Method: {result['method']}")
            print(f"All scores: {json.dumps(result['all_scores'], indent=2)}")
        else:
            print("Could not classify image.")
    else:
        parser.print_help()
