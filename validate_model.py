import argparse
import os
import sys
import yaml
from ultralytics import YOLO
from class_config import CANONICAL_CLASSES, normalize_class_name

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate YOLOv8 model classes against CANONICAL_CLASSES schema.")
    parser.add_argument('--model', type=str, default="yolov8m.pt", help="Path to YOLOv8 model weights file")
    parser.add_argument('--data', type=str, default=None, help="Optional path to dataset data.yaml file to check instance counts")
    return parser.parse_args()

def check_data_instance_counts(data_yaml_path):
    """
    Count instances per canonical class in the dataset labels folder specified by data.yaml.
    """
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml file '{data_yaml_path}' not found.")
        return {}

    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = config.get('path', os.path.dirname(data_yaml_path))
    train_rel = config.get('train', 'images/train')
    labels_rel = train_rel.replace('images', 'labels')

    labels_dir = os.path.join(dataset_path, labels_rel)
    if not os.path.exists(labels_dir):
        labels_dir = os.path.join(os.path.dirname(data_yaml_path), 'labels', 'train')

    dataset_names = config.get('names', {})
    if isinstance(dataset_names, dict):
        sorted_keys = sorted(dataset_names.keys())
        dataset_names = [dataset_names[k] for k in sorted_keys]

    instance_counts = {canonical_name: 0 for canonical_name in CANONICAL_CLASSES}

    if os.path.exists(labels_dir):
        for root, _, files in os.walk(labels_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                if cls_id < len(dataset_names):
                                    raw_name = dataset_names[cls_id]
                                    canonical_name = normalize_class_name(raw_name)
                                    if canonical_name in instance_counts:
                                        instance_counts[canonical_name] += 1

    return instance_counts

def main():
    args = parse_args()

    print(f"Loading YOLOv8 model from '{args.model}'...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Normalize model class names
    model_raw_names = set(model.names.values()) if hasattr(model, 'names') else set()
    model_normalized_classes = {normalize_class_name(name) for name in model_raw_names}

    canonical_keys = list(CANONICAL_CLASSES.keys())
    total_canonical = len(canonical_keys)
    supported_count = 0

    instance_counts = {}
    if args.data:
        print(f"Analyzing training dataset instance counts from '{args.data}'...")
        instance_counts = check_data_instance_counts(args.data)

    print("\n" + "=" * 80)
    print(f"{'Canonical Class':<25} | {'Supported':<10} | {'Train Instances':<16} | {'Status Note':<25}")
    print("=" * 80)

    for cls_name in canonical_keys:
        is_supported = cls_name in model_normalized_classes
        sup_str = "YES" if is_supported else "NO"

        if is_supported:
            supported_count += 1

        inst_cnt_str = "-"
        status_note = ""

        if args.data:
            cnt = instance_counts.get(cls_name, 0)
            inst_cnt_str = str(cnt)
            if cnt < 50:
                status_note = "LOW DATA — detection may be unreliable"

        print(f"{cls_name:<25} | {sup_str:<10} | {inst_cnt_str:<16} | {status_note:<25}")

    print("=" * 80)
    print(f"Summary: {supported_count}/{total_canonical} expected classes supported by this model.")
    print("=" * 80)

if __name__ == '__main__':
    main()
