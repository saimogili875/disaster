import argparse
import json
import os
import random
import shutil
import sys
import cv2
import numpy as np
import yaml
from class_config import CANONICAL_CLASSES, normalize_class_name

# RescueNet class index mapping
RESCUENET_RAW_CLASSES = {
    0: "background",
    1: "debris",
    2: "water",
    3: "building-no-damage",
    4: "building-minor-damage",
    5: "building-major-damage",
    6: "building-total-destruction",
    7: "road-clear",
    8: "road-blocked",
    9: "vehicle",
    10: "tree",
    11: "pool"
}

def parse_args():
    """Parse CLI arguments for merging datasets."""
    parser = argparse.ArgumentParser(description="Merge RescueNet (Dataset A) and Roboflow YOLO (Dataset B) into unified YOLO dataset.")
    parser.add_argument('--dataset-a', type=str, required=True, help="Path to Dataset A (RescueNet root containing images/ and masks/)")
    parser.add_argument('--dataset-b', type=str, required=True, help="Path to Dataset B (Roboflow YOLO root containing images/, labels/, and data.yaml)")
    parser.add_argument('--output', type=str, required=True, help="Path to output unified YOLO dataset directory")
    parser.add_argument('--min-area', type=int, default=200, help="Minimum contour area in pixels for Dataset A masks (default: 200)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducible train/val split (default: 42)")
    return parser.parse_args()

def load_dataset_b_classes(dataset_b_path):
    """Load class names from Dataset B's data.yaml file."""
    yaml_path = os.path.join(dataset_b_path, "data.yaml")
    if not os.path.exists(yaml_path):
        yaml_path_sub = os.path.join(dataset_b_path, "data.yml")
        if os.path.exists(yaml_path_sub):
            yaml_path = yaml_path_sub
        else:
            print(f"Warning: data.yaml not found in '{dataset_b_path}'. Assuming default classes ['fire', 'smoke'].")
            return ['fire', 'smoke']

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    names = data.get('names', [])
    if isinstance(names, dict):
        sorted_keys = sorted(names.keys())
        names = [names[k] for k in sorted_keys]
    return names

def process_dataset_a(dataset_a_path, min_area, unified_class_to_id):
    """
    Process RescueNet mask files, find contours per class, normalize names, and convert to YOLO bounding boxes.
    """
    images_dir = os.path.join(dataset_a_path, "images")
    masks_dir = os.path.join(dataset_a_path, "masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Dataset A must contain 'images' and 'masks' subdirectories in '{dataset_a_path}'.")
        sys.exit(1)

    samples = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

    print(f"Processing Dataset A (RescueNet): found {len(image_files)} images...")

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        mask_path = None
        for ext in ('.png', '.jpg'):
            candidate = os.path.join(masks_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if not mask_path:
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        img_h, img_w = mask.shape[:2]
        labels = []

        for raw_cls_id, raw_name in RESCUENET_RAW_CLASSES.items():
            if raw_cls_id == 0 or raw_name in ("background", "road-clear", "road-blocked", "pool"):
                continue

            canonical_name = normalize_class_name(raw_name)
            if canonical_name not in unified_class_to_id:
                continue

            unified_cls_id = unified_class_to_id[canonical_name]

            binary_mask = (mask == raw_cls_id).astype(np.uint8)
            if not np.any(binary_mask):
                continue

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = (x + w / 2.0) / img_w
                    cy = (y + h / 2.0) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    labels.append((unified_cls_id, cx, cy, norm_w, norm_h))

        samples.append({
            'source': 'Dataset A',
            'image_path': img_path,
            'filename': img_name,
            'labels': labels
        })

    return samples

def process_dataset_b(dataset_b_path, b_class_names, unified_class_to_id):
    """
    Process Roboflow YOLO dataset files, normalizing class names to canonical classes.
    """
    images_dir = os.path.join(dataset_b_path, "images")
    labels_dir = os.path.join(dataset_b_path, "labels")

    img_files = []
    if os.path.exists(images_dir):
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_files.append(os.path.join(root, f))
    else:
        print(f"Error: Dataset B must contain an 'images' subdirectory in '{dataset_b_path}'.")
        sys.exit(1)

    samples = []
    print(f"Processing Dataset B (Roboflow): found {len(img_files)} images...")

    for img_path in img_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]

        label_file = None
        if os.path.exists(labels_dir):
            for root, _, files in os.walk(labels_dir):
                if (base_name + '.txt') in files:
                    label_file = os.path.join(root, base_name + '.txt')
                    break

        labels = []
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        if orig_cls_id < len(b_class_names):
                            raw_name = b_class_names[orig_cls_id]
                            canonical_name = normalize_class_name(raw_name)
                            if canonical_name in unified_class_to_id:
                                unified_cls_id = unified_class_to_id[canonical_name]
                                labels.append((unified_cls_id, cx, cy, w, h))

        samples.append({
            'source': 'Dataset B',
            'image_path': img_path,
            'filename': filename,
            'labels': labels
        })

    return samples

def main():
    args = parse_args()

    # Use CANONICAL_CLASSES order for dataset configuration
    unified_classes = list(CANONICAL_CLASSES.keys())
    unified_class_to_id = {cls_name: i for i, cls_name in enumerate(unified_classes)}

    print("Canonical Class Mapping:")
    for idx, name in enumerate(unified_classes):
        print(f"  ID {idx}: {name}")

    b_class_names = load_dataset_b_classes(args.dataset_b)

    samples_a = process_dataset_a(args.dataset_a, args.min_area, unified_class_to_id)
    samples_b = process_dataset_b(args.dataset_b, b_class_names, unified_class_to_id)

    all_samples = samples_a + samples_b
    print(f"Total merged sample images collected: {len(all_samples)}")

    random.seed(args.seed)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.85)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    output_dir = os.path.abspath(args.output)
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    stats = {
        'train': {'images': 0, 'instances': {cls_name: 0 for cls_name in unified_classes}},
        'val': {'images': 0, 'instances': {cls_name: 0 for cls_name in unified_classes}}
    }

    def write_dataset_split(samples, img_dest_dir, lbl_dest_dir, split_key):
        for idx, sample in enumerate(samples):
            ext = os.path.splitext(sample['filename'])[1]
            unique_name = f"{sample['source'].replace(' ', '_').lower()}_{idx:06d}{ext}"

            dst_img_path = os.path.join(img_dest_dir, unique_name)
            shutil.copy2(sample['image_path'], dst_img_path)

            txt_name = os.path.splitext(unique_name)[0] + ".txt"
            dst_lbl_path = os.path.join(lbl_dest_dir, txt_name)

            with open(dst_lbl_path, 'w') as f:
                for cls_id, cx, cy, w, h in sample['labels']:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    cls_name = unified_classes[cls_id]
                    stats[split_key]['instances'][cls_name] += 1

            stats[split_key]['images'] += 1

    print("Copying files into train and val splits...")
    write_dataset_split(train_samples, train_img_dir, train_lbl_dir, 'train')
    write_dataset_split(val_samples, val_img_dir, val_lbl_dir, 'val')

    data_yaml_content = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(unified_classes)}
    }

    yaml_output_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_output_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"Generated dataset configuration file at '{yaml_output_path}'.")

    print("\n" + "=" * 65)
    print(f"{'Dataset Split & Summary':^65}")
    print("=" * 65)
    print(f"Total Train Images: {stats['train']['images']}")
    print(f"Total Val Images:   {stats['val']['images']}")
    print(f"Total Combined:     {len(all_samples)}")
    print("-" * 65)
    print(f"{'Class Name':<25} | {'Train Count':<12} | {'Val Count':<12} | {'Total':<10}")
    print("-" * 65)

    grand_total_instances = 0
    for cls_name in unified_classes:
        t_cnt = stats['train']['instances'][cls_name]
        v_cnt = stats['val']['instances'][cls_name]
        tot = t_cnt + v_cnt
        grand_total_instances += tot
        print(f"{cls_name:<25} | {t_cnt:<12} | {v_cnt:<12} | {tot:<10}")

    print("-" * 65)
    print(f"{'Total Instances':<25} | {sum(stats['train']['instances'].values()):<12} | {sum(stats['val']['instances'].values()):<12} | {grand_total_instances:<10}")
    print("=" * 65)

if __name__ == '__main__':
    main()
