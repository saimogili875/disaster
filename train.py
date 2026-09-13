import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on disaster aerial datasets")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs with no mAP improvement)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--export-onnx", action="store_true", help="Export best model to ONNX after training")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory")
    parser.add_argument("--name", type=str, default="train", help="Run name within project directory")
    return parser.parse_args()


def train_model(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Learning rate schedule for fine-tuning
        lr0=0.01,
        lrf=0.01,
        # Augmentation tuned for aerial/drone imagery
        mosaic=1.0,
        mixup=0.15,
        fliplr=0.5,
        flipud=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        perspective=0.001,
    )

    return model, results


def validate_model(model, args):
    print("\n" + "=" * 50)
    print("Running validation on best weights...")
    print("=" * 50)

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_path):
        best_model = YOLO(best_path)
        metrics = best_model.val(data=args.data, imgsz=args.imgsz)
        print(f"\nValidation Results:")
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
        return best_model
    else:
        print(f"Warning: best.pt not found at {best_path}")
        return model


def export_model(model, args):
    print("\nExporting best model to ONNX...")
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_path):
        export_model = YOLO(best_path)
        export_model.export(format="onnx", imgsz=args.imgsz)
        print("ONNX export complete.")
    else:
        print("Skipping ONNX export: best.pt not found.")


if __name__ == "__main__":
    args = parse_args()

    print(f"Base model:      {args.model}")
    print(f"Dataset:         {args.data}")
    print(f"Epochs:          {args.epochs}")
    print(f"Image size:      {args.imgsz}")
    print(f"Batch size:      {args.batch}")
    print(f"Early stopping:  {args.patience} epochs patience")
    print(f"Output:          {args.project}/{args.name}")

    model, results = train_model(args)
    best_model = validate_model(model, args)

    if args.export_onnx:
        export_model(best_model, args)

    print(f"\nTraining complete. Best weights at: {args.project}/{args.name}/weights/best.pt")
