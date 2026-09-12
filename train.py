import argparse
from ultralytics import YOLO

def train_model(model_path, data_path, epochs, imgsz, batch):
    # Load the model
    model = YOLO(model_path)

    # Train the model
    results = model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch)

    # Print the results
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on a dataset')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset (data.yaml)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()

    train_model(args.model, args.data, args.epochs, args.imgsz, args.batch)
