import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def extract_and_convert(dataset_root, output_root, frame_skip=5, frame_skip_no_gun=50, train_split=0.8):
    """Extract frames and convert labels to YOLO format while limiting background frames."""
    output_root = Path(output_root)
    dataset_root = Path(dataset_root)

    for split in ["train", "val"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    categories = ["Handgun", "Machine_Gun", "No_Gun"]
    class_mapping = {"Handgun": 0, "Machine_Gun": 1}

    all_videos = []
    for category in categories:
        category_path = dataset_root / category
        if category_path.exists():
            video_folders = [f for f in category_path.iterdir() if f.is_dir()]
            all_videos.extend([(category, vf) for vf in video_folders])

    random.shuffle(all_videos)
    split_idx = int(len(all_videos) * train_split)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    print("=" * 60)
    print("EXTRACTING FRAMES AND CONVERTING LABELS")
    print("=" * 60)
    print("\nDataset Split:")
    print(f"   Training: {len(train_videos)} videos")
    print(f"   Validation: {len(val_videos)} videos\n")

    total_images = 0
    total_annotations = 0
    total_backgrounds = 0

    for split_name, video_list in [("train", train_videos), ("val", val_videos)]:
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} videos...")
        print(f"{'='*60}\n")

        for category, video_folder in tqdm(video_list, desc=f"{split_name}"):
            video_path = video_folder / "video.mp4"
            if not video_path.exists():
                continue

            label_json = video_folder / "label.json"
            label_data = {}
            if label_json.exists():
                with open(label_json, "r") as f:
                    label_data = json.load(f)

            annotations_by_frame = {}
            if "annotations" in label_data:
                for ann in label_data["annotations"]:
                    img_id = ann.get("image_id", 0)
                    annotations_by_frame.setdefault(img_id, []).append(ann)

            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_skip = frame_skip_no_gun if category == "No_Gun" else frame_skip
                if frame_idx % current_skip == 0:
                    json_image_id = frame_idx + 1
                    has_boxes = False
                    yolo_lines = []

                    if json_image_id in annotations_by_frame:
                        height, width = frame.shape[:2]
                        for ann in annotations_by_frame[json_image_id]:
                            bbox = ann.get("bbox", [0, 0, 0, 0])
                            if len(bbox) == 4:
                                x, y, w, h = bbox
                                if w > 0 and h > 0:
                                    x_center = max(0, min(1, (x + w / 2) / width))
                                    y_center = max(0, min(1, (y + h / 2) / height))
                                    norm_w = max(0, min(1, w / width))
                                    norm_h = max(0, min(1, h / height))
                                    class_id = class_mapping.get(category, 0)
                                    yolo_lines.append(
                                        f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                                    )
                                    has_boxes = True

                    should_save = False
                    if has_boxes:
                        should_save = True
                        total_annotations += len(yolo_lines)
                    elif category == "No_Gun":
                        should_save = True
                        total_backgrounds += 1
                    else:
                        if random.random() < 0.05:
                            should_save = True
                            total_backgrounds += 1

                    if should_save:
                        frame_name = f"{category}_{video_folder.name}_f{frame_idx:05d}"
                        img_path = output_root / split_name / "images" / f"{frame_name}.jpg"
                        label_path = output_root / split_name / "labels" / f"{frame_name}.txt"

                        cv2.imwrite(str(img_path), frame)
                        with open(label_path, "w") as f:
                            for line in yolo_lines:
                                f.write(line)
                        total_images += 1

                frame_idx += 1

            cap.release()

    print("\nEXTRACTION COMPLETE!")
    print(f"Total Saved Images: {total_images}")
    print(f"Total Bounding Boxes: {total_annotations}")
    if total_images > 0:
        pct = (total_backgrounds / total_images) * 100
        print(f"Total Background Images: {total_backgrounds} ({pct:.1f}% of dataset)")


def create_yaml(output_root):
    output_root = Path(output_root)

    config = {
        "path": str(output_root.absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": {0: "Handgun", 1: "Machine_Gun"},
    }

    yaml_path = output_root / "gun_detection.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("=" * 60)
    print("YAML CONFIGURATION CREATED")
    print("=" * 60)
    print(f"\nLocation: {yaml_path}\n")
    print("Contents:")
    print("-" * 40)
    with open(yaml_path, "r") as f:
        print(f.read())
    print("-" * 40)

    return yaml_path


def train_model(yaml_path, model_size, epochs, img_size, batch_size, checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING YOLO TRAINING")
    print("=" * 60)
    print("\nTraining Configuration:")
    print(f"   Model: {model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {img_size}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Config: {yaml_path}")
    print(f"   Checkpoint Dir: {checkpoint_dir}\n")

    last_checkpoint = checkpoint_dir / "gun_detection" / "weights" / "last.pt"

    if last_checkpoint.exists():
        print("FOUND EXISTING CHECKPOINT!")
        print(f"   Resuming from: {last_checkpoint}\n")
        model = YOLO(str(last_checkpoint))
        results = model.train(resume=True)
    else:
        print("Starting fresh training...\n")
        model = YOLO(f"{model_size}.pt")

        print("Starting training...\n")
        print("=" * 60)

        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=20,
            project=str(checkpoint_dir),
            name="gun_detection",
            save_period=5,
            device=device,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nBest model saved to:")
    print(f"   {checkpoint_dir}/gun_detection/weights/best.pt")
    print("\nResults folder:")
    print(f"   {checkpoint_dir}/gun_detection/")

    return results


def copy_best_model(checkpoint_dir, final_models_dir):
    checkpoint_dir = Path(checkpoint_dir)
    final_models_dir = Path(final_models_dir)
    final_models_dir.mkdir(parents=True, exist_ok=True)

    best_model = checkpoint_dir / "gun_detection" / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, final_models_dir / "gun_detection_best.pt")
        print("\nFINAL MODEL COPIED TO:")
        print(f"   {final_models_dir}/gun_detection_best.pt")


def validate_model(model_path):
    print("=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)

    best_model = YOLO(str(model_path))
    metrics = best_model.val()

    print("\nValidation Metrics:")
    print(f"\n   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

    print("\n   Per-Class AP50:")
    class_names = ["Handgun", "Machine_Gun"]
    for i, name in enumerate(class_names):
        print(f"      {name}: {metrics.box.ap50[i]:.4f}")


def test_on_video(model_path, video_path, output_dir, conf=0.25):
    print(f"Testing on: {Path(video_path).name}\n")
    model = YOLO(str(model_path))
    model.predict(
        source=str(video_path),
        save=True,
        conf=conf,
        project=str(output_dir),
        name="gun_detection",
        exist_ok=True,
    )
    print("\nTest complete!")
    print(f"Results saved in: {Path(output_dir) / 'gun_detection'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 gun detection locally")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_root", type=str, default="./outputs/yolo_dataset", help="Output dataset path")
    parser.add_argument("--prepare", action="store_true", help="Extract frames and build dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--val", action="store_true", help="Validate model")
    parser.add_argument("--test_video", type=str, default="", help="Path to sample video")

    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--frame_skip_no_gun", type=int, default=50)
    parser.add_argument("--train_split", type=float, default=0.7)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_size", type=str, default="yolov8s")
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--final_models_dir", type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./outputs/test_results")

    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        extract_and_convert(
            args.dataset_root,
            output_root,
            frame_skip=args.frame_skip,
            frame_skip_no_gun=args.frame_skip_no_gun,
            train_split=args.train_split,
        )

    yaml_path = create_yaml(output_root)

    if args.train:
        train_model(
            yaml_path,
            args.model_size,
            args.epochs,
            args.img_size,
            args.batch_size,
            args.checkpoint_dir,
            args.device,
        )
        copy_best_model(args.checkpoint_dir, args.final_models_dir)

    if args.val:
        best_model_path = Path(args.final_models_dir) / "gun_detection_best.pt"
        validate_model(best_model_path)

    if args.test_video:
        best_model_path = Path(args.final_models_dir) / "gun_detection_best.pt"
        test_on_video(best_model_path, args.test_video, args.results_dir)


if __name__ == "__main__":
    main()
