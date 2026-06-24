from __future__ import annotations
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from it.akron.config import Config
from it.akron.dataset.dataset import PotholeDatasetManager


class YOLOSegmentationConfig:
    """Configurazione YOLOv8 segmentation presa dal notebook modello_YOLO.ipynb."""

    model_name = Config.YOLO_MODEL_NAME
    image_size = Config.YOLO_IMAGE_SIZE
    epochs = Config.YOLO_EPOCHS
    batch_size = Config.YOLO_BATCH_SIZE
    dataset_yaml = Config.YOLO_DATASET_YAML
    project_dir = Config.YOLO_PROJECT_DIR
    model_path = Config.YOLO_MODEL_PATH
    class_names = {0: "pothole"}


class YOLOSegmentationPipeline:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, config: type[YOLOSegmentationConfig] = YOLOSegmentationConfig) -> None:
        self.config = config
        Config.create_project_folders()

    def prepare_dataset(self) -> Path:
        if not Config.SPLIT_DATA_DIR.exists():
            PotholeDatasetManager().prepare()

        for split_name in ("train", "valid", "test"):
            self._copy_split_from_unet(split_name)
            self._create_labels_from_masks(split_name)
        return self._write_dataset_yaml()

    def train(self):
        from ultralytics import YOLO

        yaml_path = self.prepare_dataset()
        model = YOLO(self.config.model_name)
        results = model.train(
            data=str(yaml_path),
            imgsz=self.config.image_size,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            project=str(self.config.project_dir),
            name="train",
            exist_ok=True,
        )
        best_model = self.config.project_dir / "train" / "weights" / "best.pt"
        if best_model.exists():
            shutil.copy2(best_model, self.config.model_path)
        return results

    def validate(self) -> dict[str, float]:
        from ultralytics import YOLO

        if not self.config.dataset_yaml.exists():
            self.prepare_dataset()

        model_path = self.config.model_path if self.config.model_path.exists() else self.config.model_name
        model = YOLO(str(model_path))
        metrics = model.val(data=str(self.config.dataset_yaml), imgsz=self.config.image_size, split="test")
        precision = float(metrics.seg.mp)
        recall = float(metrics.seg.mr)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": float(metrics.seg.map50),
            "map50_95": float(metrics.seg.map),
        }
        results.update(self._evaluate_test_masks(model))
        Config.YOLO_METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    @staticmethod
    def result_to_mask(result, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if result.masks is None:
            return mask
        for result_mask in result.masks.data:
            raw_mask = result_mask.cpu().numpy()
            raw_mask = cv2.resize(raw_mask, (shape[1], shape[0]))
            mask = np.maximum(mask, raw_mask)
        return (mask > 0.5).astype(np.uint8)

    @staticmethod
    def label_to_mask(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        height, width = shape
        if not label_path.exists():
            return mask

        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = list(map(float, line.strip().split()))
            polygon = np.array(values[1:]).reshape(-1, 2)
            polygon[:, 0] *= width
            polygon[:, 1] *= height
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

    @staticmethod
    def dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        prediction = prediction.astype(bool)
        ground_truth = ground_truth.astype(bool)
        intersection = np.logical_and(prediction, ground_truth).sum()
        return float((2 * intersection) / (prediction.sum() + ground_truth.sum() + 1e-8))

    @staticmethod
    def pothole_postprocess(
        mask: np.ndarray,
        min_area: int = Config.POSTPROCESS_MIN_AREA,
        kernel_size: int = Config.POSTPROCESS_KERNEL_SIZE,
    ) -> np.ndarray:
        mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean[labels == label_id] = 1
        return clean

    def _copy_split_from_unet(self, split_name: str) -> None:
        source_dir = Config.SPLIT_DATA_DIR / split_name / "images"
        destination_dir = Config.DATA_DIR / "yolo" / "images" / split_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for old_file in destination_dir.glob("*.*"):
            old_file.unlink()
        for image_path in source_dir.iterdir():
            if image_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                shutil.copy2(image_path, destination_dir / image_path.name)

    def _create_labels_from_masks(self, split_name: str) -> None:
        masks_dir = Config.SPLIT_DATA_DIR / split_name / "masks"
        labels_dir = Config.DATA_DIR / "yolo" / "labels" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        for old_file in labels_dir.glob("*.txt"):
            old_file.unlink()

        for mask_path in masks_dir.iterdir():
            if mask_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            mask = (mask > 0).astype(np.uint8)
            height, width = mask.shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            label_path = labels_dir / mask_path.with_suffix(".txt").name

            with label_path.open("w", encoding="utf-8") as file:
                for contour in contours:
                    if cv2.contourArea(contour) < Config.POSTPROCESS_MIN_AREA:
                        continue

                    epsilon = 0.002 * cv2.arcLength(contour, closed=True)
                    polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
                    if len(polygon) < 3:
                        continue

                    normalized = []
                    for x, y in polygon:
                        normalized.append(float(x) / width)
                        normalized.append(float(y) / height)

                    file.write(f"0 {' '.join(map(str, normalized))}\n")

    def _write_dataset_yaml(self) -> Path:
        names = "\n".join(f"  {key}: {value}" for key, value in self.config.class_names.items())
        content = f"""path: {Config.DATA_DIR / "yolo"}

train: images/train
val: images/valid
test: images/test

names:
{names}
"""
        self.config.dataset_yaml.write_text(content.strip(), encoding="utf-8")
        return self.config.dataset_yaml

    def _evaluate_test_masks(self, model) -> dict[str, float]:
        image_dir = Config.DATA_DIR / "yolo" / "images" / "test"
        label_dir = Config.DATA_DIR / "yolo" / "labels" / "test"
        counts = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}

        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            result = model.predict(
                str(image_path),
                imgsz=self.config.image_size,
                verbose=False,
            )[0]

            prediction = self.result_to_mask(result, (height, width))
            prediction = self.pothole_postprocess(prediction)

            label_path = label_dir / image_path.with_suffix(".txt").name
            ground_truth = self.label_to_mask(label_path, (height, width))

            self._add_mask_counts(counts, prediction, ground_truth)

        return self._metrics_from_counts(counts)

    @staticmethod
    def _add_mask_counts(counts: dict[str, float], prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        prediction = prediction.astype(bool)
        ground_truth = ground_truth.astype(bool)
        counts["tp"] += np.logical_and(prediction, ground_truth).sum()
        counts["tn"] += np.logical_and(~prediction, ~ground_truth).sum()
        counts["fp"] += np.logical_and(prediction, ~ground_truth).sum()
        counts["fn"] += np.logical_and(~prediction, ground_truth).sum()

    @staticmethod
    def _metrics_from_counts(counts: dict[str, float]) -> dict[str, float]:
        tp = counts["tp"]
        tn = counts["tn"]
        fp = counts["fp"]
        fn = counts["fn"]
        eps = 1e-7
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        return {
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "balanced_accuracy": (recall + specificity) / 2,
        }
