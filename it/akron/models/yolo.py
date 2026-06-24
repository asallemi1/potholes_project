from __future__ import annotations
import json
import shutil
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from it.akron.config import Config


class YOLOSegmentationConfig:
    """Configurazione YOLOv8 segmentation presa dal notebook modello_YOLO.ipynb."""

    model_name = Config.YOLO_MODEL_NAME
    image_size = Config.YOLO_IMAGE_SIZE
    epochs = Config.YOLO_EPOCHS
    batch_size = Config.YOLO_BATCH_SIZE
    dataset_yaml = Config.YOLO_DATASET_YAML
    project_dir = Config.YOLO_PROJECT_DIR
    model_path = Config.YOLO_MODEL_PATH
    class_names = None


class YOLOSegmentationPipeline:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, config: type[YOLOSegmentationConfig] = YOLOSegmentationConfig) -> None:
        self.config = config
        Config.create_project_folders()

    def prepare_dataset(self) -> Path:
        self._copy_images("train")
        self._copy_images("valid")
        self._create_labels_from_coco("train")
        self._create_labels_from_coco("valid")
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

        model_path = self.config.model_path if self.config.model_path.exists() else self.config.model_name
        model = YOLO(str(model_path))
        metrics = model.val(data=str(self.config.dataset_yaml), imgsz=self.config.image_size)
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

    def _copy_images(self, split_name: str) -> None:
        source_dir = Config.RAW_DATA_DIR / split_name
        destination_dir = Config.DATA_DIR / "yolo" / "images" / split_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for old_file in destination_dir.glob("*.*"):
            old_file.unlink()
        for image_path in source_dir.iterdir():
            if image_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                shutil.copy2(image_path, destination_dir / image_path.name)

    def _create_labels_from_coco(self, split_name: str) -> None:
        source_dir = Config.RAW_DATA_DIR / split_name
        annotation_path = source_dir / "_annotations.coco.json"
        labels_dir = Config.DATA_DIR / "yolo" / "labels" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        for old_file in labels_dir.glob("*.txt"):
            old_file.unlink()

        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {category["id"]: index for index, category in enumerate(coco["categories"])}
        images = {image["id"]: image for image in coco["images"]}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        for image_id, annotations in annotations_by_image.items():
            image_info = images[image_id]
            width, height = image_info["width"], image_info["height"]
            label_path = labels_dir / Path(image_info["file_name"]).with_suffix(".txt").name
            with label_path.open("w", encoding="utf-8") as file:
                for annotation in annotations:
                    class_id = categories[annotation["category_id"]]
                    for segmentation in annotation.get("segmentation", []):
                        normalized = [
                            value / width if index % 2 == 0 else value / height
                            for index, value in enumerate(segmentation)
                        ]
                        file.write(f"{class_id} {' '.join(map(str, normalized))}\n")

    def _write_dataset_yaml(self) -> Path:
        class_names = self._class_names_from_coco()
        names = "\n".join(f"  {key}: {value}" for key, value in class_names.items())
        content = f"""path: {Config.DATA_DIR / "yolo"}

train: images/train
val: images/valid

names:
{names}
"""
        self.config.dataset_yaml.write_text(content.strip(), encoding="utf-8")
        return self.config.dataset_yaml

    @staticmethod
    def _class_names_from_coco() -> dict[int, str]:
        annotation_path = Config.RAW_DATA_DIR / "train" / "_annotations.coco.json"
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        return {
            index: category.get("name", f"class_{index}")
            for index, category in enumerate(coco["categories"])
        }
