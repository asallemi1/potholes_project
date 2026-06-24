from __future__ import annotations
import json
import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from it.akron.config import Config


class PotholeDatasetManager:
    """Scarica una volta il dataset, crea maschere e prepara train/valid/test."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def download_from_roboflow(self, api_key: str | None = None) -> Path:
        api_key = "vhzjMQXcncYPLEbfG8Pd"
        key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError("Imposta ROBOFLOW_API_KEY solo per il primo download.")

        from roboflow import Roboflow

        Config.create_project_folders()
        roboflow = Roboflow(api_key=key)
        project = roboflow.workspace(Config.ROBOFLOW_WORKSPACE).project(Config.ROBOFLOW_PROJECT)
        dataset = project.version(Config.ROBOFLOW_VERSION).download(Config.ROBOFLOW_FORMAT)

        source = Path(dataset.location)
        if Config.RAW_DATA_DIR.exists():
            shutil.rmtree(Config.RAW_DATA_DIR)
        shutil.copytree(source, Config.RAW_DATA_DIR)
        return Config.RAW_DATA_DIR

    def prepare(self) -> dict[str, int]:
        Config.create_project_folders()
        self._create_masks_from_coco(Config.RAW_DATA_DIR / "train")
        self._create_masks_from_coco(Config.RAW_DATA_DIR / "valid")

        if Config.SPLIT_DATA_DIR.exists():
            shutil.rmtree(Config.SPLIT_DATA_DIR)

        train_pairs = self.find_pairs(Config.RAW_DATA_DIR / "train", Config.RAW_DATA_DIR / "train" / "masks")
        valid_pairs = self.find_pairs(Config.RAW_DATA_DIR / "valid", Config.RAW_DATA_DIR / "valid" / "masks")

        random.shuffle(valid_pairs)
        test_count = int(len(valid_pairs) * Config.TEST_RATIO_FROM_VALID)
        test_pairs = valid_pairs[:test_count]
        new_valid_pairs = valid_pairs[test_count:]

        self._save_train_with_augmentation(train_pairs)
        self._copy_pairs(new_valid_pairs, "valid")
        self._copy_pairs(test_pairs, "test")

        return {
            "train_original": len(train_pairs),
            "train_total": len(train_pairs) * (1 + Config.AUGMENTATIONS_PER_IMAGE),
            "valid": len(new_valid_pairs),
            "test": len(test_pairs),
        }

    def find_pairs(self, images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
        images = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        masks = {
            path.stem: path
            for path in masks_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
        }
        pairs = [(images[stem], masks[stem]) for stem in sorted(set(images) & set(masks))]
        if not pairs:
            raise RuntimeError(f"Nessuna coppia immagine-maschera trovata in {images_dir} e {masks_dir}.")
        return pairs

    def _create_masks_from_coco(self, split_dir: Path) -> None:
        annotation_file = split_dir / "_annotations.coco.json"
        masks_dir = split_dir / "masks"
        if masks_dir.exists() and any(masks_dir.glob("*.png")):
            return
        if not annotation_file.exists():
            raise RuntimeError(f"File annotazioni non trovato: {annotation_file}")

        from pycocotools import mask as mask_utils

        masks_dir.mkdir(parents=True, exist_ok=True)
        coco = json.loads(annotation_file.read_text(encoding="utf-8"))
        images = {image["id"]: image for image in coco["images"]}
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in coco["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image_id, image_info in images.items():
            height, width = image_info["height"], image_info["width"]
            mask = np.zeros((height, width), dtype=np.uint8)
            for annotation in annotations_by_image.get(image_id, []):
                segmentation = annotation["segmentation"]
                if isinstance(segmentation, list):
                    rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
                else:
                    rle = segmentation
                decoded = mask_utils.decode(rle)
                mask[decoded > 0] = 255
            cv2.imwrite(str(masks_dir / Path(image_info["file_name"]).with_suffix(".png").name), mask)

    def _save_train_with_augmentation(self, pairs: list[tuple[Path, Path]]) -> None:
        import albumentations as A

        images_dir = Config.SPLIT_DATA_DIR / "train" / "images"
        masks_dir = Config.SPLIT_DATA_DIR / "train" / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(1.0, 1.2),
                    rotate=(-15, 15),
                    shear={"x": (-5, 5), "y": (-5, 5)},
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.0, p=1.0),
                A.RandomGamma(gamma_limit=(75, 125), p=1.0),
            ]
        )

        for image_path, mask_path in pairs:
            shutil.copy2(image_path, images_dir / image_path.name)
            shutil.copy2(mask_path, masks_dir / mask_path.name)
            image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            for index in range(1, Config.AUGMENTATIONS_PER_IMAGE + 1):
                augmented = transform(image=image, mask=mask)
                stem = f"{image_path.stem}_aug{index}"
                Image.fromarray(augmented["image"]).save(images_dir / f"{stem}{image_path.suffix}")
                Image.fromarray((augmented["mask"] > 0).astype(np.uint8) * 255).save(masks_dir / f"{stem}.png")

    def _copy_pairs(self, pairs: list[tuple[Path, Path]], split_name: str) -> None:
        images_dir = Config.SPLIT_DATA_DIR / split_name / "images"
        masks_dir = Config.SPLIT_DATA_DIR / split_name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for image_path, mask_path in pairs:
            shutil.copy2(image_path, images_dir / image_path.name)
            shutil.copy2(mask_path, masks_dir / mask_path.name)


class PotholeSegmentationDataset(Dataset):
    """Dataset PyTorch per immagini e maschere gia salvate nel progetto."""

    def __init__(self, split_name: str) -> None:
        self.images_dir = Config.SPLIT_DATA_DIR / split_name / "images"
        self.masks_dir = Config.SPLIT_DATA_DIR / split_name / "masks"
        self.pairs = PotholeDatasetManager().find_pairs(self.images_dir, self.masks_dir)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB").resize(Config.IMAGE_SIZE, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(Config.IMAGE_SIZE, Image.NEAREST)

        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std
        image_array = np.transpose(image_array, (2, 0, 1))

        mask_array = (np.array(mask) > 0).astype(np.float32)
        mask_array = np.expand_dims(mask_array, axis=0)

        return torch.tensor(image_array), torch.tensor(mask_array)
