from __future__ import annotations
import json
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from it.akron.config import Config
from it.akron.dataset.dataset import PotholeDatasetManager, PotholeSegmentationDataset
from it.akron.models.model import ModelSummary, UNet
from it.akron.models.yolo import YOLOSegmentationPipeline


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probabilities * targets).sum(dim=1)
        union = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return 1.0 - dice.median()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.4 * self.bce(logits, targets) + 0.6 * self.dice(logits, targets)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, save_path) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, valid_loss: float, model: nn.Module) -> None:
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Nuovo best model salvato. Valid loss: {valid_loss:.5f}")
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True

class PotholeTrainer:
    """Classe principale: addestra, salva, carica, valuta e crea la dashboard."""

    def __init__(self) -> None:
        Config.create_project_folders()
        self.device = Config.device()

    def ensure_model(self) -> str:
        if Config.MODEL_PATH.exists():
            return "loaded"
        if not Config.SPLIT_DATA_DIR.exists():
            PotholeDatasetManager().prepare()
        self.train()
        return "trained"

    def train(self) -> dict[str, list[float]]:
        self._set_seed()
        train_dataset = PotholeSegmentationDataset("train")
        valid_dataset = PotholeSegmentationDataset("valid")
        train_loader = self._loader(train_dataset, shuffle=True)
        valid_loader = self._loader(valid_dataset, shuffle=False)

        model = self.load_model(load_weights=False)
        criterion = BCEDiceLoss(self._pos_weight(train_dataset))
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        history = self._empty_history()
        early_stopping = EarlyStopping(
            patience=Config.PATIENCE,
            min_delta=1e-4,
            save_path=Config.MODEL_PATH,
        )

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss, train_metrics = self._run_epoch(model, train_loader, criterion, optimizer)
            valid_loss, valid_metrics = self._run_epoch(model, valid_loader, criterion)
            scheduler.step(valid_loss)
            self._update_history(history, train_loss, valid_loss, train_metrics, valid_metrics, optimizer)

            print(
                f"Epoch {epoch:03d}/{Config.EPOCHS} | "
                f"train loss {train_loss:.5f} | valid loss {valid_loss:.5f} | "
                f"valid dice {valid_metrics['dice']:.5f}"
            )

            early_stopping.step(valid_loss, model)
            if early_stopping.should_stop:
                print("Early stopping attivato.")
                break

        self._save_json(Config.HISTORY_PATH, history)
        self.save_training_plots(history)
        self.save_model_summary()
        return history

    def evaluate(self, threshold: float = Config.THRESHOLD) -> dict[str, float]:
        test_dataset = PotholeSegmentationDataset("test")
        test_loader = self._loader(test_dataset, shuffle=False)
        model = self.load_model()
        criterion = BCEDiceLoss(self._pos_weight(test_dataset))
        _, metrics, counts = self._run_epoch(model, test_loader, criterion, threshold=threshold, return_counts=True)
        metrics = {**metrics, "threshold": threshold, **counts}
        self._save_json(Config.METRICS_PATH, metrics)
        self.save_confusion_matrix(counts, normalize=True)
        self.save_confusion_matrix(counts, normalize=False)
        return metrics

    def refresh_dashboard_artifacts(self) -> dict[str, object]:
        self.ensure_model()
        history = self._read_json(Config.HISTORY_PATH)
        if history:
            self.save_training_plots(history)
        metrics = self.evaluate(Config.THRESHOLD)
        self.save_predictions()
        summary = self.save_model_summary()
        return {"metrics": metrics, "summary": summary}

    def load_model(self, load_weights: bool = True) -> UNet:
        model = UNet(base_channels=Config.BASE_CHANNELS).to(self.device)
        if load_weights and Config.MODEL_PATH.exists():
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=self.device))
        model.eval()
        return model

    def save_model_summary(self) -> str:
        summary = ModelSummary().build(self.load_model(load_weights=Config.MODEL_PATH.exists()))
        Config.SUMMARY_PATH.write_text(summary, encoding="utf-8")
        return summary

    def save_training_plots(self, history: dict[str, list[float]]) -> None:
        plot_pairs = [
            ("train_loss", "valid_loss", "loss.png", "Loss"),
            ("train_dice", "valid_dice", "dice.png", "Dice"),
            ("train_iou", "valid_iou", "iou.png", "IoU"),
            ("train_balanced_accuracy", "valid_balanced_accuracy", "balanced_accuracy.png", "Balanced accuracy"),
        ]
        for train_key, valid_key, filename, ylabel in plot_pairs:
            self._plot_lines(history, [train_key, valid_key], Config.PLOTS_DIR / filename, ylabel)
        self._plot_lines(history, ["lr"], Config.PLOTS_DIR / "learning_rate.png", "Learning rate")

    def save_confusion_matrix(self, counts: dict[str, float], normalize: bool) -> None:
        matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=np.float64)
        filename = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
        title = "Confusion Matrix normalizzata" if normalize else "Confusion Matrix"
        shown = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-7) if normalize else matrix
        fmt = ".3f" if normalize else ".0f"

        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(shown)
        fig.colorbar(image, ax=axis)
        axis.set_title(title)
        axis.set_xticks([0, 1], ["Background", "Pothole"])
        axis.set_yticks([0, 1], ["Background", "Pothole"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Ground Truth")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, format(shown[row, column], fmt), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(Config.PLOTS_DIR / filename, dpi=150)
        plt.close(fig)

    def save_predictions(self, count: int = 5, threshold: float = Config.THRESHOLD, min_area: int = 20) -> None:
        for old_file in Config.PREDICTIONS_DIR.glob("*.png"):
            old_file.unlink()

        model = self.load_model()
        dataset = PotholeSegmentationDataset("test")
        indices = random.sample(
            range(len(dataset)),
            min(count, len(dataset))
        )

        for output_index, dataset_index in enumerate(indices, start=1):
            image, mask = dataset[dataset_index]
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(self.device))
                probability_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prediction = (probability_map > threshold).astype(np.uint8)
            prediction = YOLOSegmentationPipeline.pothole_postprocess(prediction, min_area=min_area)
            image_np = self._denormalize(image)
            mask_np = mask[0].numpy()
            boxes = self._boxes(prediction, min_area)
            self._save_prediction_figure(output_index, image_np, mask_np, prediction, boxes)

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        threshold: float = Config.THRESHOLD,
        return_counts: bool = False,
    ):
        is_training = optimizer is not None
        model.train(is_training)
        total_loss = 0.0
        counts = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}

        for images, masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            if is_training:
                optimizer.zero_grad()
            with torch.enable_grad() if is_training else torch.no_grad():
                logits = model(images)
                loss = criterion(logits, masks)
            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            self._add_counts(counts, logits.detach(), masks, threshold)

        metrics = self._metrics(counts)
        loss = total_loss / len(loader.dataset)
        return (loss, metrics, counts) if return_counts else (loss, metrics)

    def _pos_weight(self, dataset: PotholeSegmentationDataset) -> torch.Tensor:
        positive = 0.0
        negative = 0.0
        for _, mask in dataset:
            positive += mask.sum().item()
            negative += mask.numel() - mask.sum().item()
        return torch.tensor([min(negative / (positive + 1e-7), 10.0)], dtype=torch.float32).to(self.device)

    def _loader(self, dataset: PotholeSegmentationDataset, shuffle: bool) -> DataLoader:
        workers = 2 if torch.cuda.is_available() else 0
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle, num_workers=workers)

    @staticmethod
    def _add_counts(counts: dict[str, float], logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> None:
        probability_maps = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        postprocessed_predictions = []
        for probability_map in probability_maps:
            prediction = (probability_map > threshold).astype(np.uint8)
            prediction = YOLOSegmentationPipeline.pothole_postprocess(prediction)
            postprocessed_predictions.append(prediction)

        predictions = torch.tensor(
            np.stack(postprocessed_predictions),
            device=masks.device,
            dtype=torch.bool,
        ).unsqueeze(1)
        targets = masks > 0.5
        counts["tp"] += (predictions & targets).sum().item()
        counts["tn"] += (~predictions & ~targets).sum().item()
        counts["fp"] += (predictions & ~targets).sum().item()
        counts["fn"] += (~predictions & targets).sum().item()

    @staticmethod
    def _metrics(counts: dict[str, float]) -> dict[str, float]:
        tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
        eps = 1e-7
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        return {
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "iou": tp / (tp + fp + fn + eps),
            "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
        }

    @staticmethod
    def _empty_history() -> dict[str, list[float]]:
        return {
            "train_loss": [],
            "valid_loss": [],
            "train_dice": [],
            "valid_dice": [],
            "train_iou": [],
            "valid_iou": [],
            "train_balanced_accuracy": [],
            "valid_balanced_accuracy": [],
            "lr": [],
        }

    @staticmethod
    def _update_history(history, train_loss, valid_loss, train_metrics, valid_metrics, optimizer) -> None:
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        for metric in ["dice", "iou", "balanced_accuracy"]:
            history[f"train_{metric}"].append(train_metrics[metric])
            history[f"valid_{metric}"].append(valid_metrics[metric])
        history["lr"].append(optimizer.param_groups[0]["lr"])

    @staticmethod
    def _set_seed() -> None:
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)

    @staticmethod
    def _plot_lines(history: dict[str, list[float]], keys: list[str], output_path, ylabel: str) -> None:
        if not all(key in history for key in keys):
            return
        fig, axis = plt.subplots(figsize=(8, 5))
        for key in keys:
            axis.plot(history[key], label=key)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    @staticmethod
    def _boxes(prediction_mask: np.ndarray, min_area: int) -> list[dict[str, float]]:
        prediction_mask = (prediction_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(prediction_mask, connectivity=8)
        boxes = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            boxes.append(
                {
                    "x": int(stats[label_id, cv2.CC_STAT_LEFT]),
                    "y": int(stats[label_id, cv2.CC_STAT_TOP]),
                    "w": int(stats[label_id, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[label_id, cv2.CC_STAT_HEIGHT]),
                    #"confidence": 1.0,
                }
            )
        return boxes

    @staticmethod
    def _save_prediction_figure(index: int, image_np, mask_np, prediction, boxes) -> None:
        fig = plt.figure(figsize=(16, 4))
        for position, title, data, cmap in [
            (1, "Immagine", image_np, None),
            (2, "Ground truth", mask_np, "gray"),
            (3, "Predizione mask", prediction, "gray"),
        ]:
            axis = fig.add_subplot(1, 4, position)
            axis.imshow(data, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")

        axis = fig.add_subplot(1, 4, 4)
        axis.imshow(image_np)
        axis.set_title("Bounding box predette")
        axis.axis("off")
        for box in boxes:
            axis.add_patch(Rectangle((box["x"], box["y"]), box["w"], box["h"], fill=False, edgecolor="red", linewidth=2))
            #axis.text(box["x"], max(box["y"] - 5, 0), f"{box['confidence'] * 100:.1f}%", fontsize=9)

        fig.tight_layout()
        fig.savefig(Config.PREDICTIONS_DIR / f"prediction_{index:02d}.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def _save_json(path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

