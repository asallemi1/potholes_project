from pathlib import Path
import torch


class Config:
    """Configurazione unica del progetto, semplice da leggere e modificare."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw" / "Pothole_Segmentation_YOLOv8-1"
    SPLIT_DATA_DIR = DATA_DIR / "processed" / "dataset_unet_split"

    MODEL_DIR = PROJECT_ROOT / "saved_model"
    MODEL_PATH = MODEL_DIR / "best_unet_potholes.pt"

    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    PLOTS_DIR = ARTIFACTS_DIR / "plots"
    PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
    METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    HISTORY_PATH = ARTIFACTS_DIR / "history.json"
    SUMMARY_PATH = ARTIFACTS_DIR / "model_summary.txt"

    ROBOFLOW_WORKSPACE = "farzad"
    ROBOFLOW_PROJECT = "pothole_segmentation_yolov8-k6npi"
    ROBOFLOW_VERSION = 1
    ROBOFLOW_FORMAT = "coco-segmentation"

    SEED = 42
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 12
    EPOCHS = 80
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 12
    TEST_RATIO_FROM_VALID = 0.5
    AUGMENTATIONS_PER_IMAGE = 4
    THRESHOLD = 0.4
    BASE_CHANNELS = 32

    @staticmethod
    def device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_project_folders(cls) -> None:
        for folder in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR.parent,
            cls.SPLIT_DATA_DIR.parent,
            cls.MODEL_DIR,
            cls.ARTIFACTS_DIR,
            cls.PLOTS_DIR,
            cls.PREDICTIONS_DIR,
        ]:
            folder.mkdir(parents=True, exist_ok=True)
