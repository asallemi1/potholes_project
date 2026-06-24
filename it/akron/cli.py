import argparse

from it.akron.config import Config


class AkronPotholeCli:
    def run(self, argv: list[str] | None = None) -> None:
        args = self._build_parser().parse_args(argv)
        command = getattr(self, f"_run_{args.command}")
        command(args)

    def _run_download(self, args: argparse.Namespace) -> None:
        from it.akron.dataset.dataset import PotholeDatasetManager

        location = PotholeDatasetManager().download_from_roboflow(args.api_key)
        print(f"Dataset scaricato in: {location}")

    def _run_prepare(self, args: argparse.Namespace) -> None:
        from it.akron.dataset.dataset import PotholeDatasetManager

        stats = PotholeDatasetManager().prepare()
        print("Split completato.")
        for key, value in stats.items():
            print(f"{key}: {value}")

    def _run_train(self, args: argparse.Namespace) -> None:
        from it.akron.training.trainer import PotholeTrainer

        history = PotholeTrainer().train()
        print("Training terminato.")
        print(f"Epoche registrate: {len(history['train_loss'])}")
        print(f"Best model: {Config.MODEL_PATH}")

    def _run_evaluate(self, args: argparse.Namespace) -> None:
        from it.akron.training.trainer import PotholeTrainer

        metrics = PotholeTrainer().evaluate(args.threshold)
        print("Metriche sul test set")
        for key, value in metrics.items():
            print(f"{key}: {value:.5f}")

    def _run_predict(self, args: argparse.Namespace) -> None:
        from it.akron.training.trainer import PotholeTrainer

        PotholeTrainer().save_predictions(count=args.count, threshold=args.threshold)
        print(f"Predizioni salvate in: {Config.PREDICTIONS_DIR}")

    def _run_yolo_prepare(self, args: argparse.Namespace) -> None:
        from it.akron.models.yolo import YOLOSegmentationPipeline

        yaml_path = YOLOSegmentationPipeline().prepare_dataset()
        print(f"Dataset YOLO preparato: {yaml_path}")

    def _run_yolo_train(self, args: argparse.Namespace) -> None:
        from it.akron.models.yolo import YOLOSegmentationPipeline

        YOLOSegmentationPipeline().train()
        print(f"Modello YOLO salvato in: {Config.YOLO_MODEL_PATH}")

    def _run_yolo_evaluate(self, args: argparse.Namespace) -> None:
        from it.akron.models.yolo import YOLOSegmentationPipeline

        metrics = YOLOSegmentationPipeline().validate()
        print("Metriche YOLOv8 segmentation")
        for key, value in metrics.items():
            print(f"{key}: {value:.5f}")

    def _run_api(self, args: argparse.Namespace) -> None:
        from it.akron.api.app import create_app

        app = create_app()
        app.run(host=args.host, port=args.port, debug=args.debug)

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="akron-potholes")
        subparsers = parser.add_subparsers(dest="command", required=True)

        download = subparsers.add_parser("download")
        download.add_argument("--api-key")

        subparsers.add_parser("prepare")

        subparsers.add_parser("train")

        evaluate = subparsers.add_parser("evaluate")
        evaluate.add_argument("--threshold", type=float, default=0.5)

        predict = subparsers.add_parser("predict")
        predict.add_argument("--threshold", type=float, default=Config.THRESHOLD)
        predict.add_argument("--count", type=int, default=5)

        subparsers.add_parser("yolo_prepare")
        subparsers.add_parser("yolo_train")
        subparsers.add_parser("yolo_evaluate")

        api = subparsers.add_parser("api")
        api.add_argument("--host", default="127.0.0.1")
        api.add_argument("--port", type=int, default=5000)
        api.add_argument("--debug", action="store_true")

        return parser


def main() -> None:
    AkronPotholeCli().run()


if __name__ == "__main__":
    main()
