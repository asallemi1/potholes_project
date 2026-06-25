from __future__ import annotations
import json
from flask import Flask, redirect, render_template_string, send_from_directory, url_for
from it.akron.config import Config


def create_app() -> Flask:
    from it.akron.training.trainer import PotholeTrainer

    app = Flask(__name__)
    trainer = PotholeTrainer()
    status = "not ready"
    error = ""
    try:
        if Config.MODEL_PATH.exists():
            status = "loaded"
        else:
            status = "missing model"
            error = f"Model file not found: {Config.MODEL_PATH}"
        #status = trainer.ensure_model()
        #trainer.refresh_dashboard_artifacts()
    except Exception as exc:
        error = str(exc)

    @app.get("/")
    def dashboard():
        metrics = _read_json(Config.METRICS_PATH)
        yolo_metrics = _read_json(Config.YOLO_METRICS_PATH)
        summary = Config.SUMMARY_PATH.read_text(encoding="utf-8") if Config.SUMMARY_PATH.exists() else ""
        plots = sorted(path.name for path in Config.PLOTS_DIR.glob("*.png"))
        predictions = sorted(path.name for path in Config.PREDICTIONS_DIR.glob("*.png"))
        return render_template_string(
            HTML,
            status=status,
            error=error,
            metrics=metrics,
            yolo_metrics=yolo_metrics,
            summary=summary,
            plots=plots,
            predictions=predictions,
        )

    @app.post("/refresh")
    def refresh():
        trainer.refresh_dashboard_artifacts()
        return redirect(url_for("dashboard"))
    
    @app.post("/refresh-predictions")
    def refresh_predictions():
        trainer.save_predictions(count=5, threshold=Config.THRESHOLD)
        return redirect(url_for("dashboard"))

    @app.get("/plots/<path:filename>")
    def plots(filename: str):
        return send_from_directory(Config.PLOTS_DIR, filename)

    @app.get("/predictions/<path:filename>")
    def predictions(filename: str):
        return send_from_directory(Config.PREDICTIONS_DIR, filename)

    return app


def _read_json(path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


HTML = """
<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Akron Pothole Segmentation</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f5f7fa; color: #17202a; }
    header { padding: 24px 32px; background: #17324d; color: white; }
    main { max-width: 1200px; margin: 0 auto; padding: 24px 32px; }
    section { margin-bottom: 32px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }
    .card { background: white; border: 1px solid #d8dee6; border-radius: 6px; padding: 16px; }
    .metric { display: flex; justify-content: space-between; gap: 16px; padding: 8px 0; border-bottom: 1px solid #edf0f3; }
    img { width: 100%; background: white; border: 1px solid #d8dee6; }
    pre { overflow: auto; background: #0f1720; color: #e6edf3; padding: 16px; border-radius: 6px; }
    button { background: #2166a5; color: white; border: 0; border-radius: 4px; padding: 10px 14px; cursor: pointer; }
  </style>
</head>
<body>
  <header>
    <h1>Akron Pothole Segmentation</h1>
    <p>Modello: {{ status }}</p>
    {% if error %}<p>{{ error }}</p>{% endif %}
  </header>
  <main>
    <section>
      <form method="post" action="{{ url_for('refresh') }}">
        <button type="submit">Cambia threshold</button>
      </form>
    </section>

    <section>
      <form method="post" action="{{ url_for('refresh_predictions') }}">
        <button type="submit">Mostra altre predizioni</button>
      </form>
    </section>

    <section>
      <h2>Metriche</h2>
      <div class="card">
        {% for key, value in metrics.items() %}
          <div class="metric"><strong>{{ key }}</strong><span>{{ "%.5f"|format(value) if value is number else value }}</span></div>
        {% else %}
          <p>Nessuna metrica disponibile.</p>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>Metriche YOLOv8</h2>
      <div class="card">
        {% for key, value in yolo_metrics.items() %}
          <div class="metric"><strong>{{ key }}</strong><span>{{ "%.5f"|format(value) if value is number else value }}</span></div>
        {% else %}
          <p>Nessuna metrica YOLO disponibile. Esegui <code>akron-potholes yolo_evaluate</code>.</p>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>Grafici e Confusion Matrix</h2>
      <div class="grid">
        {% for plot in plots %}
          <div class="card">
            <h3>{{ plot }}</h3>
            <img src="{{ url_for('plots', filename=plot) }}" alt="{{ plot }}">
          </div>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>Predizioni con Bounding Box</h2>
      <div class="grid">
        {% for prediction in predictions %}
          <div class="card">
            <h3>{{ prediction }}</h3>
            <img src="{{ url_for('predictions', filename=prediction) }}" alt="{{ prediction }}">
          </div>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>Model Summary</h2>
      <pre>{{ summary }}</pre>
    </section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    create_app().run(host='0.0.0.0', port=5000, debug=True)
