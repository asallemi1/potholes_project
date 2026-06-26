# Potholes Project

Progetto Python per la segmentazione automatica delle buche stradali a partire da immagini. La repository contiene una pipeline completa per scaricare il dataset da Roboflow, prepararlo in formato train/validation/test, addestrare un modello U-Net in PyTorch, valutare le metriche, generare predizioni visuali e consultare i risultati tramite una piccola dashboard Flask.

Nel progetto e' presente anche una pipeline basata su YOLO segmentation con Ultralytics, utile per confrontare le prestazioni del modello U-Net con un modello YOLO pre-addestrato e riaddestrato sullo stesso dataset.

## Funzionalita principali

- Download del dataset da Roboflow tramite API key.
- Conversione delle annotazioni COCO in maschere binarie.
- Split automatico in `train`, `valid` e `test`.
- Data augmentation sulle immagini di training.
- Training di un modello U-Net per segmentazione semantica.
- Training di YOLO segmentation.
- Valutazione con metriche Dice, IoU, precision, recall, balanced accuracy.
- Salvataggio automatico di modello, metriche, grafici e predizioni.
- Dashboard Flask per visualizzare metriche, confusion matrix, curve di training e predizioni.

## Struttura del progetto

```text
potholes_project/
|-- artifacts/
|   |-- history.json                 # Storico del training U-Net
|   |-- metrics.json                 # Metriche U-Net sul test set
|   |-- yolo_metrics.json            # Metriche YOLO sul test set
|   |-- model_summary.txt            # Riepilogo architettura U-Net
|   |-- plots/                       # Grafici di training e confusion matrix
|   |-- predictions/                 # Predizioni salvate come immagini PNG
|   `-- yolo_runs/                   # Output generati da Ultralytics YOLO
|-- data/
|   |-- raw/                         # Dataset Roboflow scaricato in formato COCO
|   |-- processed/                   # Dataset preparato per U-Net
|   `-- yolo/                        # Dataset convertito in formato YOLO segmentation
|-- it/
|   `-- akron/
|       |-- api/
|       |   `-- app.py               # Dashboard Flask
|       |-- dataset/
|       |   `-- dataset.py           # Download, maschere, split e augmentation
|       |-- models/
|       |   |-- model.py             # Architettura U-Net
|       |   `-- yolo.py              # Pipeline YOLO segmentation
|       |-- training/
|       |   `-- trainer.py           # Training, evaluation, plotting e prediction
|       |-- cli.py                   # Entry point CLI del progetto
|       `-- config.py                # Configurazione centralizzata
|-- saved_model/
|   |-- best_unet_potholes.pt        # Miglior modello U-Net salvato
|   `-- yolo/best.pt                 # Miglior modello YOLO salvato
|-- requirements.txt                 # Dipendenze Python
|-- Dockerfile                       # Bozza container Docker
|-- yolo11n.pt                       # Peso YOLO pre-addestrato
`-- yolov8n-seg.pt                   # Peso YOLO segmentation pre-addestrato
```

## Requisiti

- Python 3.11 consigliato.
- `pip` aggiornato.
- Connessione internet per il primo download del dataset e per eventuali pesi/model assets non ancora presenti.
- API key Roboflow per scaricare il dataset.
- GPU NVIDIA opzionale ma consigliata per velocizzare il training. Se CUDA non e' disponibile, il progetto usa automaticamente la CPU.

## Installazione

Clona la repository e spostati nella cartella del progetto:

```bash
git clone <URL_REPOSITORY>
cd potholes_project
```

Crea un ambiente virtuale:

```bash
python -m venv .venv
```

Attivalo su Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Oppure su Linux/macOS:

```bash
source .venv/bin/activate
```

Installa le dipendenze:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Primo utilizzo: download dataset e training U-Net

La pipeline principale usa il dataset Roboflow configurato in `it/akron/config.py`:

- workspace: `farzad`
- project: `pothole_segmentation_yolov8-k6npi`
- version: `1`
- format: `coco-segmentation`

Imposta la API key Roboflow.

Su Windows PowerShell:

```powershell
$env:ROBOFLOW_API_KEY="la_tua_api_key"
```

Su Linux/macOS:

```bash
export ROBOFLOW_API_KEY="la_tua_api_key"
```

Scarica il dataset:

```bash
python -m it.akron.cli download
```

In alternativa puoi passare la chiave direttamente da CLI:

```bash
python -m it.akron.cli download --api-key la_tua_api_key
```

Prepara il dataset per U-Net:

```bash
python -m it.akron.cli prepare
```

Questo comando:

- crea le maschere binarie dalle annotazioni COCO;
- divide i dati in `train`, `valid` e `test`;
- applica augmentation al training set;
- salva il dataset finale in `data/processed/dataset_unet_split/`.

Avvia il training U-Net:

```bash
python -m it.akron.cli train
```

Il training salva automaticamente:

- il miglior modello in `saved_model/best_unet_potholes.pt`;
- lo storico in `artifacts/history.json`;
- i grafici in `artifacts/plots/`;
- il riepilogo del modello in `artifacts/model_summary.txt`.

Valuta il modello sul test set:

```bash
python -m it.akron.cli evaluate
```

Puoi cambiare la soglia di segmentazione:

```bash
python -m it.akron.cli evaluate --threshold 0.4
```

Genera immagini di predizione:

```bash
python -m it.akron.cli predict --count 5
```

Le immagini vengono salvate in `artifacts/predictions/`.

## Pipeline YOLO segmentation

La pipeline YOLO usa come base `yolov8n-seg.pt` e converte il dataset preparato per U-Net nel formato richiesto da Ultralytics.

Prepara il dataset YOLO:

```bash
python -m it.akron.cli yolo_prepare
```

Avvia il training YOLO:

```bash
python -m it.akron.cli yolo_train
```

Il miglior peso viene copiato in:

```text
saved_model/yolo/best.pt
```

Valuta YOLO sul test set:

```bash
python -m it.akron.cli yolo_evaluate
```

Le metriche vengono salvate in `artifacts/yolo_metrics.json`.

## Dashboard Flask

Dopo aver trainato o valutato il modello, avvia la dashboard:

```bash
python -m it.akron.cli api
```

Apri il browser su:

```text
http://127.0.0.1:5000
```

Per esporre l'app su tutte le interfacce:

```bash
python -m it.akron.cli api --host 0.0.0.0 --port 5000
```

La dashboard mostra:

- stato del modello;
- metriche U-Net;
- metriche YOLO, se disponibili;
- grafici di training;
- confusion matrix;
- predizioni con maschere e bounding box;
- summary dell'architettura U-Net.

## Comandi CLI disponibili

```bash
python -m it.akron.cli download [--api-key API_KEY]
python -m it.akron.cli prepare
python -m it.akron.cli train
python -m it.akron.cli evaluate [--threshold 0.4]
python -m it.akron.cli predict [--threshold 0.4] [--count 5]
python -m it.akron.cli yolo_prepare
python -m it.akron.cli yolo_train
python -m it.akron.cli yolo_evaluate
python -m it.akron.cli api [--host 127.0.0.1] [--port 5000] [--debug]
```

## Configurazione

I parametri principali si trovano in `it/akron/config.py`.

Valori rilevanti per U-Net:

```python
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 12
EPOCHS = 80
LEARNING_RATE = 1e-3
PATIENCE = 12
THRESHOLD = 0.4
BASE_CHANNELS = 32
```

Valori rilevanti per YOLO:

```python
YOLO_MODEL_NAME = "yolov8n-seg.pt"
YOLO_IMAGE_SIZE = 256
YOLO_EPOCHS = 15
YOLO_BATCH_SIZE = 12
```

Se il training esaurisce la memoria della GPU, prova a ridurre `BATCH_SIZE` o `YOLO_BATCH_SIZE`.

## Output generati

| Percorso | Contenuto |
| --- | --- |
| `data/raw/` | Dataset scaricato da Roboflow |
| `data/processed/dataset_unet_split/` | Dataset U-Net con immagini e maschere |
| `data/yolo/` | Dataset YOLO con immagini, label e `dataset.yaml` |
| `saved_model/best_unet_potholes.pt` | Miglior checkpoint U-Net |
| `saved_model/yolo/best.pt` | Miglior checkpoint YOLO |
| `artifacts/history.json` | Storico loss/metriche per epoca |
| `artifacts/metrics.json` | Metriche U-Net |
| `artifacts/yolo_metrics.json` | Metriche YOLO |
| `artifacts/plots/` | Curve e confusion matrix |
| `artifacts/predictions/` | Predizioni visuali |


## Docker

La repository include un `Dockerfile` per creare un'immagine Docker dell'ambiente Python.

L'immagine non contiene dataset, modelli addestrati o artifact generati. Al primo utilizzo l'utente deve quindi scaricare il dataset, prepararlo e addestrare il modello tramite i comandi CLI.

Build dell'immagine:

```bash
docker build -t potholes-project .
```

Per mantenere persistenti dataset, modelli e risultati tra un container e l'altro, monta le cartelle `data`, `saved_model` e `artifacts` come volumi.

Download del dataset:

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli download --api-key SUA_API_KEY
```

Preparazione del dataset:

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli prepare
```

Training U-Net:

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli train
```

Valutazione e predizioni:

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli evaluate
```

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli predict
```

Pipeline YOLO:

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli yolo_prepare
```

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli yolo_train
```

```bash
docker run --rm \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli yolo_evaluate
```

Avvio della dashboard Flask:

```bash
docker run --rm -p 5000:5000 \
  -v ${PWD}/data:/akron-potholes/data \
  -v ${PWD}/saved_model:/akron-potholes/saved_model \
  -v ${PWD}/artifacts:/akron-potholes/artifacts \
  potholes-project python -m it.akron.cli api --host 0.0.0.0 --port 5000
```

Apri poi il browser su:

```text
http://127.0.0.1:5000
```

