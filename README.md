# Akron Pothole Segmentation

Progetto Python object oriented semplice per segmentare buche stradali con una U-Net in PyTorch.

## Struttura

```text
progetto formatemp/
  data/                 # dataset salvato nel progetto
    raw/                # dataset originale scaricato una sola volta
    processed/          # train/valid/test pronti per PyTorch
  saved_model/          # modello addestrato
  artifacts/            # metriche, grafici e immagini per la dashboard
  it/
    akron/
      api/
        app.py          # dashboard Flask
      dataset/
        dataset.py      # download, preparazione dataset, Dataset PyTorch
      models/
        model.py        # U-Net e summary
        yolo.py         # YOLOv8
      training/
        trainer.py      # training, evaluation, grafici, predizioni
      cli.py            # comandi da terminale
      config.py         # configurazione
```

## Uso rapido

Installa le dipendenze:

```powershell
pip install -e .
```

Scarica il dataset da Roboflow impostando la chiave come variabile d'ambiente:

```powershell
$env:ROBOFLOW_API_KEY="..."
akron-potholes download
```

Il download serve solo la prima volta: il dataset viene copiato in `data/raw/`.

Prepara maschere, split train/valid/test e augmentation in `data/processed/`:

```powershell
akron-potholes prepare
```

Allena il modello:

```powershell
akron-potholes train
```

Valuta sul test set:

```powershell
akron-potholes evaluate --threshold 0.4
```

Salva alcune predizioni con bounding box:

```powershell
akron-potholes predict --boxes
```

Avvia l'API Flask:

```powershell
akron-potholes api
```

La dashboard sara disponibile su `http://127.0.0.1:5000`.

## Dashboard e caching

Al primo avvio la dashboard controlla se esiste `saved_model/best_unet_potholes.pt`.

- Se il modello esiste, viene solo caricato.
- Se il modello non esiste, prova a usare il dataset locale gia salvato nel progetto, prepara lo split se serve, allena il modello e salva i pesi.
- Se non esiste nessun dataset locale, bisogna scaricarlo una sola volta con `akron-potholes download` e poi prepararlo con `akron-potholes prepare`.

Gli output generati dall'API vengono salvati dentro `artifacts/`:

- `artifacts/metrics.json`
- `artifacts/history.json`
- `artifacts/model_summary.txt`
- `artifacts/plots/*.png`
- `artifacts/predictions/*.png`

## Nota sul notebook originale

Il notebook contiene una pipeline corretta come prototipo, ma non dovrebbe tenere una API key in chiaro. In questo progetto la chiave Roboflow viene letta da `ROBOFLOW_API_KEY`.
