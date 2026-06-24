# Usa un'immagine base di Python
FROM python:3.11-slim

# Imposta la directory di lavoro nel container
WORKDIR /akron-potholes

# Copia i file locali nel container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# . . Copia tutto il contenuto della directory locale
# (build context) dentro la directory corrente del container (WORKDIR)
COPY . .

# Espone la porta dell'app Flask
EXPOSE 5000

# Comando di esecuzione dell'app
CMD ["python", "app.py"]