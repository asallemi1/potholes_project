FROM python:3.11-slim

WORKDIR /akron-potholes

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["python", "-m", "it.akron.cli"]
CMD ["api", "--host", "0.0.0.0", "--port", "5000"]