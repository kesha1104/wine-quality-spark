FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything (train, predict, etc.)
COPY .. .

# default entrypoint: run prediction script
ENTRYPOINT ["python", "predict/predict_app.py"]
