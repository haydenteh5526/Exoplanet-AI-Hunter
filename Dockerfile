FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

COPY models/ models/
COPY data/processed/ data/processed/
COPY src/ src/
COPY web_app/ web_app/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "web_app", "app:app"]
