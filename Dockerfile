FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY my_data.txt .

EXPOSE 7860

# Build FAISS index first, then start the app
CMD ["sh", "-c", "python scripts/ingest.py --input my_data.txt && streamlit run src/ui/app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true"]