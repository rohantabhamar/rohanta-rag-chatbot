FROM python:3.11-slim

WORKDIR /app

# Set a writable, explicit cache directory for HuggingFace models
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       torch==2.5.1 \
       --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Make the cache dir before downloading into it
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY my_data.txt .
COPY start.sh .

# ⬇️ NEW: download the embedding model at BUILD time, not runtime
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

RUN chmod +x start.sh
RUN chmod -R 777 /app

EXPOSE 7860

CMD ["/bin/bash", "start.sh"]