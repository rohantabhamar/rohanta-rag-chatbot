FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface
ENV PIP_NO_CACHE_DIR=1

# Only curl needed — build-essential removed (not required for prebuilt wheels)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY my_data.txt .
COPY start.sh .

# Pre-download embedding model at build time
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

RUN chmod +x start.sh && chmod -R 777 /app

EXPOSE 7860

CMD ["/bin/bash", "start.sh"]