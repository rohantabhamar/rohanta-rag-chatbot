#!/bin/bash
set -e

echo "=== Starting ingestion ==="
python scripts/ingest.py --input my_data.txt

echo "=== Starting Streamlit ==="
streamlit run src/ui/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true