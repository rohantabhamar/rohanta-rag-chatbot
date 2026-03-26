"""
scripts/ingest.py
──────────────────
CLI tool to build (or rebuild) the FAISS vector index from source documents.

Usage:
    python scripts/ingest.py --input my_data.txt
    python scripts/ingest.py --input docs/           # directory of .txt files
    python scripts/ingest.py --input data.txt --chunk-size 600 --chunk-overlap 80
"""

import argparse
import os
import sys

# Allow running from project root: python scripts/ingest.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_settings
from src.services import get_vector_store_service
from src.utils import get_logger, IngestionError

logger = get_logger(__name__)


def load_documents(input_path: str) -> list[Document]:
    """Load documents from a file or directory."""
    if os.path.isfile(input_path):
        logger.info("Loading file: %s", input_path)
        loader = TextLoader(input_path, encoding="utf-8")
        return loader.load()

    if os.path.isdir(input_path):
        logger.info("Loading directory: %s", input_path)
        loader = DirectoryLoader(input_path, glob="**/*.txt", loader_cls=TextLoader)
        return loader.load()

    raise IngestionError(f"Input path does not exist: '{input_path}'")


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Splitting complete: %d documents → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def run_ingestion(input_path: str, chunk_size: int, chunk_overlap: int) -> None:
    """Full ingestion pipeline: load → split → embed → index → save."""
    # 1. Load raw documents
    documents = load_documents(input_path)
    if not documents:
        raise IngestionError("No documents found. Check input path.")

    # 2. Chunk documents
    chunks = split_documents(documents, chunk_size, chunk_overlap)

    # 3. Build and persist vector store
    vs_service = get_vector_store_service()
    vs_service.build_from_documents(chunks)

    logger.info("✅ Ingestion complete. FAISS index saved to: %s", get_settings().index_dir)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from source documents."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a .txt file or directory of .txt files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Token chunk size (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap tokens (default: {settings.chunk_overlap})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_ingestion(args.input, args.chunk_size, args.chunk_overlap)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)
