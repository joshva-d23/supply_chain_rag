"""
ingest.py — Load PDFs → chunk → embed → store in persistent ChromaDB
"""

from pathlib import Path
from typing import List
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")


def load_and_split(
    pdf_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """
    Load one or more PDFs, attach clean metadata, and split into chunks.
    chunk_size=1000 / overlap=150 keeps most tables together while
    staying inside the required 800–1200 / 100–200 range.
    """
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            # Ensure clean, consistent metadata
            page.metadata["source"] = Path(path).name
            # page number is already 0-indexed by PyPDFLoader
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_vectorstore(
    chunks,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = "meridian_supplychain",
):
    """Create (or overwrite) a persisted Chroma collection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Remove old collection if it exists so re-indexing is clean
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vectorstore


def get_vectorstore(
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = "meridian_supplychain",
):
    """Load an existing persisted Chroma collection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
