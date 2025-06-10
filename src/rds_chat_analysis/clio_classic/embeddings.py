from pathlib import Path

import pandas as pd
import torch
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# We use gte-multilingual, best mix of fast, small, high MTEB scores
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MODEL_KWARGS = {
    "device": "cuda:0" if cuda_available else "cpu",
    "trust_remote_code": True,  # Required to run gte-multilingual in sentence-transformers
}
ENCODE_KWARGS = {"normalize_embeddings": False}


def load_embedder(cache_dir: Path | None = None) -> Embeddings:
    """
    Load the embedding model with optional caching support.

    Args:
        cache_dir (Path | None): Optional directory to store cached embeddings.

    Returns:
        HuggingFaceEmbeddings: The loaded embedding model.
    """
    embedder = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    )

    if cache_dir:
        store = LocalFileStore(cache_dir)
        embedder = CacheBackedEmbeddings.from_bytes_store(
            embedder,
            store,
            namespace=MODEL_NAME,
        )

    return embedder


def embed_column(
    df: pd.DataFrame,
    column: str,
    embedder: Embeddings,
    embedded_column_name: str | None = None,
    batch_size: int = 8,
    inplace: bool = True,
) -> pd.DataFrame:
    texts = df[column].tolist()
    embeddings = create_embeddings(texts, embedder, batch_size)
    if not inplace:
        df = df.copy()
    df[f"{column}_embedding"] = embeddings
    return df


def create_embeddings(
    texts: list[str],
    embedder: Embeddings,
    batch_size: int = 8,
) -> list[list[float]]:
    embeddings = []
    batched_texts = []

    for i, text in tqdm(enumerate(texts), total=len(texts), desc="Creating embeddings"):
        if not text:
            continue
        batched_texts.append(text)

        if len(batched_texts) == batch_size:
            embeddings_batch = embedder.embed_documents(batched_texts)
            embeddings.extend(embeddings_batch)
            batched_texts.clear()

    if batched_texts:
        embeddings_batch = embedder.embed_documents(batched_texts)
        embeddings.extend(embeddings_batch)

    return embeddings
