"""
This module provides utilities for embeddinga and storing chat logs in a PostgreSQL vector database.

The intended pipeline has two main steps:
1. Embedding chat logs into Parquet files.
2. Loading these Parquet files into a PostgreSQL vector database.

These steps are split so experimenting with different embedding models and database configurations is easier.
"""

import itertools
import json
from pathlib import Path
from typing import Iterable

import psycopg
import pyarrow
import pyarrow.parquet as pq
from langchain_core.embeddings import Embeddings
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from tqdm import tqdm


def _write_to_parquet(rows: list[dict], filepath: Path) -> Path:
    table = pyarrow.Table.from_pylist(rows)
    pq.write_table(table, filepath, compression="zstd")
    return filepath


def embed_to_parquet(
    logs: list[list[dict]],
    embedder: Embeddings,
    output_dir: Path | str = "./embeddings/",
    rows_per_parquet_file: int = 250_000,
    embedder_batch_size: int = 32,
    num_logs: int | None = None,
):
    """
    Expects logs as a list of lists of dicts, where each inner list is a preprocessed log (list of messages).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    buffer = []
    saved_files = []

    if num_logs is None and hasattr(logs, "__len__"):
        num_logs = len(logs)

    with tqdm(total=num_logs, desc="Embedding logs") as pbar:
        for logs_batch in itertools.batched(logs, embedder_batch_size * 10):
            preprocessed_messages = [msg for log in logs_batch for msg in log]

            texts = [msg["text"] for msg in preprocessed_messages]
            embeddings = []
            for batch_to_embed in itertools.batched(texts, embedder_batch_size):
                embedded_batch = embedder.embed_documents(batch_to_embed)
                embeddings.extend(embedded_batch)

            for msg, embedding in zip(preprocessed_messages, embeddings):
                msg["embedding"] = embedding
                # msg["metadata"] = json.dumps(msg["metadata"])

            buffer.extend(preprocessed_messages)
            if len(buffer) >= rows_per_parquet_file:
                fp = _write_to_parquet(
                    buffer, output_dir / f"embeddings_{file_idx}.parquet"
                )
                saved_files.append(fp)
                buffer.clear()
                file_idx += 1

            pbar.update(len(logs_batch))

    # Write any remaining rows to a final Parquet file
    if buffer:
        fp = _write_to_parquet(buffer, output_dir / f"embeddings_{file_idx}.parquet")
        saved_files.append(fp)

    print(f"Saved {len(saved_files)} Parquet files to {output_dir}")


def load_embeddings_from_file(
    filepath: Path | str,
) -> list[dict]:
    table = pq.read_table(filepath)
    return table.to_pylist()


def load_embeddings_from_file_batched(
    filepath: Path | str,
    batch_size: int = 1000,
) -> Iterable[list[dict]]:
    # Batched loading when inserting into postgres
    reader = pq.ParquetFile(filepath)
    for batch in reader.iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


def load_embeddings_from_dir_batched(
    dir: Path | str,
    batch_size: int = 1000,
) -> Iterable[list[dict]]:
    dir = Path(dir)
    for file in dir.glob("*.parquet"):
        yield from load_embeddings_from_file_batched(file, batch_size=batch_size)


def count_embeddings_from_file(
    filepath: Path | str,
) -> int:
    table = pq.read_table(filepath)
    return table.num_rows


def count_embeddings_from_dir(
    dir: Path | str,
) -> int:
    # Count number of rows in all Parquet files, for progress bar
    dir = Path(dir)
    total_count = 0
    for file in dir.glob("*.parquet"):
        total_count += count_embeddings_from_file(file)
    return total_count


def _add_batch_to_db(conn: psycopg.Connection, table_name: str, data_to_insert: list):
    if not data_to_insert:
        return

    processed_data = []
    for row in data_to_insert:
        id_val, text, metadata, embedding = row
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                raise ValueError(f"Invalid JSON string in metadata: {metadata}")
        if not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be dict, got {type(metadata)}: {metadata}")
        processed_data.append((id_val, text, Jsonb(metadata), embedding))

    insert_query = f"""
        INSERT INTO "{table_name}" (id, text, metadata, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
    """

    with conn.cursor() as cur:
        cur.executemany(insert_query, processed_data)
    conn.commit()


def add_embeddings_to_db(
    conn: psycopg.Connection,
    table_name: str,
    embeddings_dir: Path | str,
    batch_size: int = 256,
):
    """
    Adds embeddings from Parquet files to the custom vector table.
    """
    total_count = count_embeddings_from_dir(embeddings_dir)
    embedding_iterator = load_embeddings_from_dir_batched(
        dir=embeddings_dir,
        batch_size=batch_size,
    )

    with tqdm(total=total_count, desc=f"Inserting into {table_name}") as pbar:
        for batch_of_dicts in embedding_iterator:
            data_to_insert = [
                (
                    row["id"],
                    row["text"],
                    row["metadata"],
                    row["embedding"],
                )
                for row in batch_of_dicts
            ]

            _add_batch_to_db(conn, table_name, data_to_insert)
            pbar.update(len(batch_of_dicts))


def add_embeddings_to_db_pooled(
    pool: ConnectionPool,
    table_name: str,
    embeddings_dir: Path | str,
    batch_size: int = 256,
    limit: int | None = None,
):
    """
    Adds embeddings from Parquet files to the custom vector table using a connection pool.
    """
    total_count = count_embeddings_from_dir(embeddings_dir)
    embedding_iterator = load_embeddings_from_dir_batched(
        dir=embeddings_dir,
        batch_size=batch_size,
    )

    with tqdm(total=total_count, desc=f"Inserting into {table_name}") as pbar:
        for batch_of_dicts in embedding_iterator:
            data_to_insert = [
                (
                    row["id"],
                    row["text"],
                    row["metadata"],
                    row["embedding"],
                )
                for row in batch_of_dicts
            ]

            with pool.connection() as conn:
                _add_batch_to_db(conn, table_name, data_to_insert)

            pbar.update(len(data_to_insert))
