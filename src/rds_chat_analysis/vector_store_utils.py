import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Iterable

import psycopg
import pyarrow
import pyarrow.parquet as pq
from langchain_core.embeddings import Embeddings
from psycopg_pool import ConnectionPool
from tqdm import tqdm


def _write_to_parquet(rows: list[dict], filepath: Path):
    table = pyarrow.Table.from_pylist(
        rows,
        schema=pyarrow.schema(
            [
                ("id", pyarrow.string()),
                ("text", pyarrow.string()),
                ("metadata", pyarrow.string()),
                ("embedding", pyarrow.list_(pyarrow.float32())),
            ]
        ),
    )
    pq.write_table(table, filepath, compression="zstd")


def embed_to_parquet(
    logs: Iterable[dict],
    embedder: Embeddings,
    preprocess_fn: Callable[[dict], list[dict]],
    output_dir: Path | str = "./embeddings/",
    rows_per_parquet_file: int = 1_000_000,
    embedder_batch_size: int = 32,
    num_logs: int = 0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    buffer = []

    with tqdm(total=num_logs, desc="Embedding logs") as pbar:
        for logs_batch in itertools.batched(logs, embedder_batch_size * 10):
            preprocessed_logs = [preprocess_fn(log) for log in logs_batch]
            preprocessed_messages = [msg for log in preprocessed_logs for msg in log]

            texts = [msg["text"] for msg in preprocessed_messages]
            embeddings = []
            for batch_to_embed in itertools.batched(texts, embedder_batch_size):
                embedded_batch = embedder.embed_query(batch_to_embed)
                embeddings.extend(embedded_batch)

            for msg, embedding in zip(preprocessed_messages, embeddings):
                msg["embedding"] = embedding
                msg["metadata"] = json.dumps(msg["metadata"])

            buffer.extend(preprocessed_messages)
            if len(buffer) >= rows_per_parquet_file:
                _write_to_parquet(buffer, output_dir / f"embeddings_{file_idx}.parquet")
                buffer.clear()
                file_idx += 1

            pbar.update(len(logs_batch))

    # Write any remaining rows to a final Parquet file
    if buffer:
        _write_to_parquet(buffer, output_dir / f"embeddings_{file_idx}.parquet")


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
    if not len(data_to_insert):
        return

    insert_query = f"""
        INSERT INTO "{table_name}" (id, text, metadata, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
    """

    with conn.cursor() as cur:
        cur.executemany(insert_query, data_to_insert)
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


def build_vector_store_query(
    query_embedding: list[float],
    table_name: str,
    k: int = 5,
    distance_threshold: float | None = None,
    filters: dict | None = None,
) -> tuple[str, list]:
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    params = [query_embedding_str]

    # JSONB filters, e.g. metadata @> '{"language": "English"}'
    # TODO only supports equality filters
    where_clauses = []
    if filters:
        for key, value in filters.items():
            where_clauses.append("metadata @> %s")
            params.append(json.dumps({key: value}))

    # Distance threshold
    # Only supports cosine distance (<=>), because that's what the index is built with
    if distance_threshold is not None:
        where_clauses.append("(embedding <=> %s) <= %s")
        params.append(query_embedding_str)
        params.append(distance_threshold)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    params.append(k)

    query = f"""
        SELECT id, text, metadata, (embedding <=> %s) AS distance
        FROM "{table_name}"
        {where_sql}
        ORDER BY distance ASC
        LIMIT %s;
    """

    return query, params


def query_vector_store(
    conn: psycopg.Connection,
    embedder,
    table_name: str,
    query_text: str,
    k: int = 5,
    distance_threshold: float | None = None,
    filters: dict | None = None,
) -> list[dict]:
    query_embedding = embedder.embed_query(query_text)
    sql, params = build_vector_store_query(
        query_embedding,
        table_name,
        k=k,
        distance_threshold=distance_threshold,
        filters=filters,
    )

    results = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        results = cur.fetchall()
    return results
