"""
SQL query generators for vector database operations.

This module provides functions to generate parameterized SQL queries for vector
similarity search and log retrieval operations on PostgreSQL with pgvector extension.
"""

import json
from textwrap import dedent


def get_vector_store_query(
    query_embedding: list[float],
    table_name: str,
    k: int = 5,
    distance_threshold: float | None = None,
    filters: dict | None = None,
) -> tuple[str, tuple]:
    """
    Generate a parameterized SQL query for vector similarity search.

    Creates a PostgreSQL query that performs k-nearest neighbor search using cosine
    distance on vector embeddings, with optional filtering and distance thresholds.

    Args:
        query_embedding: The query vector as a list of floats to search for similar vectors.
        table_name: Name of the table containing the embeddings.
        k: Number of nearest neighbors to return. Defaults to 5.
        distance_threshold: Maximum cosine distance threshold. If provided, only
            results within this distance will be returned. Defaults to None.
        filters: Dictionary of metadata filters for exact matching. Each key-value
            pair will be applied as a JSONB containment filter. Defaults to None.

    Returns:
        A tuple containing:
            - The parameterized SQL query string
            - Tuple of parameters to be passed to the query

    Note:
        Currently only supports equality filters for metadata. The query uses cosine
        distance (<=> operator) which requires a corresponding index for performance.

    Example:
        >>> embedding = [0.1, 0.2, 0.3]
        >>> query, params = get_vector_store_query(
        ...     embedding, "embeddings", k=10,
        ...     filters={"language": "en"}
        ... )
    """
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
    query = dedent(query).strip()
    return query, tuple(params)


def get_full_log_query(
    log_id: str, table_name: str = "log_embeddings"
) -> tuple[str, tuple]:
    """
    Generate a parameterized SQL query to retrieve all messages from a specific log.

    Creates a PostgreSQL query that fetches all records belonging to a specific log ID,
    ordered by message index to maintain the chronological order of the conversation.

    Args:
        log_id: The unique identifier of the log to retrieve.
        table_name: Name of the table containing the log embeddings.
            Defaults to "log_embeddings".

    Returns:
        A tuple containing:
            - The parameterized SQL query string
            - Tuple of parameters to be passed to the query (log_id,)

    Note:
        The query assumes the metadata JSONB column contains 'log_id' and 'message_idx'
        fields, where message_idx can be cast to integer for proper ordering.

    Example:
        >>> query, params = get_full_log_query("abc-123", "chat_embeddings")
        >>> # Returns query to fetch all messages from log "abc-123" in order
    """
    query = f"""
        SELECT * FROM "{table_name}"
        WHERE metadata->>'log_id' = %s
        ORDER BY (metadata->>'message_idx')::int ASC;
    """
    return dedent(query).strip(), (log_id,)
