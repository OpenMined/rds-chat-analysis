import json
from textwrap import dedent


def get_vector_store_query(
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
    query = dedent(query).strip()
    return query, params


def get_full_log_query(log_id: str, table_name: str = "log_embeddings") -> str:
    query = f"""
        SELECT * FROM "{table_name}"
        WHERE metadata->>'log_id' = %s
        ORDER BY (metadata->>'message_idx')::int ASC;
    """
    return dedent(query).strip(), [log_id]
