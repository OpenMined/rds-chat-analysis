"""
This module implements a simple vector database on top of PostgreSQL with pgvector extension.
"""

import psycopg
from psycopg.rows import dict_row

DEFAULT_VECTOR_INDEX_SETTINGS = {
    "index_type": "hnsw",
    "index_params": {"m": 16, "ef_construction": 64},
    "distance_opclass": "vector_cosine_ops",
}


def _init_db(conn: psycopg.Connection, index_type: str):
    with conn.cursor() as cur:
        if index_type == "hnsw":
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        elif index_type == "diskann":
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_diskann CASCADE;")
        else:
            raise ValueError(
                f"Unsupported index type: {index_type}. Supported types are 'hnsw' and 'diskann'."
            )
    conn.commit()
    print(f"Initialized database for index type: {index_type}")


def _init_schema(
    conn: psycopg.Connection,
    table_name: str,
    embedding_size: int,
    overwrite_existing: bool = False,
):
    """Creates the main table for storing vectors and metadata."""
    with conn.cursor() as cur:
        if overwrite_existing:
            cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id TEXT PRIMARY KEY,
                text TEXT,
                metadata JSONB,
                embedding vector({embedding_size})
            );
        """)
    conn.commit()
    print(f"Created table {table_name} with embedding size {embedding_size}")


def _setup_metadata_index(
    conn: psycopg.Connection,
    table_name: str,
    overwrite_existing: bool = False,
):
    """Create GIN index on metadata JSONB"""
    with conn.cursor() as cur:
        if overwrite_existing:
            cur.execute(f'DROP INDEX IF EXISTS "idx_gin_{table_name}_metadata";')
        idx_gin_metadata_name = f"idx_gin_{table_name}_metadata"
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS "{idx_gin_metadata_name}"
            ON "{table_name}"
            USING GIN (metadata jsonb_path_ops);
        """)
    conn.commit()
    print(f"Created GIN index on metadata for table {table_name}")


def _setup_vector_index(
    conn: psycopg.Connection,
    table_name: str,
    index_type: str,
    index_params: dict,
    distance_opclass: str = "vector_cosine_ops",
    overwrite_existing: bool = False,
    maintenance_mem: str = "2GB",
    parallel_workers: int = 4,
):
    """
    Creates a vector index (HNSW or DiskANN) on the embedding column
    using the provided type and settings.
    """
    with conn.cursor() as cur:
        index_name = f"idx_{index_type.lower()}_{table_name}_{distance_opclass}"
        if overwrite_existing:
            cur.execute(f'DROP INDEX IF EXISTS "{index_name}";')

        cur.execute(f"SET local maintenance_work_mem = '{maintenance_mem}';")
        cur.execute(f"SET local max_parallel_maintenance_workers = {parallel_workers};")
        cur.execute(f"SET local max_parallel_workers = {parallel_workers};")

        with_clauses = []
        if index_params:
            for key, value in index_params.items():
                if isinstance(value, str):
                    with_clauses.append(f"{key} = '{value}'")
                else:
                    with_clauses.append(f"{key} = {value}")

        with_statement = ""
        if with_clauses:
            with_statement = "WITH (" + ", ".join(with_clauses) + ")"

        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS "{index_name}"
            ON "{table_name}"
            USING {index_type.lower()} (embedding {distance_opclass})
            {with_statement};
        """)
    conn.commit()
    print(f"Created {index_type} index on {table_name} with params {index_params}")


def setup_db(
    conn: psycopg.Connection,
    table_name: str,
    embedding_size: int,
    vector_index_settings: dict | None = None,
    overwrite_existing: bool = False,
):
    """
    Sets up the vector store with the specified table name, embedding size, and index type.
    """
    if vector_index_settings is None:
        print("No vector index settings provided. Using default settings.")
        vector_index_settings = DEFAULT_VECTOR_INDEX_SETTINGS
    index_type = vector_index_settings["index_type"]

    _init_db(conn, index_type=index_type)
    _init_schema(
        conn, table_name, embedding_size, overwrite_existing=overwrite_existing
    )


def setup_indices(
    conn: psycopg.Connection,
    table_name: str,
    vector_index_settings: dict | None = None,
    overwrite_existing: bool = False,
    pg_maintenance_mem: str = "2GB",
    pg_parallel_workers: int = 4,
):
    """
    Sets up the vector store with the specified table name, embedding size, and index type.
    """
    if vector_index_settings is None:
        print("No vector index settings provided. Using default settings.")
        vector_index_settings = DEFAULT_VECTOR_INDEX_SETTINGS
    index_type = vector_index_settings["index_type"]
    index_params = vector_index_settings["index_params"]
    distance_opclass = vector_index_settings["distance_opclass"]

    _setup_metadata_index(
        conn,
        table_name,
        overwrite_existing=overwrite_existing,
    )
    _setup_vector_index(
        conn,
        table_name,
        index_type,
        index_params,
        distance_opclass=distance_opclass,
        overwrite_existing=overwrite_existing,
        maintenance_mem=pg_maintenance_mem,
        parallel_workers=pg_parallel_workers,
    )


def connect_to_db(config: dict) -> psycopg.Connection:
    db_config = config["db"]

    db_conn_settings = {
        "host": db_config["postgres_host"],
        "port": db_config["postgres_port"],
        "dbname": db_config["postgres_db"],
        "user": db_config["postgres_user"],
        "password": db_config["postgres_password"],
        "sslmode": db_config.get("postgres_sslmode", "require"),
    }

    keepalive_kwargs = {
        "keepalives": 1,
        "keepalives_idle": 60,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }

    return psycopg.connect(
        **db_conn_settings,
        row_factory=dict_row,
        **keepalive_kwargs,
    )
