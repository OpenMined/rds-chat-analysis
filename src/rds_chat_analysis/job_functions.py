from pathlib import Path

from rds_chat_analysis.aggregation import get_aggregation_fn
from rds_chat_analysis.utils import (
    load_config,
    load_embedder_from_config,
    load_llm_from_config,
)
from rds_chat_analysis.vector_db import connect_to_db, get_full_log
from rds_chat_analysis.vector_store_utils import build_vector_store_query

CHAT_ANALYSIS_CODE_TEMPLATE = """
from pathlib import Path
import os
from rds_chat_analysis.job_functions import execute_chat_log_analysis
import json

DATA_DIR = Path(os.environ["DATA_DIR"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
CODE_DIR = Path(os.environ["CODE_DIR"])

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

job_config = CODE_DIR / "job_config.json"
with open(job_config, "r") as f:
    job_config = json.load(f)

result = execute_chat_log_analysis(
    dataset_dir=DATA_DIR,
    **job_config,
)

with open(OUTPUT_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)
""".strip()


def execute_chat_log_analysis(
    dataset_dir: Path,
    vector_store_query: str,
    aggregation_fn: str,
    aggregation_query: str | None = None,
    max_vector_store_results: int = 5,
    distance_threshold: float = 0.5,
    filters: dict | None = None,
):
    config = load_config(dataset_dir / "config.toml")
    db_conn = connect_to_db(config)
    embedder = load_embedder_from_config(config)
    llm = load_llm_from_config(config)
    aggregation_fn_ = get_aggregation_fn(aggregation_fn)

    vector_store_query, query_params = build_vector_store_query(
        query_embedding=embedder.embed_query(vector_store_query),
        table_name="log_embeddings",
        k=max_vector_store_results,
        distance_threshold=distance_threshold,
        filters=filters,
    )

    with db_conn.cursor() as cursor:
        print(f"Executing vector store query: {vector_store_query}")
        cursor.execute(vector_store_query, query_params)
        results = cursor.fetchall()

        print(f"Found {len(results)} results.")
        log_ids = set(result["metadata"]["log_id"] for result in results)
        full_logs = []
        for log_id in log_ids:
            full_log = get_full_log(db_conn, log_id)
            full_logs.append(full_log)
        print(f"Fetched {len(full_logs)} full logs.")

    print("Aggregating results with aggregation function:", aggregation_fn)
    aggregated = aggregation_fn_(
        full_logs, llm=llm, aggregation_query=aggregation_query
    )
    return aggregated
