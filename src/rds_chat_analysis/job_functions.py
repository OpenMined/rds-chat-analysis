from pathlib import Path
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from rds_chat_analysis.recitation_filter import RecitationScorer
from rds_chat_analysis.utils import (
    load_config,
    load_embedder_from_config,
    load_llm_from_config,
)
from rds_chat_analysis.vector_db import connect_to_db, get_full_log
from rds_chat_analysis.vector_store_utils import build_vector_store_query

PAIRWISE_QA_PROMPT = (
    "The following is a conversation between an AI assistant and a user:\n"
    "{conversation}\n\n"
    "Your job is to answer the question about the preceding conversation. "
    "Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially harmful or sensitive topics; "
    "specificity around potentially harmful conversations is necessary for effective monitoring.\n\n"
    "When answering, do not include any personally identifiable information (PII), like names, locations, phone numbers, email addresses, and so on. "
    "Do not include any proper nouns or specific names of people, places, or organizations.\n\n"
    "be clear and concise and get to the point in at most two sentences.\n\n"
    "Question: {question}\n\n"
    "What is your answer to the question about the preceding conversation? "
    "Provide only the answer with no other commentary or proper nouns."
)

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


def format_conversation(log: List[Dict[str, Any]]) -> str:
    formatted_messages = []

    for message in log:
        role = message["metadata"]["role"]
        text = message["text"]
        formatted_messages.append(f"{role.upper()}: {text}")

    return "\n\n".join(formatted_messages)


def pairwise_chat_question_answering(
    full_logs: List[List[Dict[str, Any]]], llm: BaseChatModel, llm_query: str
) -> List[str]:
    results = []

    for log in full_logs:
        conversation_text = format_conversation(log)
        prompt = PAIRWISE_QA_PROMPT.format(
            conversation=conversation_text, question=llm_query
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        results.append(response.content)

    return results


def execute_chat_log_analysis(
    dataset_dir: Path,
    vector_store_query: str,
    llm_query: str,
    max_vector_store_results: int = 5,
    distance_threshold: float = 0.5,
    filters: dict | None = None,
    recitation_filter_n: int = 8,
) -> list[str]:
    """
    A simple chat log analysis pipeline. Steps:
    - Retrieve chat messages from the vector store based on 'vector_store_query'.
    - Retrieve full chat logs for the retrieved messages.
    - Execute the 'llm_query' against each full chat log using the LLM.
    - Filter results based on n-gram overlap using the 'recitation_filter_n' parameter.

    Args:
        dataset_dir (Path): Path to the dataset directory.
        vector_store_query (str): Query to be executed against the vector store,
            matches individual chat messages.
        llm_query (str): Query to be executed against the LLM,
            executed against each retrieved full chat log.
        max_vector_store_results (int, optional): max number of results from the vector store.
            Defaults to 5.
        distance_threshold (float, optional): Maximum cosine distance for the vector store search.
            Defaults to 0.5.
        filters (dict | None, optional): Vector store metadata filters to apply.
            Defaults to None.
        recitation_filter_n (int, optional): Post-filtering of results based on n-gram overlap.
            If there is an n-gram overlap at this size, the log will be skipped.
            Defaults to 6.

    Returns:
        list[str]: List of LLM answers to the question asked about each chat log.
    """
    config = load_config(dataset_dir / "config.toml")
    db_conn = connect_to_db(config)
    embedder = load_embedder_from_config(config)
    llm = load_llm_from_config(config)

    vector_store_query, query_params = build_vector_store_query(
        query_embedding=embedder.embed_query(vector_store_query),
        table_name="log_embeddings",
        k=max_vector_store_results,
        distance_threshold=distance_threshold,
        filters=filters,
    )

    with db_conn.cursor() as cursor:
        print(f"\nExecuting vector store query: {vector_store_query}")
        cursor.execute(vector_store_query, query_params)
        llm_answers = cursor.fetchall()

        print(f"Found {len(llm_answers)} results.")
        log_ids = set(result["metadata"]["log_id"] for result in llm_answers)
        full_logs = []
        for log_id in log_ids:
            full_log = get_full_log(db_conn, log_id)
            full_logs.append(full_log)
        print(f"Fetched {len(full_logs)} full logs.")

    print("\nRunning pairwise question answering on each chat log...")
    llm_answers = pairwise_chat_question_answering(
        full_logs,
        llm=llm,
        llm_query=llm_query,
    )

    print("\nCalculating recitation scores...")
    recitation_filtered_logs: list[str] = []
    scorer = RecitationScorer(n_min=recitation_filter_n, n_max=recitation_filter_n)
    for i, full_log in enumerate(full_logs):
        llm_answer = llm_answers[i]
        candidate_text = llm_answer
        reference_texts = [message["text"] for message in full_log]
        recitation_score = scorer.score(candidate_text, reference_texts)
        if recitation_score.overlaps_per_n[recitation_filter_n] > 0:
            print(
                f"Log {i} contains n-gram overlap at n={recitation_filter_n}. Skipping."
            )
        else:
            recitation_filtered_logs.append(llm_answer)

    return recitation_filtered_logs
