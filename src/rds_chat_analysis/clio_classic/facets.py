import json
from pathlib import Path

import pandas as pd
from langchain.chat_models.base import BaseChatModel
from lingua import LanguageDetectorBuilder
from tqdm import tqdm

from rds_chat_analysis.facet_prompts import (
    FACET_EXTRACTORS,
    format_facet_extraction_prompt,
)
from rds_chat_analysis.llm_utils import batch_process_llm_requests


def _extract_request_facet_answer(text: str) -> str | None:
    text = text.strip()
    if "<answer>" in text:
        text = text.split("<answer>", 1)[-1]
    if "</answer>" in text:
        text = text.split("</answer>", 1)[0]
    return text.strip() or None


def extract_request_facet(
    conversation_df: pd.DataFrame,
    output_path: str | Path,
    llm: BaseChatModel,
    rps: int = 50,
    num_retries: int = 3,
) -> dict[str, str]:
    facet_name = "Request"
    facet_kwargs = FACET_EXTRACTORS[facet_name]

    _extract_llm_facets_parallel(
        conversation_df=conversation_df,
        output_path=output_path,
        facet_extractor_kwargs=facet_kwargs,
        llm=llm,
    )

    request_facet_results = pd.read_json(
        output_path,
        lines=True,
    )

    request_facet_results["request"] = request_facet_results["response"].apply(
        lambda x: _extract_request_facet_answer(x["content"])
    )

    return {row.id: row.request for row in request_facet_results.itertuples()}


def extract_language_facet(
    conversation_df: pd.DataFrame,
) -> dict[str, str]:
    """Extract the main languages of the conversation using lingua."""
    lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
    languages = {}

    for row in conversation_df.itertuples():
        conversation_concatenated = "\n".join(
            [msg["content"] for msg in row.conversation]
        )
        language = lingua_detector.detect_language_of(conversation_concatenated)
        languages[row.id] = language.name if language else "unknown"

    return languages


def extract_turns_facet(
    conversation_df: pd.DataFrame,
) -> dict[str, int]:
    """
    Extract the number of turns in each conversation.
    One turn is defined as a pair of messages (user and assistant).
    """
    return {row.id: len(row.conversation) // 2 for row in conversation_df.itertuples()}


def extract_facets(
    conversation_df: pd.DataFrame,
    llm: BaseChatModel,
    llm_cache_dir: str | Path,
    llm_rps: int = 50,
    llm_num_retries: int = 3,
) -> pd.DataFrame:
    print("Extracting Request facet...")
    request_facet = extract_request_facet(
        conversation_df=conversation_df,
        output_path=Path(llm_cache_dir) / "request_facet.jsonl",
        llm=llm,
        rps=llm_rps,
        num_retries=llm_num_retries,
    )
    print("Extracting Language facet...")
    language_facet = extract_language_facet(conversation_df)
    print("Extracting Turns facet...")
    turns_facet = extract_turns_facet(conversation_df)
    results = [
        {
            "id": id_,
            "request": request_facet.get(id_),
            "language": language_facet.get(id_),
            "turns": turns_facet.get(id_),
        }
        for id_ in conversation_df["id"]
    ]
    return pd.DataFrame(results)


def _extract_llm_facets_parallel(
    conversation_df: pd.DataFrame,
    output_path: str | Path,
    facet_extractor_kwargs: dict,
    llm: BaseChatModel,
    num_concurrent_requests: int = 10,
    num_retries: int = 3,
    batch_size: int = 100,
) -> None:
    retrying_llm = llm.with_retry(stop_after_attempt=num_retries)
    output_path = Path(output_path)
    seen = set()

    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                seen.add(json.loads(line)["id"])

    batch = []
    batch_ids = []
    with output_path.open("a") as out:
        for row in tqdm(conversation_df.itertuples(), total=len(conversation_df)):
            if row.id in seen:
                continue

            messages = format_facet_extraction_prompt(
                conversation=list(row.conversation),
                **facet_extractor_kwargs,
            )
            batch.append(messages)
            batch_ids.append(row.id)

            if len(batch) < batch_size:
                continue

            responses = batch_process_llm_requests(
                batch, retrying_llm, num_concurrent_requests
            )
            for id_, response in zip(batch_ids, responses):
                if response is not None:
                    json.dump({"id": id_, "response": response}, out)
                    out.write("\n")
            batch.clear()
            batch_ids.clear()

        if batch:
            responses = batch_process_llm_requests(
                batch, retrying_llm, num_concurrent_requests
            )
            for id_, response in zip(batch_ids, responses):
                if response is not None:
                    json.dump({"id": id_, "response": response}, out)
                    out.write("\n")
