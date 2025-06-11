from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

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


def pairwise_qa_aggregation(
    full_logs: List[List[Dict[str, Any]]], llm: BaseChatModel, aggregation_query: str
) -> List[str]:
    results = []

    for log in full_logs:
        conversation_text = format_conversation(log)
        prompt = PAIRWISE_QA_PROMPT.format(
            conversation=conversation_text, question=aggregation_query
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        results.append(response.content)

    return results


def count_aggregation(
    full_logs: List[List[Dict[str, Any]]],
    **kwargs,
) -> List[int]:
    return len(full_logs)


def identity_aggregation(
    full_logs: List[List[Dict[str, Any]]], **kwargs
) -> List[List[Dict[str, Any]]]:
    return full_logs


def format_conversation(log: List[Dict[str, Any]]) -> str:
    formatted_messages = []

    for message in log:
        role = message["metadata"]["role"]
        text = message["text"]
        formatted_messages.append(f"{role.upper()}: {text}")

    return "\n\n".join(formatted_messages)


AGGREGATION_FUNCTIONS = {
    "identity": identity_aggregation,
    "count": count_aggregation,
    "pairwise_qa": pairwise_qa_aggregation,
}


def get_aggregation_fn(name: str):
    if name not in AGGREGATION_FUNCTIONS:
        raise ValueError(
            f"Unknown aggregation function: {name}, available: {list(AGGREGATION_FUNCTIONS.keys())}"
        )
    return AGGREGATION_FUNCTIONS[name]
