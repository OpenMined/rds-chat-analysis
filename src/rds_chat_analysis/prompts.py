from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

EXAMPLES_SECTION = (
    "When answering, do not include any personally identifiable information (PII), like names, locations, phone numbers, email addresses, and so on. "
    "Do not include any proper nouns. Output your answer to the question in English inside <answer> tags; "
    'be clear and concise and get to the point in at most two sentences (don\'t say "Based on the conversation..." and avoid mentioning the chatbot).\n\n'
    "For example:\n"
    "<examples>\n"
    "The user asked for help with a trignometry problem.\n"
    "The user asked for advice on how to fix a broken dishwasher. It took several attempts to get the right answer.\n"
    "The user asked how to make Anthrax and the AI system refused the requests.\n"
    "</examples>"
)

FACET_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            "The following is a conversation between an AI assistant and a user:\n{conversation}"
        ),
        AIMessagePromptTemplate.from_template("I understand."),
        HumanMessagePromptTemplate.from_template(
            (
                "Your job is to answer the question <question> {question} </question> about the preceding conversation. "
                "Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially harmful or sensitive topics; "
                "specificity around potentially harmful conversations is necessary for effective monitoring.\n\n"
                "{examples_section}"
                "What is your answer to the question <question> {question} </question> about the preceding conversation, in <answer> tags? "
                "Again, provide only the answer with no other commentary or proper nouns."
            )
        ),
        AIMessagePromptTemplate.from_template(
            "Sure, the privacy-preserving answer to the question about the preceding conversation is: <answer> {prefill}"
        ),
    ]
)

FACET_EXTRACTORS = {
    "Request": {
        "question": "What is the user's overall request for the assistant?",
        "prefill": "The user's overall request for the assistant is to ",
        "include_examples_section": True,
    },
    "Language": {
        "question": (
            "What are the main languages of the conversation? Do not mention programming languages and do not "
            "provide only language codes; only the full names for human languages, like 'English' or 'Arabic; Indonesian'. "
            "Do not include language families (just provide the general language, e.g., 'Arabic' and not 'Tripolitanian Arabic'; "
            "something that a standard langcodes package would be able to identify). Only consider languages in or after the human's "
            "first request. Output each language as a single full word with no other commentary."
        ),
        "prefill": "",
        "include_examples_section": False,
    },
    "Task": {
        "question": "What task is the model being asked to perform in this conversation?",
        "prefill": "The task is to ",
        "include_examples_section": True,
    },
    "Concerning": {
        "question": (
            "On a scale of 1 to 5, how concerning or potentially harmful is this conversation from a safety perspective? "
            "Use the following scale:\n\n"
            "1: Not concerning at all. Completely benign conversation.\n"
            "Example: A friendly chat about favorite books or movies.\n"
            "2: Slightly concerning, but likely harmless.\n"
            "Example: A heated but civil debate about politics.\n"
            "3: Moderately concerning. May warrant a closer look.\n"
            "Example: Discussion of controversial topics that could escalate.\n"
            "4: Very concerning. Likely needs review.\n"
            "Example: Conversations involving potential self-harm or harmful ideation.\n"
            "5: Extremely concerning. Immediate review needed.\n"
            "Example: Explicit threats of violence or illegal activities.\n\n"
            "Answer with only a single number from 1 to 5."
        ),
        "prefill": "",
        "include_examples_section": False,
    },
}


class MessageDict(TypedDict):
    role: Literal["assistant", "user", "system"]
    content: str


def _format_conversation(conversation: list[MessageDict]) -> str:
    return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)


def format_facet_extraction_prompt(
    conversation: list[MessageDict] | str,
    question: str,
    prefill: str = "",
    include_examples_section: bool = True,
) -> list[BaseMessage]:
    """
    Formats the FACET_EXTRACTION_PROMPT with the provided conversation, question, and prefill.

    Args:
        conversation (str): The conversation text.
        question (str): The question to be answered.
        prefill (str): The prefilled answer to the question.
        include_examples_section (bool): Whether to include the examples section in the prompt.

    Returns:
        ChatPromptTemplate: The formatted prompt ready for use.
    """
    if isinstance(conversation, list):
        conversation = _format_conversation(conversation)
    if not isinstance(conversation, str):
        raise ValueError(
            "Conversation must be a string or a list of dictionaries with role: str, content: str."
        )
    return FACET_EXTRACTION_PROMPT.format_messages(
        conversation=conversation,
        question=question,
        prefill=prefill,
        examples_section=EXAMPLES_SECTION if include_examples_section else "",
    )
