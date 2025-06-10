from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

CLUSTER_DESCRIPTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            (
                "You are tasked with summarizing a group of related statements into a short, precise, and accurate description and name.\n"
                "Your goal is to create a concise summary that captures the essence of these statements and distinguishes them from other similar groups.\n\n"
                "Summarize all the statements into a clear, precise, two-sentence description in the past tense. "
                "Your summary should be specific to this group and distinguish it from the contrastive answers of the other groups.\n\n"
                "After creating the summary, generate a short name for the group of statements. This name should be at most ten words long (perhaps less) "
                "and be specific but also reflective of most of the statements (rather than just one or two).\n\n"
                "The name should distinguish this group from the contrastive examples. For instance:\n"
                '- "Write fantasy sexual roleplay with octopi and monsters"\n'
                '- "Generate blog spam for gambling websites"\n'
                '- "Assist with high school math homework"\n\n'
                'These are better than vague terms like "Write erotic content" or "Help with homework".\n\n'
                "Be as descriptive as possible and assume neither good nor bad faith. Do not hesitate to identify and describe socially harmful or sensitive topics specifically; "
                "specificity is necessary for monitoring.\n\n"
                "Present your output in the following format:\n"
                "<summary> [Insert your two-sentence summary here] </summary>\n"
                "<name> [Insert your generated short name here] </name>\n\n"
                "The names you propose must follow these requirements:\n"
                "<criteria>{criteria}</criteria>\n\n"
                "Below are the related statements:\n"
                "<answers>\n"
                "{answers}\n"
                "</answers>\n\n"
                "For context, here are statements from nearby groups that are NOT part of the group you’re summarizing:\n"
                "<contrastive_answers>\n"
                "{contrastive_answers}\n"
                "</contrastive_answers>\n\n"
                "Do not elaborate beyond what you say in the tags. Analyze both the statements and the contrastive statements carefully to ensure your summary and name "
                "accurately represent the group while distinguishing it from others."
            )
        ),
        AIMessagePromptTemplate.from_template(
            "Sure, I will provide a clear, precise, and accurate summary and name for this cluster. "
            "I will be descriptive and assume neither good nor bad faith. Here is the summary, which I will follow with the name:\n<summary>"
        ),
    ]
)

FACET_CRITERIA: dict[str, list[str]] = {
    "Request": [
        "The cluster name should be a sentence in the imperative that captures the user’s request.",
        "For example: 'Brainstorm ideas for a birthday party', 'Help me find a new job'.",
    ],
}


def format_cluster_description_prompt(
    answers: list[str],
    contrastive_answers: list[str],
    criteria: list[str],
) -> list[BaseMessage]:
    return CLUSTER_DESCRIPTION_PROMPT.format_messages(
        answers="\n".join(answers),
        contrastive_answers="\n".join(contrastive_answers),
        criteria="\n".join(criteria),
    )
