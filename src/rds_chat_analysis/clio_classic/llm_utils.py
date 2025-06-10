from concurrent.futures import ThreadPoolExecutor

from langchain.chat_models.base import BaseChatModel


def batch_process_llm_requests(
    messages_batch: list,
    retrying_llm: BaseChatModel,
    num_concurrent_requests: int,
) -> list:
    """
    Process a batch of messages with the LLM in parallel and return the list of responses.
    """

    def call_llm(messages):
        try:
            response = retrying_llm.invoke(messages)
            return response.model_dump(mode="json")
        except Exception as e:
            print(f"Error processing messages: {e}")
            return None

    with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        results = list(executor.map(call_llm, messages_batch))
    return results
