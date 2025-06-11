import tomllib
from pathlib import Path

from langchain.chat_models.base import BaseChatModel, init_chat_model
from langchain.embeddings.base import Embeddings, init_embeddings


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def load_llm_from_config(config: dict) -> BaseChatModel:
    if "llm" not in config:
        raise ValueError("Config must contain 'llm' key.")

    llm_config = config["llm"]
    model = llm_config["model"]
    provider = llm_config.get("provider", None)
    model_kwargs = dict(llm_config.get("kwargs", {}))

    return init_chat_model(
        model=model,
        model_provider=provider,
        **model_kwargs,
    )


def load_embedder_from_config(config: dict) -> Embeddings:
    if "embedder" not in config:
        raise ValueError("Config must contain 'embedder' key.")

    embedder_config = config["embedder"]
    device = embedder_config.get("device", None)
    if device == "auto":
        import torch

        cuda_available = torch.cuda.is_available()
        device = "cuda:0" if cuda_available else "cpu"
        print(f"CUDA available: {cuda_available}, embedding device set to: {device}")

    embedder_kwargs = dict(embedder_config.get("kwargs", {}))
    model_kwargs = dict(embedder_kwargs.get("model_kwargs", {}))
    if device is not None:
        model_kwargs["device"] = device
    embedder_kwargs["model_kwargs"] = model_kwargs

    return init_embeddings(
        model=embedder_config["model"],
        provider=embedder_config.get("provider", None),
        **embedder_kwargs,
    )
