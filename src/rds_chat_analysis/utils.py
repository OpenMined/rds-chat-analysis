"""
rds-chat-analysis uses a TOML config file for all configuration related to the chat analysis pipeline.
This module provides functions to load the configuration and initialize the LLM and embedder based on that configuration.

See notebooks/v2/config_private.example.toml for an example configuration file.
"""

import tomllib
from pathlib import Path

from langchain.chat_models.base import BaseChatModel, init_chat_model
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.base import Embeddings, init_embeddings
from langchain.storage import LocalFileStore


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def load_llm_from_config(config: dict) -> BaseChatModel:
    if "llm" not in config:
        raise ValueError("Config must contain 'llm' key.")

    llm_config = config["llm"]
    model = llm_config["model"]
    provider = llm_config.get("provider", None)
    llm_kwargs = dict(llm_config.get("kwargs", {}))

    return init_chat_model(
        model=model,
        model_provider=provider,
        **llm_kwargs,
    )


def load_embedder_from_config(
    config: dict, cache_dir: str | Path | None = None
) -> Embeddings:
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
    # For Hugginface models, add device to model_kwargs
    model_kwargs = dict(embedder_kwargs.get("model_kwargs", {}))
    if device is not None:
        model_kwargs["device"] = device
    embedder_kwargs["model_kwargs"] = model_kwargs

    embedder = init_embeddings(
        model=embedder_config["model"],
        provider=embedder_config.get("provider", None),
        **embedder_kwargs,
    )

    if cache_dir:
        file_store = LocalFileStore(root_path=Path(cache_dir))
        return CacheBackedEmbeddings.from_bytes_store(
            embedder,
            file_store,
        )
    else:
        return embedder
