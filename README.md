# RDS Chat Analysis

## Requirements

<!-- - [just](https://github.com/casey/just?tab=readme-ov-file#installation) -->

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [ollama](https://ollama.com/download)

## Setup

```bash
uv venv -p 3.12
uv sync
source .venv/bin/activate

ollama pull gemma3:1b-it-qat
```

## Running the notebooks

the /notebooks folder have two flows:

- `notebooks/v1` is a rough implementation of embedding, clustering, and visualization of WildChat data.
- `notebooks/v2` Contains the full pipeline to setup and use syft-RDS for WildChat data.

To run the v2 flow:

```bash
cd notebooks/v2
jupyter notebook
```

## Development

```bash
pre-commit install
```
