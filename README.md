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

To set up and start the RDS server:

```bash
cd notebooks/v2

# Start the RDS server
export RDS_DO_CONFIG=./.rds/wildchat/do_config.json
export RDS_DS_CONFIG=./.rds/wildchat/ds_config.json

# Create a syftbox config for the data owner and data scientist
# NOTE only needed if you do not already have a data owner and data scientist running on a real syftbox server.
python -m syft_rds.cli init-test-datasite --email data_owner@test.openmined.org --data-dir ./.rds/wildchat/ --config-path ${RDS_DO_CONFIG}
python -m syft_rds.cli init-test-datasite --email data_scientist@test.openmined.org --data-dir ./.rds/wildchat/ --config-path ${RDS_DS_CONFIG}

# Start the RDS server
python -m syft_rds.cli server --syftbox-config ${RDS_DO_CONFIG}
```

Next, to run the notebooks:

```bash
jupyter notebook
```

## Development

```bash
pre-commit install
```
