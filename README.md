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

## Run against a local syftbox server

Clone [syftbox](https://github.com/OpenMined/syftbox) and run:

```
just run-minio
just run-server -f config/server.dev.yaml

# launch DO and DS
just run-client-simple alice@rds.openmined.org
just run-client-simple bob@rds.openmined.org

export RDS_DO_CONFIG=~/.syftbox/clients/alice@rds.openmined.org/config.json
export RDS_DS_CONFIG=~/.syftbox/clients/bob@rds.openmined.org/config.json
```

## Run against a mock setup

```bash
cd notebooks/v2

# Start the RDS server
export RDS_DO_CONFIG=./.rds/wildchat/data_owner_config.json
export RDS_DS_CONFIG=./.rds/wildchat/data_scientist_config.json

# Create a mock syftbox config for the data owner and data scientist
python -m syft_rds.cli init-test-datasite --email data_owner@test.openmined.org --data-dir ./.rds/wildchat/ --config-path ${RDS_DO_CONFIG}
python -m syft_rds.cli init-test-datasite --email data_scientist@test.openmined.org --data-dir ./.rds/wildchat/ --config-path ${RDS_DS_CONFIG}
```

## Running the notebooks

the /notebooks folder have two flows:

- `notebooks/v1` is a rough implementation of embedding, clustering, and visualization of WildChat data.
- `notebooks/v2` Contains the full pipeline to setup and use syft-RDS for WildChat data.

To start the RDS server:

```bash
cd notebooks/v2

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
