#!/bin/bash

rm -rf .venv
uv venv -p 3.12
uv pip install --upgrade .
uv run syft-rds server