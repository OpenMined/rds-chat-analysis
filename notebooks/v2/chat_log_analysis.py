from pathlib import Path
import os
from rds_chat_analysis.job_functions import execute_chat_log_analysis
import json

DATA_DIR = Path(os.environ["DATA_DIR"])  # Contains the dataset config.toml
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])  # Dir we're writing the results to
CODE_DIR = Path(
    os.environ["CODE_DIR"]
)  # Dir containing the data scientist's input parameters (the llm query, etc.)

# Load user parameters
print(f"Loading user parameters from: {CODE_DIR / 'user_params.json'}")
job_config = CODE_DIR / "user_params.json"
with open(job_config, "r") as f:
    job_config = json.load(f)

# Execute the chat log analysis pipeline
print(f"Executing chat log analysis with config: {json.dumps(job_config, indent=2)}")
result = execute_chat_log_analysis(
    dataset_dir=DATA_DIR,
    **job_config,
)

# Save the results to the output directory
print(f"Saving results to: {OUTPUT_DIR / 'result.json'}")
with open(OUTPUT_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)
