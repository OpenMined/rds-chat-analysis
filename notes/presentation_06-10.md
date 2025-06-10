# Plan for June

1. Create a flow in syft-rds where a datascientist can query a private postgres vector store, do a pre-defined aggregation (counts, QA with an LLM, ...) and get results back.
2. Write a thin library on top of RDS to make this easy for both DS and DO
3. End of this week: local end-to-end demo with WildChat dataset
4. Deploy a first version with our cloud provider
5. First milestone: deployed end-to-end demo with real dataset end of this month

# How does this work over RDS?

## 1. Creating 'Custom APIs' like in PySyft

- Alongside RDS, both DO and DS install a `rds-log-analysis` package. this contains utilities and wrappers for submitting standardized jobs

```python
# Example:

import rds_log_analysis

# Client is a subclass of RDSClient, with a few extra methods for easy job submission
client = rds_log_analysis.connect(host="data_owner@openmined.org")
job = client.submit_job(
    vector_store_query="Messages about food and drink",
    ...
)

# get_results loads the result file, and formats it as a Pandas DataFrame
client.get_results(job)
```

## 2. Submitting a job creates 2 files:

- `job.json`: Contains the job arguments ('vector_store_query', ...)

```json
{
  "vector_store_query": "Messages about food and drinks",
  "aggregation_query": "What are the main topics discussed in this conversation?",
  "aggregation_fn": "pairwise_qa",
  "k": 10,
  "distance_threshold": 0.4,
  "filters": {
    "role": "user"
  }
}
```

- `main.py` loads the job args, calls the job function, and writes the result
  - This file is standardized, and not created by the user

```python
from pathlib import Path
import os
from rds_chat_analysis.job import execute_chat_log_analysis
import dotenv

DATA_DIR = os.environ["DATA_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

dotenv.load_dotenv(DATA_DIR / "credentials.env", override=True)

job_args = json.load("./job.json")
result = execute_chat_log_analysis(job_args, DATA_DIR)
result.to_csv(Path(OUTPUT_DIR) / "result.csv", index=False)
```

## 3. Running the job

- Default RDS flow.
- We can do this in a container or on the server.

### 4. DS obtains result

- Default RDS flow.
- We can provide a utility function to load the result file, and format it as a Pandas DataFrame.
