import json
import tempfile
from pathlib import Path
from typing import Any, Self
from uuid import UUID

import rich
from IPython.display import HTML, display
from pydantic import BaseModel
from rich.prompt import Confirm, Prompt
from syft_core import Client as SyftBoxClient
from syft_event import SyftEvents
from syft_rds import init_session as _rds_init_session
from syft_rds.client.rds_client import RDSClient
from syft_rds.models.html_format import create_html_repr
from syft_rds.models.models import Job, JobStatus

from rds_chat_analysis.job_functions import CHAT_ANALYSIS_CODE_TEMPLATE


class JobOutput(BaseModel):
    job: Job
    execution_output_dir: Path

    @property
    def logs_dir(self) -> Path:
        return self.execution_output_dir / "logs"

    @property
    def output_dir(self) -> Path:
        return self.execution_output_dir / "output"

    @property
    def stderr_file(self) -> Path:
        return self.logs_dir / "stderr.log"

    @property
    def stdout_file(self) -> Path:
        return self.logs_dir / "stdout.log"

    @property
    def stderr(self) -> str | None:
        if self.stderr_file.exists():
            return self.stderr_file.read_text()
        return None

    @property
    def stdout(self) -> str | None:
        if self.stdout_file.exists():
            return self.stdout_file.read_text()
        return None

    @property
    def log_files(self) -> list[Path]:
        return list(self.logs_dir.glob("*"))

    @property
    def output_files(self) -> list[Path]:
        return list(self.output_dir.glob("*"))

    @property
    def outputs(self) -> dict[str, Any]:
        output_files = list(self.output_dir.glob("*.json"))
        outputs = {}
        for file in output_files:
            if file.name.endswith(".json"):
                with open(file, "r") as f:
                    outputs[file.name] = json.load(f)
            elif file.name.endswith(".parquet"):
                import pandas as pd

                outputs[file.name] = pd.read_parquet(file)
            elif file.name.endswith(".csv"):
                import pandas as pd

                outputs[file.name] = pd.read_csv(file)
            elif file.name.endswith(".txt"):
                with open(file, "r") as f:
                    outputs[file.name] = f.read()
            else:
                rich.print(
                    f":warning: Unsupported file type {file.name}. Please check this file manually."
                )
        return outputs

    def describe(self):
        display_paths = ["output_dir"]
        if self.stdout_file.exists():
            display_paths.append("stdout_file")
        if self.stderr_file.exists():
            display_paths.append("stderr_file")

        html_repr = create_html_repr(
            obj=self,
            fields=["output_dir", "logs_dir"],
            display_paths=display_paths,
        )

        display(HTML(html_repr))


def init_session(
    host: str,
    syftbox_client: SyftBoxClient | None = None,
    mock_server: SyftEvents | None = None,
    syftbox_client_config_path: str | None = None,
    **config_kwargs,
) -> "RDSChatAnalysisClient":
    """
    Initialize a session with the RDSChatAnalysisClient.

    Args:
        host (str): The email of the remote datasite.
        syftbox_client (SyftBoxClient, optional): Pre-configured SyftBox client instance.
            Takes precedence over syftbox_client_config_path.
        mock_server (SyftEvents, optional): Server for testing. If provided, uses
            a mock in-process RPC connection.
        syftbox_client_config_path (str, optional): Path to client config file.
            Only used if syftbox_client is not provided.
        **config_kwargs: Additional configuration options for the RDSChatAnalysisClient.

    Returns:
        RDSChatAnalysisClient: The configured RDS client instance.
    """
    rds_client = _rds_init_session(
        host=host,
        syftbox_client=syftbox_client,
        mock_server=mock_server,
        syftbox_client_config_path=syftbox_client_config_path,
        **config_kwargs,
    )
    return RDSChatAnalysisClient.from_rdsclient(rds_client)


class RDSChatAnalysisClient(RDSClient):
    @classmethod
    def from_rdsclient(cls, rds_client: RDSClient) -> Self:
        return cls(rds_client.config, rds_client.rpc, rds_client.local_store)

    def _infer_dataset_name(self, dataset_name: str | None) -> str:
        if dataset_name is not None:
            return dataset_name

        datasets = self.dataset.get_all()
        if len(datasets) != 1:
            raise ValueError(
                "Multiple datasets found. Please specify the dataset_name explicitly."
            )

        return datasets[0].name

    def submit_job(
        self,
        vector_store_query: str,
        aggregation_fn: str,
        aggregation_query: str | None = None,
        max_vector_store_results: int = 5,
        distance_threshold: float = 0.5,
        filters: dict | None = None,
        dataset_name: str | None = None,
    ) -> Job:
        dataset_name = self._infer_dataset_name(dataset_name)
        job_config = {
            "vector_store_query": vector_store_query,
            "aggregation_fn": aggregation_fn,
            "aggregation_query": aggregation_query,
            "max_vector_store_results": max_vector_store_results,
            "distance_threshold": distance_threshold,
            "filters": filters,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            job_config_path = tmp_path / "job_config.json"
            job_code_path = tmp_path / "main.py"

            with open(job_config_path, "w") as f:
                json.dump(job_config, f, indent=2)

            with open(job_code_path, "w") as f:
                f.write(CHAT_ANALYSIS_CODE_TEMPLATE)

            job = self.jobs.submit(
                user_code_path=temp_dir,
                dataset_name=dataset_name,
                entrypoint="main.py",
                tags=["chat_analysis"],
            )

        return job

    def _get_job(self, job: str | UUID | Job) -> Job:
        if isinstance(job, str):
            uid = UUID(job)
        elif isinstance(job, UUID):
            uid = job
        elif isinstance(job, Job):
            uid = job.uid
        else:
            raise ValueError("Invalid job identifier type. Must be str, UUID, or Job.")

        return self.jobs.get(uid)

    def get_job_config(self, job: str | UUID | Job) -> dict:
        job = self._get_job(job)
        config_file = job.user_code.local_dir / "job_config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file {config_file} does not exist. Please contact the administrator."
            )

        with open(config_file, "r") as f:
            config = json.load(f)
        return config

    def get_job_result(
        self, job: str | UUID | Job, output_filename: str = "output/result.json"
    ) -> JobOutput:
        job = self._get_job(job)

        if job.status != JobStatus.shared:
            raise ValueError(
                f"Job {job.uid} has no shared results yet. Current status: {job.status.name}."
            )

        return JobOutput(execution_output_dir=job.output_path, job=job)

    def get_execution_result(self, job: str | UUID | Job) -> JobOutput:
        job_output_folder: Path = (
            self.config.runner_config.job_output_folder / job.uid.hex
        )

        if not job_output_folder.exists():
            raise FileNotFoundError(
                f"Output folder {job_output_folder} does not exist."
            )

        return JobOutput(execution_output_dir=job_output_folder, job=job)

    def _review_job_auto_reject_checks(self, job) -> tuple[bool, str | None]:
        """
        Run auto-reject checks for a job. Returns (failed, reason) where failed is True if any check failed.
        """
        local_code_dir = job.user_code.local_dir
        auto_reject_reason = None

        rich.print(":mag: Running auto-reject checks on the job...")
        if not local_code_dir.is_dir():
            rich.print(f":x: Local code directory {local_code_dir} does not exist.")
            auto_reject_reason = (
                f"Local code directory {local_code_dir} does not exist."
            )
        else:
            rich.print(":white_check_mark: Local code directory exists.")

        code_files = list(local_code_dir.iterdir())
        if len(code_files) != 2:
            rich.print(
                f":x: Expected 2 files in the code directory, found {len(code_files)}."
            )
            if not auto_reject_reason:
                auto_reject_reason = (
                    f"Expected 2 files in the code directory, found {len(code_files)}."
                )
        else:
            rich.print(":white_check_mark: Found 2 files in the code directory.")

        if "job_config.json" not in [f.name for f in code_files]:
            rich.print(":x: job_config.json file is missing in the code directory.")
            if not auto_reject_reason:
                auto_reject_reason = (
                    "job_config.json file is missing in the code directory."
                )
        else:
            rich.print(":white_check_mark: job_config.json found.")

        if "main.py" not in [f.name for f in code_files]:
            rich.print(":x: main.py file is missing in the code directory.")
            if not auto_reject_reason:
                auto_reject_reason = "main.py file is missing in the code directory."
        else:
            rich.print(":white_check_mark: main.py found.")

        code_str = (local_code_dir / "main.py").read_text()
        from rds_chat_analysis.job_functions import CHAT_ANALYSIS_CODE_TEMPLATE

        if not code_str.strip() == CHAT_ANALYSIS_CODE_TEMPLATE.strip():
            rich.print(":x: Job contains custom or modified code.")
            if not auto_reject_reason:
                auto_reject_reason = "Job contains custom or modified code."
        else:
            rich.print(":white_check_mark: main.py matches template.")

        if auto_reject_reason is not None:
            rich.print(f":x: Auto-reject checks failed. Reason: {auto_reject_reason}")
        return (auto_reject_reason is not None, auto_reject_reason)

    def review_job(self, job: str | UUID | Job) -> None:
        job = self._get_job(job)
        if job.status != JobStatus.pending_code_review:
            rich.print(
                f":x: Job {job.uid} is not in pending code review status. Current status: {job.status.name}."
            )
            return

        checks_failed, auto_reject_reason = self._review_job_auto_reject_checks(job)
        local_code_dir = job.user_code.local_dir
        rich.print("Job config:")
        with open(local_code_dir / "job_config.json", "r") as f:
            job_config = json.load(f)
            rich.print_json(data=job_config, highlight=False, indent=2)

        confirm_msg = "Do you want to reject this job?"
        job_is_denied = Confirm.ask(
            confirm_msg,
            default=checks_failed,
            show_default=True,
        )

        if job_is_denied:
            reason = Prompt.ask(
                "Please provide a reason for denying this job.",
                default=auto_reject_reason or "No reason provided",
                show_default=True,
            )
            rich.print(f":x: Job {job.uid} denied.")
            self.jobs.reject(
                job,
                reason=reason,
            )
        else:
            rich.print(f":white_check_mark: Job {job.uid} approved.")
            rich.print(
                "You can now run the job on the private data using: `client.run_private(job)`\n"
                "After the job is completed, you can share the results with the user using: `client.jobs.share_results(job)`"
            )
