import shutil
from pathlib import Path


def prepare_job_dependencies(submission_dir: Path):
    if not submission_dir.exists():
        submission_dir.mkdir(parents=True, exist_ok=True)

    # if submission dir is not empty, error
    if any(submission_dir.iterdir()):
        raise ValueError(f"Submission directory {submission_dir} is not empty.")

    from rds_chat_analysis import REPO_ROOT

    # Copy necessary files and directories to the job submission directory
    paths_to_copy = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "src",
        REPO_ROOT / "README.md",
    ]

    def ignore_pycache(dir, contents):
        return [c for c in contents if c == "__pycache__"]

    for path in paths_to_copy:
        if path.is_file():
            shutil.copy(path, submission_dir / path.name)
        elif path.is_dir():
            shutil.copytree(
                path,
                submission_dir / path.name,
                dirs_exist_ok=True,
                ignore=ignore_pycache,
            )


def test_job_submission():
    return "Test passed"
