from pathlib import PurePosixPath
import subprocess


def _tracked_files() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    ).stdout
    return output.splitlines()


def test_repository_has_no_tracked_runtime_or_secret_artifacts():
    forbidden = []
    for name in _tracked_files():
        path = PurePosixPath(name)
        if (
            "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or name.endswith(".db.lock")
            or name == ".env"
            or path.parts[:1] == ("result",)
            or name == "nvidia_docs/nvidia_docs/spiders/output.json"
        ):
            forbidden.append(name)

    assert forbidden == []
