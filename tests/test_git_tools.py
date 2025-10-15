import os
from pathlib import Path

import pytest

from tools.git import (
    GIT_REPO_STATUS_TOOL,
    GIT_ENSURE_CLONED_TOOL,
)


class DummyCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def tmp_workspace(tmp_path, monkeypatch):
    # Simulate workspace root
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    return tmp_path


def test_git_repo_status_not_available(monkeypatch, tmp_workspace):
    # Force git not available
    monkeypatch.setattr("tools.git._run", lambda args, cwd=None, timeout=8: (1, "", "err"))

    resp = GIT_REPO_STATUS_TOOL(path=str(tmp_workspace))
    assert '"ok": false' in resp
    assert '"error": "git_not_available"' in resp


def test_git_repo_status_ok_repo(monkeypatch, tmp_workspace):
    calls = {"runs": []}

    def fake_run(args, cwd=None, timeout=8):
        calls["runs"].append(list(args))
        cmd = " ".join(args)
        if args[:2] == ["git", "--version"]:
            return 0, "git version 2.42.0", ""
        if "rev-parse --is-inside-work-tree" in cmd:
            return 0, "true", ""
        if "remote get-url origin" in cmd:
            return 0, "https://example.com/repo.git", ""
        if "rev-parse --abbrev-ref HEAD" in cmd:
            return 0, "main", ""
        return 1, "", ""

    monkeypatch.setattr("tools.git._run", fake_run)

    resp = GIT_REPO_STATUS_TOOL(path=str(tmp_workspace))
    assert '"ok": true' in resp
    assert '"origin_url": "https://example.com/repo.git"' in resp
    assert '"branch": "main"' in resp


def test_git_ensure_cloned_exists(monkeypatch, tmp_workspace):
    # Ensure path exists -> no clone
    target = tmp_workspace / "repo"
    target.mkdir()

    # git available
    monkeypatch.setattr("tools.git._run", lambda args, cwd=None, timeout=8: (0, "git version 2.42.0", ""))

    resp = GIT_ENSURE_CLONED_TOOL(url="https://example.com/repo.git", dest=str(target))
    assert '"ok": true' in resp
    assert '"existed": true' in resp
    assert '"cloned": false' in resp
    assert f'"project_root": "{str(target).replace("\\", "\\\\")}"' in resp


def test_git_ensure_cloned_new(monkeypatch, tmp_workspace):
    repo_root = tmp_workspace
    target = repo_root / "repo"

    def fake_run(args, cwd=None, timeout=900):
        cmd = " ".join(args)
        # make --version succeed for availability checks
        if args[:2] == ["git", "--version"]:
            return 0, "git version 2.42.0", ""
        if args[:2] == ["git", "clone"]:
            # simulate clone by creating folder
            target.mkdir(parents=True, exist_ok=True)
            return 0, "cloned", ""
        if "remote get-url origin" in cmd:
            return 0, "https://example.com/repo.git", ""
        if "rev-parse --abbrev-ref HEAD" in cmd:
            return 0, "main", ""
        return 0, "", ""

    monkeypatch.setattr("tools.git._run", fake_run)

    resp = GIT_ENSURE_CLONED_TOOL(url="https://example.com/repo.git", dest=str(target))
    assert '"ok": true' in resp
    assert '"existed": false' in resp
    assert '"cloned": true' in resp
    assert '"remote_url": "https://example.com/repo.git"' in resp


