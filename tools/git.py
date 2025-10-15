from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from config import get_config
from agent.debug import dispInfo, debug
from tools.base import tool_response


def _get_workspace_root() -> Path:
    try:
        repo_root = os.environ.get("REPO_ROOT")
        if repo_root:
            return Path(repo_root).resolve()
    except Exception:
        pass
    try:
        cfg = get_config()
        return Path(cfg.agent_work_root).resolve()
    except Exception:
        return Path.cwd().resolve()


def _resolve_and_guard(path: str | os.PathLike[str]) -> Tuple[bool, Optional[str], Optional[Path]]:
    try:
        root = _get_workspace_root()
        p = Path(path)
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
        try:
            p.relative_to(root)
        except Exception:
            return False, "path_out_of_root", None
        return True, None, p
    except Exception:
        return False, "resolve_error", None


def _run(args: List[str], cwd: Optional[Path] = None, timeout: int = 600) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        return int(proc.returncode), (proc.stdout or ""), (proc.stderr or "")
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def _git_available() -> bool:
    code, out, err = _run(["git", "--version"], timeout=8)
    return code == 0


def _repo_name_from_url(url: str) -> str:
    u = url.strip().rstrip("/")
    if "/" in u:
        u = u.split("/")[-1]
    if u.lower().endswith(".git"):
        u = u[:-4]
    return u or "repo"


def _is_git_repo(path: Path) -> bool:
    code, out, _ = _run(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"], timeout=8)
    return code == 0 and (out.strip().lower() == "true")


def _origin_url(path: Path) -> Optional[str]:
    code, out, _ = _run(["git", "-C", str(path), "remote", "get-url", "origin"], timeout=8)
    if code == 0:
        return out.strip() or None
    return None


def _current_branch(path: Path) -> Optional[str]:
    code, out, _ = _run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], timeout=8)
    if code == 0:
        b = out.strip()
        return None if b in ("HEAD", "") else b
    return None


@tool("git_repo_status")
@dispInfo("git_repo_status")
def GIT_REPO_STATUS_TOOL(path: Optional[str] = None) -> str:
    """检测目录的 Git 状态，返回是否为仓库、origin URL、当前分支等。"""
    root = _get_workspace_root()
    target = path or str(root)
    okg, viol, p = _resolve_and_guard(target)
    if not okg or p is None:
        return tool_response(
            tool="git_repo_status",
            ok=False,
            data={"path": str(target)},
            error=viol or "invalid_path",
        )
    if not _git_available():
        return tool_response(
            tool="git_repo_status",
            ok=False,
            data={"path": str(p)},
            error="git_not_available",
        )
    try:
        is_repo = _is_git_repo(p)
        origin = _origin_url(p) if is_repo else None
        branch = _current_branch(p) if is_repo else None
        try:
            debug.note("git_is_repo", is_repo)
            debug.note("git_origin", origin)
            debug.note("git_branch", branch)
        except Exception:
            pass
        return tool_response(
            tool="git_repo_status",
            ok=True,
            data={
                "path": str(p),
                "is_repo": bool(is_repo),
                "origin_url": origin,
                "branch": branch,
            },
        )
    except Exception as e:
        return tool_response(
            tool="git_repo_status",
            ok=False,
            data={"path": str(p)},
            error=f"{type(e).__name__}: {e}",
        )


@tool("git_ensure_cloned")
@dispInfo("git_ensure_cloned")
def GIT_ENSURE_CLONED_TOOL(url: str, dest: Optional[str] = None, depth: int = 1, sparse: bool = True, branch: Optional[str] = None) -> str:
    """确保仓库已在工作区内可用：
    - 如目标目录已存在则不再克隆，直接返回路径等信息（避免重复克隆）
    - 如不存在则执行浅克隆，并返回结构化信息
    返回字段包含 repo_root/project_root/project_name 等，便于上层合并 facts
    """
    if not url or not isinstance(url, str):
        return tool_response(
            tool="git_ensure_cloned",
            ok=False,
            data={"url": url or ""},
            error="invalid_url",
        )
    if not _git_available():
        return tool_response(
            tool="git_ensure_cloned",
            ok=False,
            data={"url": url},
            error="git_not_available",
        )

    work_root = _get_workspace_root()
    repo_name = _repo_name_from_url(url)

    # 解析目标路径
    target_path: Path
    if dest is not None:
        # 将 dest='.' 或空字符串视为“未提供目标目录”，改为使用默认路径 work_root/repo_name
        dest_str = str(dest).strip()
        if dest_str in ("", ".", "./"):
            target_path = (work_root / repo_name).resolve()
        else:
            okg, viol, p = _resolve_and_guard(dest)
            if not okg or p is None:
                return tool_response(
                    tool="git_ensure_cloned",
                    ok=False,
                    data={"url": url, "dest": dest},
                    error=viol or "invalid_dest",
                )
            # 若解析后的路径等于工作区根，则仍使用 repo_name 子目录，避免误判“已存在”
            if p == work_root:
                target_path = (work_root / repo_name).resolve()
            else:
                target_path = p
    else:
        target_path = (work_root / repo_name).resolve()

    existed = target_path.exists()
    cloned = False
    stdout_all = ""
    stderr_all = ""

    # 已存在则直接返回（无论是否为 git 仓库，均避免重复克隆，交由上层决定后续操作）
    if existed:
        try:
            debug.note("git_target_exists", str(target_path))
        except Exception:
            pass
        is_repo_now = _is_git_repo(target_path)
        origin = _origin_url(target_path) if is_repo_now else None
        branch_now = _current_branch(target_path) if is_repo_now else None
        return tool_response(
            tool="git_ensure_cloned",
            ok=True,
            data={
                "existed": True,
                "cloned": False,
                "repo_root": str(work_root),
                "project_root": str(target_path),
                "project_name": repo_name,
                "remote_url": origin or url,
                "branch": branch_now,
                "is_repo": bool(is_repo_now),
            },
        )

    # 不存在：执行浅克隆
    args: List[str] = ["git", "clone"]
    if depth and int(depth) > 0:
        args += ["--depth", str(int(depth))]
    if sparse:
        # 在新 git 中可用：最小化传输体积
        args += ["--filter=blob:none"]
    if branch:
        args += ["-b", str(branch)]
    args += [url, str(target_path)]

    try:
        code, out, err = _run(args, cwd=work_root, timeout=900)
        stdout_all = out
        stderr_all = err
        try:
            debug.note("git_clone_code", code)
            debug.note("git_clone_stdout", (out or "")[:500])
            debug.note("git_clone_stderr", (err or "")[:500])
        except Exception:
            pass
        if code != 0:
            # 即便失败，也返回结构，便于上层根据错误文案做判定（如 already exists）
            return tool_response(
                tool="git_ensure_cloned",
                ok=False,
                data={
                    "existed": False,
                    "cloned": False,
                    "repo_root": str(work_root),
                    "project_root": str(target_path),
                    "project_name": repo_name,
                    "remote_url": url,
                    "branch": None,
                    "stdout": stdout_all,
                    "stderr": stderr_all,
                },
                error="git_clone_failed",
            )
        cloned = True
    except Exception as e:
        return tool_response(
            tool="git_ensure_cloned",
            ok=False,
            data={
                "existed": False,
                "cloned": False,
                "repo_root": str(work_root),
                "project_root": str(target_path),
                "project_name": repo_name,
                "remote_url": url,
            },
            error=f"{type(e).__name__}: {e}",
        )

    # 克隆完成后补充信息
    is_repo_now = _is_git_repo(target_path)
    origin = _origin_url(target_path)
    branch_now = _current_branch(target_path)
    return tool_response(
        tool="git_ensure_cloned",
        ok=True,
        data={
            "existed": False,
            "cloned": cloned,
            "repo_root": str(work_root),
            "project_root": str(target_path),
            "project_name": repo_name,
            "remote_url": origin or url,
            "branch": branch_now,
            "is_repo": bool(is_repo_now),
            "stdout": stdout_all,
            "stderr": stderr_all,
        },
    )


