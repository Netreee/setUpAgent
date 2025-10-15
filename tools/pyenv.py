from __future__ import annotations

import json
import os
import re
import shlex
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


def _json_result(**kwargs: Any) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


def _run_cmd(args: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        return int(proc.returncode), out, err
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def _where(exe: str, timeout: int = 8) -> List[str]:
    paths: List[str] = []
    code, out, _ = _run_cmd(["where", exe], timeout=timeout)
    if code == 0 and out:
        for line in out.splitlines():
            line = line.strip()
            if line:
                paths.append(line)
    return paths


def _probe_version(cmd_path: str, version_args: List[str]) -> Tuple[bool, Optional[str]]:
    code, out, err = _run_cmd([cmd_path, *version_args], timeout=8)
    if code != 0:
        # 有些工具将版本写到 stderr
        text = (out or err or "").strip()
    else:
        text = (out or err or "").strip()
    if not text:
        return False, None
    # 抽取第一个形如 X.Y 或 X.Y.Z 的版本号
    m = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", text)
    return True, (m.group(0) if m else text.splitlines()[0].strip())


@tool("pyenv_python_info")
@dispInfo("pyenv_python_info")
def PYENV_PYTHON_INFO_TOOL() -> str:
    """探测可用的 Python 解释器。"""
    candidates: List[Dict[str, Any]] = []
    # 1) 使用 where python
    python_paths = _where("python")
    try:
        debug.note("where_python_count", len(python_paths))
    except Exception:
        pass
    for p in python_paths:
        ok, ver = _probe_version(p, ["--version"])
        candidates.append({"path": p, "version": ver if ok else None})
    # 2) Windows py 启动器
    launcher = {}
    code, out, err = _run_cmd(["py", "-0p"], timeout=8)
    if code == 0 and out:
        py_paths = []
        for line in out.splitlines():
            s = line.strip()
            if s:
                # 行格式例: -V: path
                if ":" in s:
                    s = s.split(":", 1)[-1].strip()
                py_paths.append(s)
                ok, ver = _probe_version(s, ["--version"])
                if s not in [c["path"] for c in candidates]:
                    candidates.append({"path": s, "version": ver if ok else None})
        launcher = {"paths": py_paths}
        try:
            debug.note("py_launcher_paths", py_paths)
        except Exception:
            pass
    elif err:
        launcher = {"error": err}
    # 活动解释器（优先 PATH 中第一个）
    active = candidates[0] if candidates else None
    try:
        debug.note("candidates_count", len(candidates))
        debug.note("active", active)
    except Exception:
        pass
    return tool_response(
        tool="pyenv_python_info",
        ok=True,
        data={
            "active": active,
            "candidates": candidates,
            "launcher": launcher,
            "executable": active["path"] if active else None,
            "version": active["version"] if active else None
        }
    )


@tool("pyenv_tool_versions")
@dispInfo("pyenv_tool_versions")
def PYENV_TOOL_VERSIONS_TOOL(tools: List[str]) -> str:
    """探测工具(如 uv/pip/poetry/pdm/conda/pipenv)是否存在及版本。"""
    result: Dict[str, Any] = {}
    try:
        debug.note("tools_requested", tools)
    except Exception:
        pass
    for name in tools or []:
        paths = _where(name)
        if not paths:
            result[name] = {"exists": False}
            continue
        # 取第一个
        path = paths[0]
        version_args = ["--version"]
        if name == "conda":
            version_args = ["--version"]
        ok, ver = _probe_version(path, version_args)
        result[name] = {"exists": True, "path": path, "version": ver if ok else None}
    try:
        existing_tools = [k for k, v in result.items() if v.get("exists")]
        debug.note("existing_tools", existing_tools)
    except Exception:
        pass
    return tool_response(
        tool="pyenv_tool_versions",
        ok=True,
        data={"tools": result}
    )


def _load_toml(path: Path) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    try:
        try:
            import tomllib  # py311+
        except Exception:  # pragma: no cover
            tomllib = None
        if tomllib is None:
            try:
                import tomli as tomllib  # type: ignore
            except Exception:
                return False, None, "tomllib_unavailable"
        with open(path, "rb") as f:
            data = tomllib.load(f)  # type: ignore
        return True, data, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


@tool("pyenv_parse_pyproject")
@dispInfo("pyenv_parse_pyproject")
def PYENV_PARSE_PYPROJECT_TOOL(pyproject_path: Optional[str] = None) -> str:
    """读取并解析 pyproject.toml，提取关键信息。"""
    # 解析路径
    default_path = None
    try:
        proj = os.environ.get("PROJECT_ROOT") or os.environ.get("REPO_ROOT") or get_config().agent_work_root
        default_path = str(Path(proj).resolve() / "pyproject.toml")
    except Exception:
        default_path = str(_get_workspace_root() / "pyproject.toml")
    target = pyproject_path or default_path
    try:
        debug.note("target_path", target)
    except Exception:
        pass
    okg, viol, p = _resolve_and_guard(target)
    if not okg or p is None:
        return tool_response(
            tool="pyenv_parse_pyproject",
            ok=False,
            data={"path": str(target), "exists": False},
            error=viol or "unknown_error"
        )
    if not p.exists():
        try:
            debug.note("file_exists", False)
        except Exception:
            pass
        return tool_response(
            tool="pyenv_parse_pyproject",
            ok=True,
            data={"path": str(p), "exists": False}
        )
    try:
        debug.note("file_exists", True)
        debug.note("resolved_path", str(p))
    except Exception:
        pass
    ok, data, err = _load_toml(p)
    if not ok or data is None:
        return tool_response(
            tool="pyenv_parse_pyproject",
            ok=False,
            data={"path": str(p), "exists": True},
            error=err or "parse_error"
        )
    # 提取关键字段
    build = data.get("build-system", {}) if isinstance(data.get("build-system", {}), dict) else {}
    backend = build.get("build-backend")
    project = data.get("project", {}) if isinstance(data.get("project", {}), dict) else {}
    dependencies = project.get("dependencies", []) if isinstance(project.get("dependencies", []), list) else []
    optional_deps = project.get("optional-dependencies", {}) if isinstance(project.get("optional-dependencies", {}), dict) else {}
    scripts = project.get("scripts", {}) if isinstance(project.get("scripts", {}), dict) else {}
    tool_sec = data.get("tool", {}) if isinstance(data.get("tool", {}), dict) else {}
    poetry = tool_sec.get("poetry") if isinstance(tool_sec.get("poetry"), dict) else None
    pdm = tool_sec.get("pdm") if isinstance(tool_sec.get("pdm"), dict) else None
    uv = tool_sec.get("uv") if isinstance(tool_sec.get("uv"), dict) else None
    try:
        debug.note("backend", backend)
        debug.note("project_name", project.get("name"))
        debug.note("dependencies_count", len(dependencies))
        debug.note("has_poetry", bool(poetry))
        debug.note("has_pdm", bool(pdm))
        debug.note("has_uv", bool(uv))
    except Exception:
        pass
    return tool_response(
        tool="pyenv_parse_pyproject",
        ok=True,
        data={
            "path": str(p),
            "exists": True,
            "backend": backend,
            "project_name": project.get("name"),
            "project_version": project.get("version"),
            "dependencies": dependencies,
            "has_dependencies": len(dependencies) > 0,
            "optional_dependencies": optional_deps,
            "scripts": scripts,
            "has_poetry_section": bool(poetry),
            "has_pdm_section": bool(pdm),
            "has_uv_section": bool(uv),
        }
    )


@tool("pyenv_select_installer")
@dispInfo("pyenv_select_installer")
def PYENV_SELECT_INSTALLER_TOOL(project_root: Optional[str] = None) -> str:
    """基于项目文件与 pyproject 内容，选择安装器(uv|poetry|pdm|conda|pip|pipenv|none)。
    纯规则实现，不调用 LLM，不做实际安装。
    """
    # 根目录
    root_dir: Path
    if project_root:
        okg, viol, p = _resolve_and_guard(project_root)
        if not okg or p is None:
            return tool_response(
                tool="pyenv_select_installer",
                ok=False,
                data={"installer": "none"},
                error=viol or "invalid_project_root"
            )
        root_dir = p
    else:
        try:
            env_proj = os.environ.get("PROJECT_ROOT") or os.environ.get("REPO_ROOT")
            root_dir = Path(env_proj).resolve() if env_proj else _get_workspace_root()
        except Exception:
            root_dir = _get_workspace_root()

    def _exists(name: str) -> bool:
        try:
            return (root_dir / name).exists()
        except Exception:
            return False

    # 文件证据
    evidence: Dict[str, Any] = {
        "pyproject": str(root_dir / "pyproject.toml"),
        "poetry_lock": _exists("poetry.lock"),
        "pdm_lock": _exists("pdm.lock") or _exists("pdm.lock.json") or _exists("pdm.lock.yml"),
        "uv_lock": _exists("uv.lock") or _exists("uv.lock.json"),
        "requirements": any(_exists(n) for n in ["requirements.txt", "requirements.in", "requirements-dev.txt"]),
        "conda_env": any(_exists(n) for n in ["environment.yml", "environment.yaml"]),
    }

    # 解析 pyproject 判断工具声明
    pyproject_path = root_dir / "pyproject.toml"
    tool_declared = {"uv": False, "poetry": False, "pdm": False}
    backend = None
    if pyproject_path.exists():
        ok, data, _ = _load_toml(pyproject_path)
        if ok and data:
            tool_sec = data.get("tool", {}) if isinstance(data.get("tool", {}), dict) else {}
            tool_declared["uv"] = bool(tool_sec.get("uv"))
            tool_declared["poetry"] = bool(tool_sec.get("poetry"))
            tool_declared["pdm"] = bool(tool_sec.get("pdm"))
            bs = data.get("build-system", {}) if isinstance(data.get("build-system", {}), dict) else {}
            backend = bs.get("build-backend")
    evidence.update({"tool_declared": tool_declared, "build_backend": backend})

   
    tool_names = ["uv", "poetry", "pdm", "pip", "conda", "pipenv"]
    tools_info_data: Dict[str, Any] = {}
    for name in tool_names:
        paths = _where(name)
        if not paths:
            tools_info_data[name] = {"exists": False}
            continue
        path = paths[0]
        version_args = ["--version"]
        ok_ver, ver = _probe_version(path, version_args)
        tools_info_data[name] = {"exists": True, "path": path, "version": ver if ok_ver else None}

    # 选择规则（确定性）
    installer = "none"
    reason = []
    def _has(name: str) -> bool:
        return bool(tools_info_data.get(name, {}).get("exists"))
    
    try:
        debug.note("project_root", str(root_dir))
        debug.note("evidence", evidence)
    except Exception:
        pass

    if tool_declared["uv"] or evidence["uv_lock"]:
        if _has("uv"):
            installer = "uv"; reason.append("tool.uv 或 uv.lock 存在，且本机有 uv")
        else:
            reason.append("推荐 uv 但本机缺少 uv")
    elif tool_declared["poetry"] or evidence["poetry_lock"]:
        if _has("poetry"):
            installer = "poetry"; reason.append("tool.poetry 或 poetry.lock 存在，且本机有 poetry")
        else:
            reason.append("推荐 poetry 但本机缺少 poetry")
    elif tool_declared["pdm"] or evidence["pdm_lock"]:
        if _has("pdm"):
            installer = "pdm"; reason.append("tool.pdm 或 pdm.lock 存在，且本机有 pdm")
        else:
            reason.append("推荐 pdm 但本机缺少 pdm")
    elif evidence["conda_env"] and _has("conda"):
        installer = "conda"; reason.append("存在 environment.yml 且本机有 conda")
    elif evidence["requirements"]:
        if _has("uv"):
            installer = "uv"; reason.append("存在 requirements，优先使用 uv")
        elif _has("pip"):
            installer = "pip"; reason.append("存在 requirements，使用 pip")
    else:
        # 无明显信号：如果 pyproject 有 [project.dependencies]，则 uv>pip；否则 none
        if pyproject_path.exists():
            ok, data, _ = _load_toml(pyproject_path)
            deps = []
            if ok and data:
                proj = data.get("project", {}) if isinstance(data.get("project", {}), dict) else {}
                if isinstance(proj.get("dependencies"), list):
                    deps = proj.get("dependencies")
            if deps:
                if _has("uv"):
                    installer = "uv"; reason.append("pyproject 有 dependencies，且本机有 uv")
                elif _has("pip"):
                    installer = "pip"; reason.append("pyproject 有 dependencies，使用 pip")
    if not reason:
        reason.append("未找到明确证据，返回 none")
    
    try:
        debug.note("selected_installer", installer)
        debug.note("reason", "; ".join(reason))
    except Exception:
        pass

    return tool_response(
        tool="pyenv_select_installer",
        ok=True,
        data={
            "installer": installer,
            "reason": "; ".join(reason),
            "evidence": {**evidence, "tools": tools_info_data}
        }
    )


