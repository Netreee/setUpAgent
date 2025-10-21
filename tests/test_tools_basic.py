import json
from pathlib import Path

import sys
import os

# 获取当前测试文件的目录（tests 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录是 tests 目录的上层目录
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 Python 搜索路径
sys.path.append(project_root)

import pytest


# 通过 tools 包导入工具（被 @tool 装饰后对象可通过 .invoke 调用）
from tools import (
    FILES_EXISTS_TOOL,
    FILES_STAT_TOOL,
    FILES_LIST_TOOL,
    FILES_READ_TOOL,
    FILES_FIND_TOOL,
    FILES_READ_SECTION_TOOL,
    FILES_READ_RANGE_TOOL,
    FILES_GREP_TOOL,
    MD_OUTLINE_TOOL,
    PYENV_PYTHON_INFO_TOOL,
    PYENV_TOOL_VERSIONS_TOOL,
    PYENV_PARSE_PYPROJECT_TOOL,
    PYENV_SELECT_INSTALLER_TOOL,
)
import tools.pyenv as pyenv_mod


@pytest.fixture(autouse=True)
def _env_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    return repo_root


def _loads(s: str):
    return json.loads(s)


def test_fs_exists_and_stat(_env_repo_root: Path):
    p = _env_repo_root / "a.txt"
    p.write_text("hello", encoding="utf-8")

    res = _loads(FILES_EXISTS_TOOL.invoke({"path": "a.txt"}))
    assert res["ok"] is True and res["data"]["exists"] is True
    assert str(p).replace("\\", "/").endswith("/a.txt")

    st_file = _loads(FILES_STAT_TOOL.invoke({"path": "a.txt"}))
    assert st_file["ok"] is True and st_file["data"]["type"] == "file" and st_file["data"]["size"] > 0

    st_missing = _loads(FILES_STAT_TOOL.invoke({"path": "nope.txt"}))
    assert st_missing["ok"] is True and st_missing["data"]["type"] == "missing"


def test_fs_list_read_find(_env_repo_root: Path):
    # 结构: repo_root/{a.txt, sub/{b.txt, d/{c.txt}}}
    (_env_repo_root / "a.txt").write_text("A" * 10, encoding="utf-8")
    sub = _env_repo_root / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("B" * 5, encoding="utf-8")
    d = sub / "d"
    d.mkdir()
    (d / "c.txt").write_text("C" * 3, encoding="utf-8")

    # files_list: 递归 + 过滤
    lst = _loads(
        FILES_LIST_TOOL.invoke(
            {"path": ".", "files_only": True, "recurse": True, "patterns": ["*.txt"], "limit": 10}
        )
    )
    names = {e["name"] for e in lst["data"]["entries"]}
    assert {"a.txt", "b.txt", "c.txt"}.issubset(names)
    assert lst["data"]["truncated"] is False

    # files_read: raw/head/tail
    raw = _loads(FILES_READ_TOOL.invoke({"path": "a.txt", "mode": "raw", "max_bytes": 5}))
    assert raw["ok"] is True and raw["data"]["truncated"] is True and raw["data"]["content"] == "A" * 5

    head = _loads(FILES_READ_TOOL.invoke({"path": "a.txt", "mode": "head", "max_bytes": 3}))
    assert head["data"]["content"] == "A" * 3 and head["data"]["truncated"] is True

    tail = _loads(FILES_READ_TOOL.invoke({"path": "a.txt", "mode": "tail", "max_bytes": 4}))
    assert tail["data"]["content"] == "A" * 4 and tail["data"]["truncated"] is True

    # files_find: include/exclude/first_only
    found = _loads(
        FILES_FIND_TOOL.invoke(
            {"start_dir": ".", "include_globs": ["*.txt"], "exclude_globs": ["b.*"], "first_only": False}
        )
    )
    assert found["ok"] is True and any(x.endswith("a.txt") for x in found["data"]["matches"]) and not any(
        x.endswith("b.txt") for x in found["data"]["matches"]
    )

    first = _loads(
        FILES_FIND_TOOL.invoke({"start_dir": ".", "include_globs": ["*.txt"], "first_only": True})
    )
    assert first["ok"] is True and len(first["data"]["matches"]) == 1

    # files_read_section: extract specific lines
    sec = _loads(FILES_READ_SECTION_TOOL.invoke({"path": "a.txt", "start_line": 1, "end_line": 1}))
    assert sec["ok"] is True and sec["data"]["content"].startswith("A") and len(sec["data"]["content"]) >= 1

    # files_read_range: byte-range read
    rng = _loads(FILES_READ_RANGE_TOOL.invoke({"path": "a.txt", "offset": 0, "length": 5}))
    assert rng["ok"] is True and rng["data"]["content"] == "A" * 5 and rng["data"]["truncated"] is True

    # files_grep: regex search
    grep = _loads(
        FILES_GREP_TOOL.invoke({"start_dir": ".", "patterns": ["^A+$"], "include_globs": ["*.txt"], "first_only": True})
    )
    assert grep["ok"] is True and grep["data"]["matches"] and grep["data"]["matches"][0]["line"].startswith("A")

    # md_outline: build headings index
    md = _env_repo_root / "README.md"
    md.write_text("# Title\n\n## Sec1\ntext\n\n## Sec2\n", encoding="utf-8")
    outline = _loads(MD_OUTLINE_TOOL.invoke({"path": "README.md"}))
    assert outline["ok"] is True and outline["data"]["count"] >= 3


def test_pyenv_python_info_and_parse(monkeypatch: pytest.MonkeyPatch, _env_repo_root: Path):
    # 伪造 where/py -0p 与 --version 输出
    def fake_run_cmd(args, timeout: int = 10):
        if args[:2] == ["where", "python"]:
            return 0, "C:\\Python\\python.exe\n", ""
        if len(args) >= 2 and args[0].endswith("python.exe") and args[1] == "--version":
            return 0, "Python 3.12.5", ""
        if args[:2] == ["py", "-0p"]:
            return 0, " -3.12-64: C:\\Python\\python.exe\n", ""
        return 1, "", "unsupported"

    monkeypatch.setattr(pyenv_mod, "_run_cmd", fake_run_cmd, raising=True)

    info = _loads(PYENV_PYTHON_INFO_TOOL.invoke({}))
    assert info["ok"] is True and info["active"]["path"].endswith("python.exe")
    assert info["candidates"] and any(c.get("version") for c in info["candidates"])  # 有版本

    # 创建一个最小 pyproject.toml
    (_env_repo_root / "pyproject.toml").write_text(
        """
        [build-system]
        requires = ["setuptools"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "demo"
        version = "0.1.0"
        dependencies = ["requests>=2"]

        [tool.poetry]
        name = "demo"
        """.strip(),
        encoding="utf-8",
    )

    parsed = _loads(PYENV_PARSE_PYPROJECT_TOOL.invoke({}))
    assert parsed["ok"] is True and parsed["exists"] is True
    assert parsed["project"]["name"] == "demo"
    assert parsed["tool"]["poetry"] is True


def test_pyenv_tool_versions_and_select_installer(monkeypatch: pytest.MonkeyPatch, _env_repo_root: Path):
    # where: uv/pip 有，其他无
    def fake_where(name: str, timeout: int = 8):
        if name == "uv":
            return ["C:/bin/uv.exe"]
        if name == "pip":
            return ["C:/Python/Scripts/pip.exe"]
        return []

    def fake_run_cmd(args, timeout: int = 10):
        if args and args[0].endswith("uv.exe") and args[1] == "--version":
            return 0, "uv 0.1.0", ""
        if args and args[0].endswith("pip.exe") and args[1] == "--version":
            return 0, "pip 23.1", ""
        return 1, "", "unsupported"

    monkeypatch.setattr(pyenv_mod, "_where", fake_where, raising=True)
    monkeypatch.setattr(pyenv_mod, "_run_cmd", fake_run_cmd, raising=True)

    vers = _loads(PYENV_TOOL_VERSIONS_TOOL.invoke({"tools": ["uv", "pip", "poetry"]}))
    assert vers["ok"] is True and vers["tools"]["uv"]["exists"] is True and vers["tools"]["pip"]["exists"] is True
    assert vers["tools"]["poetry"]["exists"] is False

    # 有 requirements 且有 uv，优先 uv
    (_env_repo_root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    sel = _loads(PYENV_SELECT_INSTALLER_TOOL.invoke({}))
    assert sel["ok"] is True and sel["installer"] in ("uv", "pip")
    # 在我们的伪造情形下应为 uv
    assert sel["installer"] == "uv"



def test_files_exists_schema_mismatch_paths_param():
    """复现日志中的 schema 校验问题：传入 paths 而非 path 应触发验证错误。"""
    import pytest
    with pytest.raises(Exception):
        # 缺少必填的 path 字段，工具层应抛出校验异常
        FILES_EXISTS_TOOL.invoke({"paths": ["README.md", "README.rst"]})


def test_files_list_with_facts_token_reports_not_a_directory(_env_repo_root: Path):
    """复现日志中的占位符未解析：传入 facts.project_root 会被按字面拼接，导致不是目录。"""
    import json
    out = FILES_LIST_TOOL.invoke({"path": "facts.project_root"})
    data = json.loads(out)
    assert data["ok"] is False
    assert data.get("error") == "not_a_directory"
    # 目录字段应包含字面 'facts.project_root'，表明未被解析
    assert str(data.get("dir", "")).replace("\\", "/").endswith("/facts.project_root")
