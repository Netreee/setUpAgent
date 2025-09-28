import os
import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest


# 被测函数
from run_singleV2 import run_single, PowerShellSession


@pytest.fixture(autouse=True)
def _env_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 在每个测试中设置 REPO_ROOT 指向临时目录
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    # 提供一个默认的 PROJECT_ROOT（可为空）
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    return repo_root


class _RunRecorder:
    """记录 run_single 组装的 envelope 以便断言。"""

    def __init__(self) -> None:
        self.last_envelope: str | None = None
        self.run_calls: list[str] = []

    async def fake_run(self, _self: PowerShellSession, cmd: str) -> Tuple[int, str]:
        # 记录
        self.run_calls.append(cmd)

        # 模拟 run_single 内部行为：
        # - 获取当前目录
        if "Get-Location | Select-Object -ExpandProperty Path" in cmd:
            return 0, os.environ.get("REPO_ROOT", "")

        # - 设置提示函数（start时）或其他轻量命令，返回空输出
        if cmd.startswith("function prompt {"):
            return 0, ""

        # - 主体 envelope：Push-Location; try { <ps_cmd> } finally { Pop-Location }
        if "Push-Location -LiteralPath $env:REPO_ROOT; try {" in cmd:
            self.last_envelope = cmd
            # 根据命令内容返回不同输出
            if "Get-ChildItem" in cmd:
                return 0, "a.txt\nb.txt\n"
            if "Get-Content" in cmd:
                return 0, "file content"
            if "pip install -r" in cmd:
                return 0, "installed"
            if "git clone" in cmd:
                return 0, "Cloning into..."
            return 0, "ok"

        # 默认
        return 0, ""


@pytest.mark.asyncio
async def test_decide_like_list_dir(monkeypatch: pytest.MonkeyPatch, _env_repo_root: Path):
    # 模拟 decide 节点给出的自然语言 -> 翻译器输出 PowerShell 命令
    async def fake_translate(_nl: str, _wd: Path | None) -> str:
        # 简单列目录（无路径参数）
        return "Get-ChildItem -Name"

    recorder = _RunRecorder()

    # 打桩 _translate 与会话的 _run/start
    monkeypatch.setattr("run_singleV2._translate", fake_translate, raising=True)
    monkeypatch.setattr(PowerShellSession, "start", lambda self: asyncio.sleep(0), raising=True)
    monkeypatch.setattr(PowerShellSession, "_run", recorder.fake_run, raising=True)

    token, result = await run_single("列出仓库根目录内容", session_token=None)

    assert result["exit_code"] == 0
    assert "a.txt" in (result.get("stdout") or "")
    assert "Push-Location -LiteralPath $env:REPO_ROOT" in (result.get("command") or "")


@pytest.mark.asyncio
async def test_sanitizer_literalpath_dedupe(monkeypatch: pytest.MonkeyPatch, _env_repo_root: Path):
    # 翻译器直接产出已包含 -LiteralPath 的命令，验证不会重复插入
    async def fake_translate(_nl: str, _wd: Path | None) -> str:
        return "Get-Content -LiteralPath (Join-Path $env:REPO_ROOT 'README.md') -Raw"

    recorder = _RunRecorder()
    monkeypatch.setattr("run_singleV2._translate", fake_translate, raising=True)
    monkeypatch.setattr(PowerShellSession, "start", lambda self: asyncio.sleep(0), raising=True)
    monkeypatch.setattr(PowerShellSession, "_run", recorder.fake_run, raising=True)

    token, result = await run_single("读取 README.md 内容", session_token=None)

    # envelope 中不应出现重复的 -LiteralPath
    assert recorder.last_envelope is not None
    assert "-LiteralPath -LiteralPath" not in recorder.last_envelope
    assert "Get-Content -LiteralPath (Join-Path $env:REPO_ROOT 'README.md') -Raw" in recorder.last_envelope
    assert result["exit_code"] == 0
    assert result["stdout"] == "file content"


@pytest.mark.asyncio
async def test_pip_install_r_not_literalpath(monkeypatch: pytest.MonkeyPatch, _env_repo_root: Path):
    # 验证不会把 pip install -r 参数改写为 -LiteralPath
    async def fake_translate(_nl: str, _wd: Path | None) -> str:
        return "pip install -r (Join-Path $env:REPO_ROOT 'requirements.txt')"

    recorder = _RunRecorder()
    monkeypatch.setattr("run_singleV2._translate", fake_translate, raising=True)
    monkeypatch.setattr(PowerShellSession, "start", lambda self: asyncio.sleep(0), raising=True)
    monkeypatch.setattr(PowerShellSession, "_run", recorder.fake_run, raising=True)

    token, result = await run_single("安装依赖", session_token=None)

    assert recorder.last_envelope is not None
    # 不应插入 -LiteralPath 在 -r 后面
    assert "pip install -r -LiteralPath" not in recorder.last_envelope
    assert "pip install -r (Join-Path $env:REPO_ROOT 'requirements.txt')" in recorder.last_envelope
    assert result["stdout"] == "installed"


@pytest.mark.asyncio
async def test_git_clone_dest_rewrite(monkeypatch: pytest.MonkeyPatch, _env_repo_root: Path):
    # 验证 git clone 到 $env:REPO_ROOT 会被改写为子目录
    async def fake_translate(_nl: str, _wd: Path | None) -> str:
        return "git clone https://github.com/huggingface/transformers.git $env:REPO_ROOT"

    recorder = _RunRecorder()
    monkeypatch.setattr("run_singleV2._translate", fake_translate, raising=True)
    monkeypatch.setattr(PowerShellSession, "start", lambda self: asyncio.sleep(0), raising=True)
    monkeypatch.setattr(PowerShellSession, "_run", recorder.fake_run, raising=True)

    token, result = await run_single("克隆 transformers 仓库到仓库根目录", session_token=None)

    assert recorder.last_envelope is not None
    assert "(Join-Path $env:REPO_ROOT 'transformers')" in recorder.last_envelope
    assert result["stdout"].startswith("Cloning into")


