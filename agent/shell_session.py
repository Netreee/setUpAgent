from __future__ import annotations

import subprocess
import threading
import queue
import uuid
import os
import base64
from pathlib import Path
from typing import Dict, Optional, Tuple
from config import get_config


class _ProcWrapper:
    def __init__(self, proc: subprocess.Popen, work_dir: Path) -> None:
        self.proc = proc
        self.work_dir = work_dir
        self.lock = threading.Lock()


class ShellSessionManager:
    """基于持久 PowerShell 进程的会话管理器。"""

    def __init__(self) -> None:
        self._sessions: Dict[str, _ProcWrapper] = {}

    def create_session(self, shell: str = "powershell", cwd: Optional[str] = None) -> str:
        config = get_config()
        work = Path(cwd or config.agent_work_root)
        work.mkdir(exist_ok=True)

        if os.name == 'nt':
            # 启动持久 powershell 进程（-NoExit 保持进程）
            ps = subprocess.Popen(
                ["powershell", "-NoProfile", "-NoExit", "-Command", "-"],
                cwd=str(work),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        else:
            # 非 Windows 简化为 /bin/bash 持久进程
            ps = subprocess.Popen(
                ["/bin/bash", "-i"],
                cwd=str(work),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )

        sid = uuid.uuid4().hex[:8]
        self._sessions[sid] = _ProcWrapper(ps, work)
        return sid

    def run_in_session(self, session_id: str, command: str, timeout: int = 60) -> Tuple[int, str, str, str, str]:
        wrap = self._sessions.get(session_id)
        if wrap is None:
            raise RuntimeError("无效的 session_id")

        proc = wrap.proc
        sentinel = uuid.uuid4().hex
        ps_cmd = f"\n$ErrorActionPreference='Continue'; $global:LASTEXITCODE=$null; {command}; $code = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} else {{ 0 }}; Write-Output '<__END__:{sentinel}:'+$code+'>'\n"

        with wrap.lock:
            assert proc.stdin and proc.stdout and proc.stderr
            proc.stdin.write(ps_cmd)
            proc.stdin.flush()

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            code: Optional[int] = None

            # 读取到哨兵行为止
            proc.stdout.flush()
            proc.stderr.flush()
            import time
            start = time.time()
            while True:
                if time.time() - start > timeout:
                    code = -2
                    break
                line = proc.stdout.readline()
                if not line:
                    # 进程可能已退出
                    break
                if line.startswith(f"<__END__:{sentinel}:"):
                    try:
                        code = int(line.strip().split(":")[-1].rstrip(">"))
                    except Exception:
                        code = 0
                    break
                stdout_chunks.append(line)
            # 尽量清空错误输出缓冲
            while True:
                try:
                    if proc.stderr is None:
                        break
                    proc.stderr.flush()
                    if not proc.stderr.readable():
                        break
                    if not proc.stderr.peek():  # type: ignore[attr-defined]
                        break
                    stderr_chunks.append(proc.stderr.readline())
                except Exception:
                    break

        return code or 0, "".join(stdout_chunks).strip(), "".join(stderr_chunks).strip(), command, str(wrap.work_dir)

    def close_session(self, session_id: str) -> None:
        wrap = self._sessions.pop(session_id, None)
        if wrap is None:
            return
        try:
            if wrap.proc and wrap.proc.poll() is None:
                try:
                    if wrap.proc.stdin:
                        wrap.proc.stdin.write("\nexit\n")
                        wrap.proc.stdin.flush()
                except Exception:
                    pass
                wrap.proc.terminate()
        except Exception:
            pass


# 全局单例
manager = ShellSessionManager()


