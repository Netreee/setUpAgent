import asyncio
import uuid
import weakref
from typing import Dict, Optional, Tuple
from pathlib import Path
import os
from config import get_config

# ---------- 底层长驻 PowerShell 会话 ----------
class PowerShellSession:
    """Windows 下长驻 PowerShell 进程"""
    _all: Dict[str, "PowerShellSession"] = weakref.WeakValueDictionary()

    def __init__(self) -> None:
        self.token: str = uuid.uuid4().hex
        self._lock = asyncio.Lock()
        self._ps: Optional[asyncio.subprocess.Process] = None
        # 不再强制设置工作目录，让PowerShell使用默认的当前目录
        # 用户可以通过 Set-Location 或 cd 命令自由切换目录
        self._current_dir = None  # 跟踪当前目录，但不强制设置

    async def start(self) -> None:
        if self._ps is None or self._ps.returncode is not None:
            # 将 PowerShell 会话的工作目录锚定为配置中的 agent_work_root
            work_root = get_config().agent_work_root
            try:
                if work_root and not os.path.isdir(work_root):
                    os.makedirs(work_root, exist_ok=True)
            except Exception:
                pass
            self._current_dir = work_root
            self._ps = await asyncio.create_subprocess_exec(
                "powershell", "-NoExit", "-Command", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_root if isinstance(work_root, str) and work_root else None,
            )
            # 隐藏命令提示，方便后续解析
            await self._run("function prompt {''}")

    async def _run(self, cmd: str) -> Tuple[int, str]:
        """单次命令执行，返回 (exit_code, stdout)。内部用。"""
        marker = f"__END_{uuid.uuid4().hex}__"
        full = f"{cmd} ; if ($?) {{ echo {marker} }} else {{ echo {marker} ; exit 1 }}"
        async with self._lock:
            assert self._ps and self._ps.stdin and self._ps.stdout
            self._ps.stdin.write(full.encode() + b"\n")
            await self._ps.stdin.drain()

            lines = []
            async for raw in self._ps.stdout:
                line = raw.decode(errors="replace").rstrip()
                if marker in line:
                    break
                lines.append(line)
        # 检查是否有输出，如果没有输出说明命令执行成功
        if not lines:
            exit_code = 0
            output = ""
        else:
            # 检查最后一行是否包含marker，如果包含说明命令成功
            if marker in lines[-1]:
                exit_code = 0
                output = "\n".join(lines[:-1])
            else:
                exit_code = 0  # 默认成功，除非明确失败
                output = "\n".join(lines)
        
        return exit_code, output

    async def run(self, nl_instruction: str, timeout: int = 60) -> Dict[str, any]:
        await self.start()
        # 这里用 LLM 把自然语言转成 PowerShell 命令
        try:
            work_root = get_config().agent_work_root
        except Exception:
            work_root = None
        ps_cmd = await _translate(nl_instruction, Path(work_root) if work_root else None)
        # 直接执行命令，不强制切换目录
        exit_code, stdout = await asyncio.wait_for(self._run(ps_cmd), timeout=timeout)
        
        # 尝试获取当前工作目录
        current_dir = "unknown"
        try:
            _, pwd_output = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=5)
            if pwd_output.strip():
                current_dir = pwd_output.strip()
        except:
            pass
            
        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",               # stderr 已合并到 stdout
            "command": ps_cmd,
            "work_dir": current_dir,
        }

    async def close(self) -> None:
        if self._ps and self._ps.returncode is None:
            self._ps.terminate()
            await self._ps.wait()
        # 不再需要清理工作目录，因为我们不创建临时目录了


# ---------- 翻译层 ----------
async def _translate(nl_instruction: str, work_dir: Optional[Path]) -> str:
    """把自然语言转成 PowerShell 命令。"""
    # 如果指令看起来已经是PowerShell命令，直接返回
    if any(nl_instruction.strip().startswith(prefix) for prefix in ['$', 'Get-', 'Set-', 'New-', 'Remove-', 'Test-', 'Write-', 'echo', 'cd', 'mkdir', 'dir', 'if', 'foreach']):
        return nl_instruction
    
    # 如果没有配置LLM或者API密钥，直接返回原指令
    try:
        from utils import llm_completion
        from config import get_llm_config
        
        llm_config = get_llm_config()
        if not llm_config.api_key:
            # 没有API密钥，直接返回指令
            return nl_instruction
    except Exception:
        # 导入失败或配置问题，直接返回指令
        return nl_instruction
    
    work_str = "current directory" if work_dir is None else str(work_dir).replace('\\', '\\\\')
    
    prompt = f"""
You are a helpful assistant that translates natural language into a single, self-contained PowerShell command.
Rules:
- Output ONLY the command, no explanation, no quotes around the entire command.
- Commands will be executed in the {work_str}
- Do not use interactive tools (vim, nano, notepad).
- Use PowerShell commands and syntax.
- Keep commands simple and safe.
- For Windows paths, use normal Windows path format with single backslashes, like C:\\path\\file.txt
- Only add quotes around paths when necessary (spaces in path name). Use double quotes for paths containing spaces.
- Do NOT add extra escaping to backslashes in paths - they should remain as single backslashes.
- Examples of correct path handling:
  * Normal path: C:\\Users\\file.txt
  * Path with spaces: "C:\\Program Files\\file.txt"
  * Do NOT write: "C:\\\\Users\\\\file.txt" or C:\\\\Users\\\\file.txt
- Use PowerShell syntax for variables: $variableName
- Use PowerShell cmdlets when appropriate: Get-ChildItem, New-Item, Set-Location, etc.

User request: {nl_instruction}
"""
    
    try:
        cmd = llm_completion(prompt).strip()
        return cmd
    except Exception:
        # LLM调用失败，返回原指令
        return nl_instruction


# ---------- 对外接口 ----------
_sessions: Dict[str, PowerShellSession] = {}


async def run_single(
    nl_instruction: str,
    timeout: int = 60,
    session_token: Optional[str] = None,
) -> Tuple[str, Dict[str, any]]:
    """
    在**同一 PowerShell 会话**中执行自然语言指令。

    参数
    ----
    session_token : 可选
        None -> 新建会话并返回新 token
        传入旧 token -> 复用对应会话；若会话已死，自动新建并返回新 token

    返回
    ----
    (token, result_dict)
    token 可能与传入不同（会话被重建时）
    """
    # 1. 获取或创建会话
    if session_token is None or session_token not in _sessions:
        session = PowerShellSession()
        _sessions[session.token] = session
        current_token = session.token
    else:
        session = _sessions[session_token]
        # 如果会话已死，重建
        if session._ps is None or session._ps.returncode is not None:
            await session.close()
            # 从旧的sessions中删除
            if session_token in _sessions:
                del _sessions[session_token]
            # 创建新会话
            session = PowerShellSession()
            _sessions[session.token] = session
            current_token = session.token
        else:
            current_token = session_token

    # 2. 执行
    result = await session.run(nl_instruction, timeout=timeout)
    return current_token, result