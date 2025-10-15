"""
完整的 Shell 执行工具
===================

这是一个兜底化的 shell 指令执行工具，支持：
- 持久化 PowerShell/Bash 会话管理
- 自然语言到 Shell 命令的智能翻译
- 命令规范化和路径处理
- 超时控制和错误处理
- 跨平台支持（Windows PowerShell / Linux Bash）

使用示例：
    from tools.shell import RUN_INSTRUCTION_TOOL
    
    result = RUN_INSTRUCTION_TOOL(
        nl_instruction="列出当前目录的所有文件",
        timeout=60,
        session_token=None  # None=新会话，或传入已有token
    )
"""
from __future__ import annotations

import asyncio
import json
import uuid
import weakref
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from langchain_core.tools import tool

from config import get_config
from agent.debug import dispInfo, debug
from agent.async_utils import run_coro_sync
from tools.base import tool_response


# ============================================================================
# PowerShell 持久会话管理
# ============================================================================

class PowerShellSession:
    """Windows 下长驻 PowerShell 进程，支持持久化会话和环境变量管理"""
    
    _all: Dict[str, "PowerShellSession"] = weakref.WeakValueDictionary()

    def __init__(self) -> None:
        self.token: str = uuid.uuid4().hex
        self._lock = asyncio.Lock()
        self._ps: Optional[asyncio.subprocess.Process] = None
        self._current_dir = None  # 跟踪当前目录

    async def start(self) -> None:
        """启动持久化 PowerShell 进程"""
        if self._ps is None or self._ps.returncode is not None:
            # 将 PowerShell 会话的工作目录锚定为配置中的 agent_work_root
            work_root = get_config().agent_work_root
            try:
                if work_root and not os.path.isdir(work_root):
                    os.makedirs(work_root, exist_ok=True)
            except Exception:
                pass
            
            # 构造子进程环境，确保 REPO_ROOT/PROJECT_ROOT 在 PowerShell 进程启动时已存在
            parent_env = os.environ.copy()
            repo_root_env = parent_env.get("REPO_ROOT") or (work_root if work_root else os.getcwd())
            if not repo_root_env:
                repo_root_env = work_root or os.getcwd()
            project_root_env = parent_env.get("PROJECT_ROOT") or repo_root_env

            # 规范化 PROJECT_ROOT
            project_root_env = self._normalize_project_root(project_root_env, repo_root_env)
            
            child_env = parent_env.copy()
            child_env["REPO_ROOT"] = repo_root_env
            child_env["PROJECT_ROOT"] = project_root_env

            self._current_dir = work_root or repo_root_env
            self._ps = await asyncio.create_subprocess_exec(
                "powershell", "-NoExit", "-Command", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_root if isinstance(work_root, str) and work_root else None,
                env=child_env,
            )
            
            # 隐藏命令提示，方便后续解析
            await self._run("function prompt {''}")
            
            # 初始化环境变量并设置起始位置（持久生效）
            try:
                repo_root_env = child_env.get("REPO_ROOT") or (work_root or "")
                project_root_env = child_env.get("PROJECT_ROOT") or ""
                repo_root_escaped = (repo_root_env or "").replace("'", "''")
                project_root_escaped = (project_root_env or "").replace("'", "''")
                init_cmds = []
                if repo_root_escaped:
                    init_cmds.append(f"$env:REPO_ROOT = '{repo_root_escaped}'")
                if project_root_escaped:
                    init_cmds.append(f"$env:PROJECT_ROOT = '{project_root_escaped}'")
                if repo_root_escaped:
                    init_cmds.append("Set-Location -LiteralPath $env:REPO_ROOT")
                if init_cmds:
                    await self._run("; ".join(init_cmds))
            except Exception:
                pass

    async def _run(self, cmd: str, timeout: Optional[float] = None) -> Tuple[int, str]:
        """单次命令执行，返回 (exit_code, stdout)
        
        改进点：
        - 合并标准错误到标准输出（2>&1）
        - 使用带状态的标记行，不退出会话进程
        - 支持整体超时控制
        - 每行读取使用动态超时
        """
        import time
        marker = f"__END_{uuid.uuid4().hex}__"
        
        # 将 stderr 合并到 stdout；通过 marker:0/marker:1 标识成功或失败
        full = (
            "$ErrorActionPreference='Continue'; "
            f"{cmd} 2>&1 | Out-String -Stream; "
            "$code = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { if ($?) { 0 } else { 1 } }; "
            f"Write-Output {marker}:$code"
        )
        
        start_time = time.time()
        
        async with self._lock:
            assert self._ps and self._ps.stdin and self._ps.stdout
            self._ps.stdin.write(full.encode() + b"\n")
            await self._ps.stdin.drain()

            output_lines = []
            status_line = None
            
            # 默认单行超时：600秒（网络慢/大仓库克隆时可能长时间无输出）
            base_line_timeout = 600.0
            
            while True:
                try:
                    # 计算剩余时间
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise asyncio.TimeoutError("Overall timeout reached")
                        line_timeout = min(base_line_timeout, remaining + 1)
                    else:
                        line_timeout = base_line_timeout
                    
                    raw = await asyncio.wait_for(self._ps.stdout.readline(), timeout=line_timeout)
                    if not raw:  # EOF
                        break
                    line = raw.decode(errors="replace").rstrip()
                    if line.startswith(marker):
                        status_line = line
                        break
                    output_lines.append(line)
                except asyncio.TimeoutError:
                    output_lines.append(f"[单行超时: {line_timeout}秒内无新输出]")
                    raise

        exit_code = 0
        if status_line is not None and status_line.endswith(":1"):
            exit_code = 1
        output = "\n".join(output_lines)
        return exit_code, output

    async def run(self, nl_instruction: str, timeout: int = 60) -> Dict[str, Any]:
        """执行自然语言指令，返回结构化结果"""
        await self.start()
        
        # 获取当前工作目录
        try:
            work_root = get_config().agent_work_root
        except Exception:
            work_root = None
        
        cur_dir = None
        try:
            _, _pwd = await asyncio.wait_for(
                self._run("Get-Location | Select-Object -ExpandProperty Path"), 
                timeout=10
            )
            if _pwd.strip():
                cur_dir = Path(_pwd.strip())
        except Exception:
            cur_dir = Path(work_root) if work_root else None
        
        # 翻译自然语言到 PowerShell 命令
        ps_cmd = await _translate_nl_to_ps(nl_instruction, cur_dir)
        try:
            debug.note("translated_command", ps_cmd)
        except Exception:
            pass
        
        # 获取环境变量
        repo_root_env = os.environ.get("REPO_ROOT") or (work_root if work_root else os.getcwd())
        project_root_env = os.environ.get("PROJECT_ROOT") or repo_root_env
        project_root_env = self._normalize_project_root(project_root_env, repo_root_env)

        # 记录开始目录
        start_dir = ""
        try:
            _, start_dir = await asyncio.wait_for(
                self._run("Get-Location | Select-Object -ExpandProperty Path"), 
                timeout=10
            )
            start_dir = (start_dir or "").strip()
        except Exception:
            start_dir = ""

        # 同步环境变量
        try:
            sync_cmds = []
            _repo = (repo_root_env or "").replace("'", "''")
            _proj = (project_root_env or repo_root_env or "").replace("'", "''")
            if _repo:
                sync_cmds.append(f"$env:REPO_ROOT = '{_repo}'")
            if _proj:
                sync_cmds.append(f"$env:PROJECT_ROOT = '{_proj}'")
            if sync_cmds:
                await asyncio.wait_for(self._run("; ".join(sync_cmds)), timeout=10)
        except Exception as e:
            try:
                debug.note("env_sync_error", str(e))
            except Exception:
                pass

        # 规范化命令（路径处理等）
        ps_cmd = _sanitize_ps_cmd(ps_cmd, repo_root_env, project_root_env)

        # 记录环境变量（调试用）
        try:
            _, env_before = await asyncio.wait_for(
                self._run("Write-Output REPO=$env:REPO_ROOT; Write-Output PROJ=$env:PROJECT_ROOT"), 
                timeout=10
            )
            try:
                debug.note("session_env_before", env_before)
            except Exception:
                pass
        except Exception:
            pass

        # 执行命令
        try:
            exit_code, stdout = await asyncio.wait_for(
                self._run(ps_cmd, timeout=timeout), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # 超时：关闭当前会话
            try:
                await self.close()
            except Exception:
                pass
            return {
                "exit_code": 124,
                "stdout": f"Timed out after {timeout}s",
                "stderr": "",
                "command": ps_cmd,
                "work_dir": repo_root_env or start_dir,
                "start_dir": start_dir,
                "end_dir": start_dir,
                "timed_out": True,
            }
        
        # 记录环境变量（调试用）
        try:
            _, env_after = await asyncio.wait_for(
                self._run("Write-Output REPO=$env:REPO_ROOT; Write-Output PROJ=$env:PROJECT_ROOT"), 
                timeout=10
            )
            try:
                debug.note("session_env_after", env_after)
            except Exception:
                pass
        except Exception:
            pass

        # 获取当前工作目录
        current_dir = "unknown"
        try:
            _, pwd_output = await asyncio.wait_for(
                self._run("Get-Location | Select-Object -ExpandProperty Path"), 
                timeout=10
            )
            if pwd_output.strip():
                current_dir = pwd_output.strip()
            else:
                # 无输出则重试
                _, pwd_output2 = await asyncio.wait_for(
                    self._run("Get-Location | Select-Object -ExpandProperty Path"), 
                    timeout=10
                )
                if pwd_output2.strip():
                    current_dir = pwd_output2.strip()
        except:
            try:
                _, pwd_output2 = await asyncio.wait_for(
                    self._run("Get-Location | Select-Object -ExpandProperty Path"), 
                    timeout=10
                )
                if pwd_output2.strip():
                    current_dir = pwd_output2.strip()
            except:
                pass
            
        # 记录结束目录
        end_dir = current_dir
        try:
            _, end_dir_out = await asyncio.wait_for(
                self._run("Get-Location | Select-Object -ExpandProperty Path"), 
                timeout=10
            )
            if end_dir_out.strip():
                end_dir = end_dir_out.strip()
        except Exception:
            pass

        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",  # stderr 已合并到 stdout
            "command": ps_cmd,
            "work_dir": repo_root_env or current_dir,
            "start_dir": start_dir,
            "end_dir": end_dir,
        }

    async def close(self) -> None:
        """关闭 PowerShell 会话"""
        if self._ps and self._ps.returncode is None:
            self._ps.terminate()
            await self._ps.wait()

    @staticmethod
    def _normalize_project_root(pr: str | None, rr: str | None) -> str:
        """规范化 PROJECT_ROOT 路径"""
        pr = (pr or "").strip()
        rr = (rr or "").strip()
        if not rr:
            return pr
        
        # 兼容 repo_root/xxx 或 repo_root\\xxx
        lowered = pr.lower()
        if lowered.startswith("repo_root/") or lowered.startswith("repo_root\\"):
            tail = pr.split("/", 1)[1] if "/" in pr else pr.split("\\", 1)[1]
            return str(Path(rr) / tail)
        
        # 兼容 $env:REPO_ROOT 前缀
        for prefix in ("$env:REPO_ROOT\\", "$env:REPO_ROOT/", "%REPO_ROOT%\\", "%REPO_ROOT%/"):
            if pr.startswith(prefix):
                return str(Path(rr) / pr[len(prefix):])
        
        # 直接等于 repo_root 占位
        if pr in ("repo_root", "$env:REPO_ROOT", "%REPO_ROOT%", ""):
            return rr
        
        # 相对路径 -> 拼到 rr
        try:
            p = Path(pr)
            if not p.is_absolute():
                return str(Path(rr) / p)
        except Exception:
            pass
        
        return pr


# ============================================================================
# 自然语言翻译
# ============================================================================

async def _translate_nl_to_ps(nl_instruction: str, work_dir: Optional[Path]) -> str:
    """将自然语言转换为 PowerShell 命令
    
    如果指令已经是命令格式，直接返回
    如果没有 LLM 配置，直接返回原指令
    """
    # 如果指令看起来已经是PowerShell命令，直接返回
    ps_prefixes = ['$', 'Get-', 'Set-', 'New-', 'Remove-', 'Test-', 'Write-', 
                   'echo', 'cd', 'mkdir', 'dir', 'if', 'foreach']
    if any(nl_instruction.strip().startswith(prefix) for prefix in ps_prefixes):
        return nl_instruction
    
    # 如果没有配置LLM或者API密钥，直接返回原指令
    try:
        from utils import llm_completion
        from config import get_llm_config
        
        llm_config = get_llm_config()
        if not llm_config.api_key:
            return nl_instruction
    except Exception:
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
- Never emit absolute disk-rooted paths (like C:\\ or D:\\). Build all paths from $env:REPO_ROOT, or $env:PROJECT_ROOT if set.
- Prefer Join-Path with -LiteralPath when specifying file/dir arguments. Avoid Set-Location/cd; pass explicit -LiteralPath instead.
- For project-scoped operations (install/edit/read within the target project), refer to $env:PROJECT_ROOT if available; otherwise Join-Path $env:REPO_ROOT '<project_dir>'.
- For editable install, prefer: pip install -e $env:PROJECT_ROOT (or pip install -e (Join-Path $env:REPO_ROOT '<project_dir>')).
- Examples of correct path handling:
  * Get-Content -LiteralPath (Join-Path $env:REPO_ROOT 'transformers\\README.md') -Raw
  * pip install -r (Join-Path $env:REPO_ROOT 'transformers\\requirements.txt')
  * pip install -e $env:PROJECT_ROOT
- Use PowerShell syntax for variables: $variableName
- Use PowerShell cmdlets when appropriate: Get-ChildItem, New-Item, etc.

User request: {nl_instruction}
"""
    
    try:
        cmd = llm_completion(prompt).strip()
        return cmd
    except Exception:
        return nl_instruction


# ============================================================================
# 命令规范化
# ============================================================================

def _sanitize_ps_cmd(ps_cmd: str, repo_root: str, project_root: str) -> str:
    """规范化 PowerShell 命令
    
    - 移除/替换 Set-Location/cd 语句
    - 将仓库内绝对路径替换为基于 $env:REPO_ROOT 的 Join-Path
    - 将相对项目语义的 pip install -e . 改写为 pip install -e $env:PROJECT_ROOT
    - 将常见文件操作添加 -LiteralPath 并用 Join-Path 构造
    """
    text = (ps_cmd or "").strip()
    if not text:
        return ps_cmd

    # pip install -e . -> 指向项目根
    if project_root:
        text = re.sub(
            r"pip\s+install\s+-e\s+\.", 
            "pip install -e $env:PROJECT_ROOT", 
            text, 
            flags=re.IGNORECASE
        )

    # 将以 repo_root 开头的绝对路径替换为 Join-Path
    def replace_repo_abs(m: "re.Match[str]") -> str:
        abs_path = m.group(0)
        if not repo_root:
            return abs_path
        try:
            rel = abs_path[len(repo_root):].lstrip("\\/")
            rel = rel.replace("'", "''")
            return f"(Join-Path $env:REPO_ROOT '{rel}')"
        except Exception:
            return abs_path

    if repo_root:
        safe_root = re.escape(repo_root.rstrip("\\/"))
        text = re.sub(safe_root + r"[\\/][^\s'\"]+", replace_repo_abs, text)

    # 常见 cmdlet 加 -LiteralPath（仅在路径不以 - 开头时）
    def ensure_literal(cmd: str, pat: str) -> None:
        nonlocal text
        
        def _repl(m: "re.Match[str]") -> str:
            path_arg = m.group(2)
            tail = m.group(3) or ""
            already_literal = "-LiteralPath" in m.group(0)
            already_path = "-Path" in m.group(0)
            # 如果路径参数以 - 开头，说明是其他参数（如 -Name），不添加 -LiteralPath
            is_parameter = isinstance(path_arg, str) and path_arg.startswith('-')
            if already_literal or already_path or is_parameter:
                return m.group(0)
            return f"{m.group(1)} -LiteralPath {path_arg}{tail}"

        text = re.sub(pat, _repl, text, flags=re.IGNORECASE)

    # 只为 Get-Content 添加 -LiteralPath（跳过以 - 开头的参数）
    ensure_literal(
        r"Get-Content", 
        r"\b(Get-Content)\s+(?!-LiteralPath)(\([^\)]+\)|[^\s]+)(\s+-Raw)?"
    )
    # Get-ChildItem 语法太灵活，不自动添加 -LiteralPath，避免破坏命令
    # ensure_literal(
    #     r"Get-ChildItem", 
    #     r"\b(Get-ChildItem)\s+(?!-LiteralPath)(\([^\)]+\)|[^\s]+)(\b.*)?"
    # )

    # 去重 -LiteralPath
    text = re.sub(
        r"\s+-LiteralPath\s+-LiteralPath\s+", 
        " -LiteralPath ", 
        text, 
        flags=re.IGNORECASE
    )
    # 修正 "-LiteralPath -Path" 组合（仅针对 Get-Content）
    text = re.sub(
        r"\b(Get-Content)\b([^\n]*?)\s+-LiteralPath\s+-Path\s+", 
        r"\1\2 -LiteralPath ", 
        text, 
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(Get-Content)\b([^\n]*?)\s+-Path\s+-LiteralPath\s+", 
        r"\1\2 -LiteralPath ", 
        text, 
        flags=re.IGNORECASE
    )

    # 规范 git clone 目标
    def _rewrite_git_clone(m: "re.Match[str]") -> str:
        url = m.group(1)
        repo_name = url.strip().rstrip('/')
        if repo_name.lower().endswith('.git'):
            repo_name = repo_name[:-4]
        repo_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        repo_name = repo_name.replace("'", "''")
        return f"git clone {url} (Join-Path $env:REPO_ROOT '{repo_name}')"

    text = re.sub(
        r"\bgit\s+clone\s+(\S+)\s+\$env:REPO_ROOT\b", 
        _rewrite_git_clone, 
        text, 
        flags=re.IGNORECASE
    )

    return text


# ============================================================================
# 会话管理
# ============================================================================

_sessions: Dict[str, PowerShellSession] = {}


@dispInfo("run_single")
async def run_single(
    nl_instruction: str,
    timeout: int = 60,
    session_token: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """在持久 PowerShell 会话中执行自然语言指令
    
    参数:
        nl_instruction: 自然语言指令或 PowerShell 命令
        timeout: 超时时间（秒）
        session_token: 会话 token（None=新建会话）
    
    返回:
        (token, result_dict)
        - token: 会话标识符（可能与传入不同）
        - result_dict: 执行结果字典
    """
    # 获取或创建会话
    if session_token is None or session_token not in _sessions:
        session = PowerShellSession()
        _sessions[session.token] = session
        current_token = session.token
    else:
        session = _sessions[session_token]
        # 如果会话已死，重建
        if session._ps is None or session._ps.returncode is not None:
            await session.close()
            if session_token in _sessions:
                del _sessions[session_token]
            session = PowerShellSession()
            _sessions[session.token] = session
            current_token = session.token
        else:
            current_token = session_token

    # 执行
    result = await session.run(nl_instruction, timeout=timeout)
    try:
        debug.note("executed_command", result.get("command"))
        debug.note("stdout", (result.get("stdout") or "")[:800])
    except Exception:
        pass
    
    return current_token, result


# ============================================================================
# LangChain 工具接口
# ============================================================================

@tool("run_instruction")
def RUN_INSTRUCTION_TOOL(
    nl_instruction: str, 
    timeout: int = 60, 
    session_token: str | None = None
) -> str:
    """在持久 Shell 会话中执行自然语言指令或 Shell 命令。
    
    - 自动将自然语言转换为 PowerShell/Bash 命令
    - 支持持久化会话（环境变量、工作目录等会保持）
    - 自动处理路径规范化和命令优化
    - 支持超时控制和错误处理
    """
    try:
        debug.note("nl_instruction", nl_instruction)
        debug.note("timeout", timeout)
        debug.note("session_token_in", session_token or "<none>")
    except Exception:
        pass

    try:
        token, result = run_coro_sync(
            run_single(nl_instruction, timeout=timeout, session_token=session_token),
            timeout=timeout + 5,
        )
        # result 包含: exit_code, stdout, stderr, command, work_dir, start_dir, end_dir
        try:
            debug.note("session_token_out", token)
            debug.note("run_single_result", result)
        except Exception:
            pass
        
        # 判断是否成功：退出码为0
        is_ok = result.get("exit_code", 1) == 0
        
        return tool_response(
            tool="run_instruction",
            ok=is_ok,
            data={
                **result,
                "session_token": token
            }
        )
    except Exception as e:
        # 工具内部异常
        try:
            debug.note("run_single_error", str(e))
        except Exception:
            pass
        return tool_response(
            tool="run_instruction",
            ok=False,
            data={
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "command": nl_instruction,
                "work_dir": "",
                "start_dir": "",
                "end_dir": "",
                "session_token": session_token or ""
            },
            error=f"{type(e).__name__}: {e}"
        )
