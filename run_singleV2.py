import asyncio
import uuid
import weakref
from typing import Dict, Optional, Tuple
from pathlib import Path
import os
from config import get_config
from agent.debug import dispInfo, debug

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
            # 构造子进程环境，确保 REPO_ROOT/PROJECT_ROOT 在 PowerShell 进程启动时已存在
            parent_env = os.environ.copy()
            # 优先使用外部环境的 REPO_ROOT；否则使用 agent 工作目录，其次回退当前进程工作目录
            repo_root_env = parent_env.get("REPO_ROOT") or (work_root if work_root else os.getcwd())
            if not repo_root_env:
                # 兜底：若 CWD 也不可用，则使用 agent 工作目录
                repo_root_env = work_root or os.getcwd()
            # PROJECT_ROOT 未提供时默认等于 REPO_ROOT（先占位，稍后规范化）
            project_root_env = parent_env.get("PROJECT_ROOT") or repo_root_env

            def _normalize_project_root(pr: str | None, rr: str | None) -> str:
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

            project_root_env = _normalize_project_root(project_root_env, repo_root_env)
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
                # 再次在会话内设置，确保一致；优先使用传给子进程的值
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

    async def _run(self, cmd: str) -> Tuple[int, str]:
        """单次命令执行，返回 (exit_code, stdout)。内部用。
        改进点：
        - 合并标准错误到标准输出（2>&1）
        - 使用带状态的标记行，不再退出会话进程
        """
        marker = f"__END_{uuid.uuid4().hex}__"
        # 将 stderr 合并到 stdout；通过 marker:0/marker:1 标识成功或失败
        # 使用 $LASTEXITCODE 捕获外部进程退出码；若为 None 则回退到 $?（True->0/False->1）
        # 通过 Out-Host 保证标记行被写入主输出通道，避免某些情况下被缓冲吞掉
        full = (
            "$ErrorActionPreference='Continue'; "
            f"{cmd} 2>&1; "
            "$code = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { if ($?) { 0 } else { 1 } }; "
            f"Write-Output {marker}:$code | Out-Host"
        )
        async with self._lock:
            assert self._ps and self._ps.stdin and self._ps.stdout
            self._ps.stdin.write(full.encode() + b"\n")
            await self._ps.stdin.drain()

            output_lines = []
            status_line = None
            async for raw in self._ps.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line.startswith(marker):
                    status_line = line
                    break
                output_lines.append(line)

        exit_code = 0
        if status_line is not None and status_line.endswith(":1"):
            exit_code = 1
        output = "\n".join(output_lines)
        return exit_code, output

    async def run(self, nl_instruction: str, timeout: int = 60) -> Dict[str, any]:
        await self.start()
        # 这里用 LLM 把自然语言转成 PowerShell 命令
        try:
            work_root = get_config().agent_work_root
        except Exception:
            work_root = None
        # 传入真实当前目录供翻译器构造相对路径
        cur_dir = None
        try:
            _, _pwd = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=10)
            if _pwd.strip():
                cur_dir = Path(_pwd.strip())
        except Exception:
            cur_dir = Path(work_root) if work_root else None
        ps_cmd = await _translate(nl_instruction, cur_dir)
        try:
            debug.note("translated_command", ps_cmd)
        except Exception:
            pass
        # 会话已在 start() 中初始化 $env:REPO_ROOT/$env:PROJECT_ROOT
        # 以 agent 工作目录作为默认仓库根（若未设置 REPO_ROOT），否则回退当前进程 CWD；PROJECT_ROOT 默认等于 REPO_ROOT
        repo_root_env = os.environ.get("REPO_ROOT") or (work_root if work_root else os.getcwd())
        project_root_env = os.environ.get("PROJECT_ROOT") or repo_root_env

        def _normalize_project_root(pr: str | None, rr: str | None) -> str:
            pr = (pr or "").strip()
            rr = (rr or "").strip()
            if not rr:
                return pr
            lowered = pr.lower()
            if lowered.startswith("repo_root/") or lowered.startswith("repo_root\\"):
                tail = pr.split("/", 1)[1] if "/" in pr else pr.split("\\", 1)[1]
                return str(Path(rr) / tail)
            for prefix in ("$env:REPO_ROOT\\", "$env:REPO_ROOT/", "%REPO_ROOT%\\", "%REPO_ROOT%/"):
                if pr.startswith(prefix):
                    return str(Path(rr) / pr[len(prefix):])
            if pr in ("repo_root", "$env:REPO_ROOT", "%REPO_ROOT%", ""):
                return rr
            try:
                p = Path(pr)
                if not p.is_absolute():
                    return str(Path(rr) / p)
            except Exception:
                pass
            return pr

        project_root_env = _normalize_project_root(project_root_env, repo_root_env)

        # 记录开始目录
        start_dir = ""
        try:
            _, start_dir = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=10)
            start_dir = (start_dir or "").strip()
        except Exception:
            start_dir = ""

        # 每次执行前，同步会话内的环境变量，跟随最新的进程环境/状态
        # 放在规范化之前，确保后续重写使用的值与会话一致
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

        # 在执行前对命令做一次路径与安装目标的规范化重写（使用同步后的值）
        ps_cmd = _sanitize_ps_cmd(ps_cmd, repo_root_env, project_root_env)

        # 在执行命令前后记录会话内环境变量值，便于定位空输出问题
        try:
            _, env_before = await asyncio.wait_for(self._run("Write-Output REPO=$env:REPO_ROOT; Write-Output PROJ=$env:PROJECT_ROOT"), timeout=10)
            try:
                debug.note("session_env_before", env_before)
            except Exception:
                pass
        except Exception:
            pass

        try:
            exit_code, stdout = await asyncio.wait_for(self._run(ps_cmd), timeout=timeout)
        except asyncio.TimeoutError:
            # 超时：关闭当前会话，返回结构化结果，外层可据此重试或降级
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
        
        # 记录执行后会话内环境变量
        try:
            _, env_after = await asyncio.wait_for(self._run("Write-Output REPO=$env:REPO_ROOT; Write-Output PROJ=$env:PROJECT_ROOT"), timeout=10)
            try:
                debug.note("session_env_after", env_after)
            except Exception:
                pass
        except Exception:
            pass

        # 尝试获取当前工作目录
        current_dir = "unknown"
        try:
            _, pwd_output = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=10)
            if pwd_output.strip():
                current_dir = pwd_output.strip()
            else:
                # 无输出则重试一次
                _, pwd_output2 = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=10)
                if pwd_output2.strip():
                    current_dir = pwd_output2.strip()
        except:
            try:
                # 异常时重试一次
                _, pwd_output2 = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=10)
                if pwd_output2.strip():
                    current_dir = pwd_output2.strip()
            except:
                pass
            
        # 记录结束目录
        end_dir = current_dir
        try:
            _, end_dir_out = await asyncio.wait_for(self._run("Get-Location | Select-Object -ExpandProperty Path"), timeout=10)
            if end_dir_out.strip():
                end_dir = end_dir_out.strip()
        except Exception:
            pass

        return {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",               # stderr 已合并到 stdout
            "command": ps_cmd,
            "work_dir": repo_root_env or current_dir,
            "start_dir": start_dir,
            "end_dir": end_dir,
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
        # LLM调用失败，返回原指令
        return nl_instruction


def _sanitize_ps_cmd(ps_cmd: str, repo_root: str, project_root: str) -> str:
    """保底规范化翻译产物：
    - 移除/替换 Set-Location/cd 语句
    - 将仓库内绝对路径替换为基于 $env:REPO_ROOT 的 Join-Path
    - 将相对项目语义的 pip install -e . 改写为 pip install -e $env:PROJECT_ROOT（若存在）
    - 将常见文件操作添加 -LiteralPath 并用 Join-Path 构造
    该函数尽量采用稳健的字符串规则，不依赖复杂解析。
    """
    try:
        import re
    except Exception:
        re = None  # 兜底

    text = (ps_cmd or "").strip()
    if not text:
        return ps_cmd

    # 允许在持久会话中使用 Set-Location/cd，保留用户的目录切换指令

    # 2) pip install -e . -> 指向项目根（若可用）
    if project_root:
        if re:
            text = re.sub(r"pip\s+install\s+-e\s+\.", "pip install -e $env:PROJECT_ROOT", text, flags=re.IGNORECASE)
        else:
            text = text.replace("pip install -e .", "pip install -e $env:PROJECT_ROOT")

    # 3) 将以 repo_root 开头的绝对路径替换为 Join-Path
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

    if repo_root and re:
        safe_root = re.escape(repo_root.rstrip("\\/"))
        text = re.sub(safe_root + r"[\\/][^\s'\"]+", replace_repo_abs, text)

    # 4) 常见 cmdlet 加 -LiteralPath（若看起来像裸路径参数），避免重复
    def ensure_literal(cmd: str, pat: str) -> None:
        nonlocal text
        if re is None:
            return
        text = re.sub(
            pat,
            lambda m: (
                f"{m.group(1)} -LiteralPath {m.group(2)}{m.group(3) or ''}"
            ),
            text,
            flags=re.IGNORECASE,
        )

    # 形如: Get-Content <path> [-Raw]，但不要对已含 -LiteralPath 的再插入（使用负向前瞻）
    ensure_literal(r"Get-Content", r"\b(Get-Content)\s+(?!-LiteralPath)(\([^\)]+\)|\S+)(\s+-Raw)?")
    # 形如: Get-ChildItem <path>，同样避免重复 -LiteralPath
    ensure_literal(r"Get-ChildItem", r"\b(Get-ChildItem)\s+(?!-LiteralPath)(\([^\)]+\)|\S+)(\b.*)?")

    # 一次性去重：将重复出现的 -LiteralPath 标记合并一次
    if re:
        text = re.sub(r"\s+-LiteralPath\s+-LiteralPath\s+", " -LiteralPath ", text, flags=re.IGNORECASE)

    # 不要改写 pip install -r 的参数为 -LiteralPath（pip 不支持）

    # 5) 规范 git clone 目标：若写成 $env:REPO_ROOT，则改写为子目录（由 URL 推断）
    if re:
        def _rewrite_git_clone(m: "re.Match[str]") -> str:
            url = m.group(1)
            # 提取仓库名
            repo_name = url.strip().rstrip('/')
            # 去掉 .git 后缀
            if repo_name.lower().endswith('.git'):
                repo_name = repo_name[:-4]
            # 取最后一段
            repo_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
            repo_name = repo_name.replace("'", "''")
            return f"git clone {url} (Join-Path $env:REPO_ROOT '{repo_name}')"

        text = re.sub(r"\bgit\s+clone\s+(\S+)\s+\$env:REPO_ROOT\b", _rewrite_git_clone, text, flags=re.IGNORECASE)

    return text


# ---------- 对外接口 ----------
_sessions: Dict[str, PowerShellSession] = {}


@dispInfo("run_single")
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
    try:
        # 记录关键输出，便于上层查看“真实命令”和标准输出
        debug.note("executed_command", result.get("command"))
        debug.note("stdout", (result.get("stdout") or "")[:800])
    except Exception:
        pass
    return current_token, result