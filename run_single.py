import subprocess
import tempfile
import uuid
import asyncio
import shutil
import os
from pathlib import Path
from utils import llm_completion
from config import get_config
import locale
import base64

def decode_output(data: bytes) -> str:
    """
    智能解码命令输出，支持多种编码格式

    Args:
        data: 字节数据

    Returns:
        str: 解码后的字符串
    """
    if not data:
        return ""

    # 尝试多种编码方式
    encodings = [
        locale.getpreferredencoding(False),  # 系统默认编码
        'utf-8',
        'gbk',
        'cp936',
        'utf-16',
        'latin1'
    ]

    for encoding in encodings:
        try:
            return data.decode(encoding).strip()
        except (UnicodeDecodeError, LookupError):
            continue

    # 如果所有编码都失败，使用错误替换
    return data.decode('utf-8', errors='replace').strip()

def run_single_sync(nl_instruction: str, timeout: int = 30) -> dict:
    """
    同步版本的run_single函数，方便测试使用

    Args:
        nl_instruction: 自然语言指令
        timeout: 命令执行超时时间（秒）

    Returns:
        dict: 包含执行结果的字典 {"exit_code": int, "stdout": str, "stderr": str, "command": str, "work_dir": str}
    """
    async def _async_run():
        return await run_single(nl_instruction, timeout)

    return asyncio.run(_async_run())

async def run_single(nl_instruction: str, timeout: int = 30) -> dict:
    """
    自然语言 → 单条 shell 命令 → 执行 → 原始结果
    返回 {"exit_code": int, "stdout": str, "stderr": str, "command": str, "work_dir": str}
    todo 这里的instruction是，在agent work flow中，应该是由llm生成的。这里没有做格式化或者结构化。

    Args:
        nl_instruction: 自然语言指令
        timeout: 命令执行超时时间（秒）

    Returns:
        dict: 包含执行结果的字典
    """
    # # 1. 生成隔离工作区
    # config = get_config()
    # uid = uuid.uuid4().hex[:8]
    # work_root = Path(config.agent_work_root)
    # work_root.mkdir(exist_ok=True)  # 确保工作目录根路径存在
    # work = work_root / f"agent_run_{uid}"
    # work.mkdir(exist_ok=True)
    # 1. 连接到工作区
    config = get_config()
    work = Path(config.agent_work_root)
    work.mkdir(exist_ok=True)

    # 2. LLM 把自然语言翻成**单条** shell 命令
    #    这里用同步调用即可，工具函数里可阻塞
    # 将路径转换为字符串，确保在 prompt 中正确显示
    work_str = str(work).replace('\\', '\\\\')  # 转义反斜杠用于在 prompt 中显示

    prompt = f"""
You are a helpful assistant that translates natural language into a single, self-contained shell command.
Rules:
- Output ONLY the command, no explanation, no quotes.
- Assume cwd is {work_str}
- Do not use interactive tools (vim, nano, top).
- Prefer simple builtins and universally installed programs.
- Keep commands simple and safe.
- CRITICAL: For Windows paths, use normal Windows path format with single backslashes, like C:\\path\\file.txt
- CRITICAL: Only add quotes around paths when necessary (spaces in path name). Use double quotes for paths containing spaces.
- CRITICAL: Do NOT add extra escaping to backslashes in paths - they should remain as single backslashes.
- Examples of correct path handling:
  * Normal path: C:\\Users\\file.txt
  * Path with spaces: "C:\\Program Files\\file.txt"
  * Do NOT write: "C:\\\\Users\\\\file.txt" or C:\\\\Users\\\\file.txt
- IMPORTANT: This is a Windows system. Use only PowerShell and CMD supported commands.
- AVOID Unix/Linux commands like: ls, grep, sed, awk, find, head, tail, cat, wc, sort, uniq, etc.
- PREFER Windows equivalents: dir instead of ls, findstr instead of grep, type instead of cat.
- Use PowerShell commands rather than cmd commands.

User request: {nl_instruction}
"""
    cmd = llm_completion(prompt).strip()  # 返回如: echo "hello"

    # 3. 安全检查（暂时移除拦截逻辑，放行命令执行）

    # 4. 执行 优先使用PowerShell
    proc = None
    try:
        # 在Windows上优先使用PowerShell执行命令
        if os.name == 'nt':  # Windows系统
            # 使用 EncodedCommand 彻底规避引号与分号的转义问题
            # PowerShell Base64 采用 UTF-16LE 编码
            encoded = base64.b64encode(cmd.encode('utf-16le')).decode('ascii')
            ps_cmd = f'powershell -NoProfile -EncodedCommand {encoded}'
        else:
            ps_cmd = cmd

        proc = await asyncio.create_subprocess_shell(
            ps_cmd,
            cwd=work,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        if proc:
            proc.kill()
            await proc.wait()
        return {
            "exit_code": -2,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "command": cmd,
            "work_dir": str(work)
        }
    except Exception as e:
        return {
            "exit_code": -3,
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "command": cmd,
            "work_dir": str(work)
        }
    # finally:
    #     # 清理临时目录
    #     if work.exists():
    #         shutil.rmtree(work, ignore_errors=True)

    return {
        "exit_code": proc.returncode if proc and proc.returncode is not None else -1,
        "stdout": decode_output(stdout) if stdout else "",
        "stderr": decode_output(stderr) if stderr else "",
        "command": cmd,
        "work_dir": str(work)
    }