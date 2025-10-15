
import openai
from config import get_llm_config, get_config
import os
import time
from pathlib import Path
from typing import Dict, Any

def llm_completion(prompt: str, **kwargs) -> str:
    """
    统一的LLM完成函数

    Args:
        prompt: 用户提示
        **kwargs: 覆盖默认配置的参数，如temperature, max_tokens等
            - max_retries: 最大重试次数，默认为3次（当遇到429错误时）

    Returns:
        str: LLM的响应内容

    Note:
        当遇到请求频率限制 (429错误) 时，会自动等待1秒后重试，
        最多重试max_retries次。

    Example:
        # 使用默认配置
        response = llm_completion("你好")

        # 覆盖某些参数
        response = llm_completion("写一首诗", temperature=0.8, max_tokens=500)

        # 自定义重试次数
        response = llm_completion("写一首诗", max_retries=5)
    """
    llm_config = get_llm_config()

    # 设置OpenAI客户端
    client = openai.OpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
    )

    # 合并默认配置和覆盖参数
    request_params = {
        "model": llm_config.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": kwargs.get('temperature', llm_config.temperature),
        "max_tokens": kwargs.get('max_tokens', llm_config.max_tokens),
    }

    # 添加其他可选参数
    for key in ['temperature', 'max_tokens', 'timeout']:
        if key in kwargs and key not in request_params:
            request_params[key] = kwargs[key]

    # 重试逻辑
    max_retries = kwargs.get('max_retries', 8)  # 默认最大重试8次
    retry_count = 0

    # LLM 调用前记录到独立日志文件（不在调试栈里，以避免过量日志），仅截断超长内容
    def _write_llm_log(kind: str, payload: Dict[str, Any]) -> None:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(".agent_llm.log", "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {kind} ")
                # 简单脱敏：不写入 api_key；messages 中只写入 user 内容
                safe = dict(payload)
                try:
                    if "messages" in safe:
                        msgs = safe.get("messages") or []
                        safe["messages"] = [{"role": m.get("role"), "content": (m.get("content") or "")[:4000]} for m in msgs]
                    if "api_key" in safe:
                        safe["api_key"] = "***"
                except Exception:
                    pass
                import json as _json
                f.write(_json.dumps(safe, ensure_ascii=False, default=str))
                f.write("\n")
        except Exception:
            pass

    _write_llm_log("REQUEST", {"model": request_params.get("model"), "messages": request_params.get("messages"), "temperature": request_params.get("temperature"), "max_tokens": request_params.get("max_tokens")})

    while retry_count <= max_retries:
        try:
            resp = client.chat.completions.create(**request_params)
            text = resp.choices[0].message.content
            _write_llm_log("RESPONSE", {"model": request_params.get("model"), "content": (text or "")[:8000]})
            return text
        except openai.RateLimitError as e:
            retry_count += 1
            if retry_count <= max_retries:
                print(f"遇到请求频率限制 (429)，等待5秒后重试 ({retry_count}/{max_retries})")
                time.sleep(5)
            else:
                print(f"已达到最大重试次数 ({max_retries})，请求失败")
                _write_llm_log("ERROR", {"type": "RateLimitError", "message": str(e)[:1000]})
                raise e
        except Exception as e:
            # 对于其他错误，直接抛出
            _write_llm_log("ERROR", {"type": type(e).__name__, "message": str(e)[:1000]})
            raise e


def _expand_repo_placeholders(path_str: str, repo_root: str) -> str:
    """Expand placeholders like repo_root/... or $env:REPO_ROOT/... into absolute paths.

    If path_str is absolute, return as-is. If empty, return repo_root.
    """
    s = (path_str or "").strip()
    rr = (repo_root or "").strip()
    if not rr:
        return s
    if not s:
        return rr
    lowered = s.lower()
    if lowered.startswith("repo_root/") or lowered.startswith("repo_root\\"):
        tail = s.split("/", 1)[1] if "/" in s else s.split("\\", 1)[1]
        return str(Path(rr) / tail)
    for prefix in ("$env:REPO_ROOT\\", "$env:REPO_ROOT/", "%REPO_ROOT%\\", "%REPO_ROOT%/"):
        if s.startswith(prefix):
            return str(Path(rr) / s[len(prefix):])
    if s in ("repo_root", "$env:REPO_ROOT", "%REPO_ROOT%"):
        return rr
    try:
        p = Path(s)
        if not p.is_absolute():
            return str(Path(rr) / p)
    except Exception:
        return s
    return s


def normalize_facts(facts: Dict[str, Any], work_root: str | None = None) -> Dict[str, Any]:
    """Return a normalized copy of facts with absolute repo_root/project_root/exec_root.

    - repo_root: absolute; prefer env REPO_ROOT, then provided work_root, then config.agent_work_root
    - project_root/exec_root: absolute; expand placeholders (repo_root/..., $env:REPO_ROOT/...)
    """
    result = dict(facts or {})
    try:
        cfg = get_config()
        default_root = work_root or os.environ.get("REPO_ROOT") or cfg.agent_work_root or os.getcwd()
    except Exception:
        default_root = work_root or os.environ.get("REPO_ROOT") or os.getcwd()

    # repo_root
    repo_root = str(result.get("repo_root") or result.get("repo_path") or default_root)
    try:
        repo_root_abs = str(Path(repo_root).resolve()) if Path(repo_root).is_absolute() else str(Path(default_root).joinpath(repo_root).resolve())
    except Exception:
        repo_root_abs = str(Path(default_root).resolve())
    result["repo_root"] = repo_root_abs

    # project_root
    project_root = str(result.get("project_root") or "").strip()
    if project_root:
        project_root_abs = _expand_repo_placeholders(project_root, repo_root_abs)
    else:
        # derive from project_name if available
        name = str(result.get("project_name") or "").strip()
        project_root_abs = str(Path(repo_root_abs) / name) if name else repo_root_abs
    try:
        project_root_abs = str(Path(project_root_abs).resolve())
    except Exception:
        project_root_abs = project_root_abs
    result["project_root"] = project_root_abs

    # exec_root defaults to repo_root
    exec_root = str(result.get("exec_root") or repo_root_abs)
    # Expand placeholders like "repo_root/..." or "$env:REPO_ROOT/..." and handle literal "repo_root"
    exec_root_expanded = _expand_repo_placeholders(exec_root, repo_root_abs)
    try:
        exec_root_abs = str(Path(exec_root_expanded).resolve())
    except Exception:
        exec_root_abs = exec_root_expanded
    result["exec_root"] = exec_root_abs

    # cleanup legacy keys
    for legacy in ("repo_path", "work_dir"):
        result.pop(legacy, None)

    return result