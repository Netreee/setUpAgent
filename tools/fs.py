from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.tools import tool

from config import get_config
from utils import normalize_facts
from agent.debug import dispInfo, debug
from tools.base import tool_response
import re


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
    """Resolve path safely within workspace root.

    Returns (ok, violation, resolved_path)
    """
    try:
        root = _get_workspace_root()
        try:
            # 记录当前 workspace 根，便于定位根目录切换问题
            debug.note("workspace_root", str(root))
        except Exception:
            pass
        # 尝试将 repo_root 占位符展开
        try:
            cfg = get_config()
            facts = {"repo_root": str(root)}
            expanded = normalize_facts({"repo_root": str(root), "project_root": str(path)}, work_root=str(root)).get("project_root", str(path))
        except Exception:
            expanded = str(path)
        p = Path(expanded)
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


@tool("files_exists")
@dispInfo("fs_exists")
def FILES_EXISTS_TOOL(path: str) -> str:
    """Check whether a path exists within the workspace root."""
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        try:
            debug.note("resolve_failed", {"ok": ok, "violation": violation})
        except Exception:
            pass
        return tool_response(
            tool="files_exists",
            ok=False,
            data={"path": str(path)},
            error=violation or "unknown_error"
        )
    try:
        exists = p.exists()
        try:
            debug.note("resolved_path", str(p))
            debug.note("exists", exists)
        except Exception:
            pass
        return tool_response(
            tool="files_exists",
            ok=True,
            data={"exists": exists, "path": str(p)}
        )
    except Exception as e:
        return tool_response(
            tool="files_exists",
            ok=False,
            data={"path": str(p)},
            error=f"{type(e).__name__}: {e}"
        )


@tool("files_stat")
@dispInfo("fs_stat")
def FILES_STAT_TOOL(path: str) -> str:
    """Stat a file or directory."""
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        return tool_response(
            tool="files_stat",
            ok=False,
            data={"path": str(path), "type": "missing"},
            error=violation or "unknown_error"
        )
    try:
        if not p.exists():
            return tool_response(
                tool="files_stat",
                ok=True,
                data={"path": str(p), "type": "missing"}
            )
        st = p.stat()
        kind = "dir" if p.is_dir() else ("file" if p.is_file() else "other")
        return tool_response(
            tool="files_stat",
            ok=True,
            data={
                "path": str(p),
                "type": kind,
                "size": int(st.st_size),
                "mtime": float(st.st_mtime),
                "is_symlink": p.is_symlink(),
            }
        )
    except Exception as e:
        return tool_response(
            tool="files_stat",
            ok=False,
            data={"path": str(p)},
            error=f"{type(e).__name__}: {e}"
        )


def _iter_list(dir_path: Path, recurse: bool) -> Iterable[Path]:
    if not recurse:
        try:
            yield from (dir_path.iterdir())
        except Exception:
            return
        return
    # Recurse safely without following symlinks out of root
    root = _get_workspace_root()
    for base, dirs, files in os.walk(dir_path):
        base_path = Path(base)
        # Ensure we do not walk outside root via symlinks
        try:
            base_path.resolve().relative_to(root)
        except Exception:
            continue
        for d in list(dirs):
            dp = (base_path / d)
            try:
                dp.resolve().relative_to(root)
            except Exception:
                dirs.remove(d)
        for name in files + dirs:
            yield base_path / name


@tool("files_list")
@dispInfo("fs_list")
def FILES_LIST_TOOL(path: str, files_only: bool = False, recurse: bool = False, patterns: Optional[List[str]] = None, limit: int = 1000) -> str:
    """List entries under a directory with optional recursion and glob-like filtering."""
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        return tool_response(
            tool="files_list",
            ok=False,
            data={"dir": str(path), "entries": []},
            error=violation or "unknown_error"
        )
    if not p.exists() or not p.is_dir():
        return tool_response(
            tool="files_list",
            ok=False,
            data={"dir": str(p), "entries": []},
            error="not_a_directory"
        )
    try:
        pats = list(patterns or [])
        try:
            debug.note("resolved_dir", str(p))
            debug.note("files_only", files_only)
            debug.note("recurse", recurse)
            debug.note("patterns", pats)
        except Exception:
            pass
        entries: List[Dict[str, Any]] = []
        truncated = False
        for child in _iter_list(p, recurse):
            if files_only and not child.is_file():
                continue
            if pats:
                matched = any(fnmatch.fnmatch(child.name, pat) or fnmatch.fnmatch(str(child), pat) for pat in pats)
                if not matched:
                    continue
            kind = "dir" if child.is_dir() else ("file" if child.is_file() else "other")
            entries.append({"name": child.name, "path": str(child), "type": kind})
            if len(entries) >= limit:
                truncated = True
                break
        try:
            debug.note("entries_count", len(entries))
            debug.note("truncated", truncated)
        except Exception:
            pass
        return tool_response(
            tool="files_list",
            ok=True,
            data={"dir": str(p), "entries": entries, "truncated": truncated}
        )
    except Exception as e:
        return tool_response(
            tool="files_list",
            ok=False,
            data={"dir": str(p), "entries": []},
            error=f"{type(e).__name__}: {e}"
        )


@tool("files_read")
@dispInfo("fs_read")
def FILES_READ_TOOL(path: str, mode: str = "raw", max_bytes: int = 262144) -> str:
    """Read a file. mode: raw|head|tail. max_bytes caps bytes loaded."""
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        return tool_response(
            tool="files_read",
            ok=False,
            data={"path": str(path), "content": ""},
            error=violation or "unknown_error"
        )
    try:
        if not p.exists() or not p.is_file():
            return tool_response(
                tool="files_read",
                ok=False,
                data={"path": str(p), "content": ""},
                error="not_a_file"
            )
        size = p.stat().st_size
        try:
            debug.note("resolved_path", str(p))
            debug.note("file_size", size)
            debug.note("mode", mode)
            debug.note("max_bytes", max_bytes)
        except Exception:
            pass
        encoding = "utf-8"
        data: bytes
        with open(p, "rb") as f:
            if mode == "head":
                data = f.read(max_bytes)
                truncated = size > len(data)
            elif mode == "tail":
                if size <= max_bytes:
                    data = f.read()
                    truncated = False
                else:
                    f.seek(size - max_bytes)
                    data = f.read(max_bytes)
                    truncated = True
            else:
                if size > max_bytes:
                    data = f.read(max_bytes)
                    truncated = True
                else:
                    data = f.read()
                    truncated = False
        try:
            text = data.decode("utf-8", errors="replace")
            encoding = "utf-8"
        except Exception:
            try:
                text = data.decode("latin-1", errors="replace")
                encoding = "latin-1"
            except Exception:
                text = ""
                encoding = "binary"
        try:
            debug.note("bytes_read", len(data))
            debug.note("encoding", encoding)
            debug.note("truncated", truncated)
            debug.note("content_length", len(text))
        except Exception:
            pass
        return tool_response(
            tool="files_read",
            ok=True,
            data={
                "path": str(p),
                "content": text,
                "encoding": encoding,
                "size": len(text),
                "truncated": bool(truncated)
            }
        )
    except Exception as e:
        return tool_response(
            tool="files_read",
            ok=False,
            data={"path": str(p), "content": ""},
            error=f"{type(e).__name__}: {e}"
        )


@tool("files_find")
@dispInfo("fs_find")
def FILES_FIND_TOOL(start_dir: str, include_globs: Optional[List[str]] = None, exclude_globs: Optional[List[str]] = None, first_only: bool = False, limit: int = 2000) -> str:
    """Find files/dirs under start_dir using glob patterns.
    include_globs match either name or full path. exclude_globs are applied after include. If include is empty, include all.
    """
    ok, violation, p = _resolve_and_guard(start_dir)
    if not ok or p is None:
        return tool_response(
            tool="files_find",
            ok=False,
            data={"start_dir": str(start_dir), "matches": []},
            error=violation or "unknown_error"
        )
    if not p.exists() or not p.is_dir():
        return tool_response(
            tool="files_find",
            ok=False,
            data={"start_dir": str(p), "matches": []},
            error="not_a_directory"
        )
    try:
        inc = list(include_globs or [])
        exc = list(exclude_globs or [])
        try:
            debug.note("resolved_start_dir", str(p))
            debug.note("include_globs", inc)
            debug.note("exclude_globs", exc)
            debug.note("first_only", first_only)
        except Exception:
            pass
        matches: List[str] = []
        truncated = False
        for base, dirs, files in os.walk(p):
            base_path = Path(base)
            for name in files + dirs:
                fp = base_path / name
                rel_str = str(fp)
                match_inc = True if not inc else any(fnmatch.fnmatch(name, g) or fnmatch.fnmatch(rel_str, g) for g in inc)
                if not match_inc:
                    continue
                if exc and any(fnmatch.fnmatch(name, g) or fnmatch.fnmatch(rel_str, g) for g in exc):
                    continue
                matches.append(str(fp))
                if first_only:
                    try:
                        debug.note("first_match", str(fp))
                    except Exception:
                        pass
                    return tool_response(
                        tool="files_find",
                        ok=True,
                        data={
                            "start_dir": str(p),
                            "matches": [str(fp)],
                            "pattern": str(inc),
                            "truncated": False
                        }
                    )
                if len(matches) >= limit:
                    truncated = True
                    break
            if first_only or truncated:
                break
        try:
            debug.note("results_count", len(matches))
            debug.note("truncated", truncated)
        except Exception:
            pass
        return tool_response(
            tool="files_find",
            ok=True,
            data={
                "start_dir": str(p),
                "matches": matches,
                "pattern": str(inc),
                "truncated": truncated
            }
        )
    except Exception as e:
        return tool_response(
            tool="files_find",
            ok=False,
            data={"start_dir": str(p), "matches": []},
            error=f"{type(e).__name__}: {e}"
        )



@tool("files_read_section")
@dispInfo("fs_read_section")
def FILES_READ_SECTION_TOOL(path: str, start_line: int, end_line: int, max_chars: int = 262144) -> str:
    """Read lines in [start_line, end_line] (1-based, inclusive). Returns content and metadata.

    If the section exceeds max_chars, it will be truncated and flagged.
    """
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        return tool_response(
            tool="files_read_section",
            ok=False,
            data={"path": str(path), "content": ""},
            error=violation or "unknown_error",
        )
    try:
        if not p.exists() or not p.is_file():
            return tool_response(
                tool="files_read_section",
                ok=False,
                data={"path": str(p), "content": ""},
                error="not_a_file",
            )
        s = 1 if start_line is None or start_line < 1 else int(start_line)
        e = int(end_line) if end_line is not None and end_line >= s else s - 1
        if e < s:
            return tool_response(
                tool="files_read_section",
                ok=True,
                data={
                    "path": str(p),
                    "start_line": s,
                    "end_line": e,
                    "content": "",
                    "encoding": "utf-8",
                    "size": 0,
                    "truncated": False,
                },
            )
        collected: List[str] = []
        truncated = False
        char_count = 0
        # Stream lines to avoid loading whole file
        with open(p, "rb") as f:
            # decode line by line
            current_line_no = 0
            for raw in f:
                try:
                    line = raw.decode("utf-8", errors="replace")
                    enc = "utf-8"
                except Exception:
                    try:
                        line = raw.decode("latin-1", errors="replace")
                        enc = "latin-1"
                    except Exception:
                        line = ""
                        enc = "binary"
                current_line_no += 1
                if current_line_no < s:
                    continue
                if current_line_no > e:
                    break
                # enforce max_chars budget
                need = max_chars - char_count
                if need <= 0:
                    truncated = True
                    break
                if len(line) <= need:
                    collected.append(line)
                    char_count += len(line)
                else:
                    collected.append(line[:need])
                    char_count += need
                    truncated = True
                    break
        content = "".join(collected)
        return tool_response(
            tool="files_read_section",
            ok=True,
            data={
                "path": str(p),
                "start_line": s,
                "end_line": e,
                "content": content,
                "encoding": enc if 'enc' in locals() else "utf-8",
                "size": len(content),
                "truncated": bool(truncated),
            },
        )
    except Exception as e:
        return tool_response(
            tool="files_read_section",
            ok=False,
            data={"path": str(p), "content": ""},
            error=f"{type(e).__name__}: {e}",
        )


@tool("files_read_range")
@dispInfo("fs_read_range")
def FILES_READ_RANGE_TOOL(path: str, offset: int, length: int) -> str:
    """Read a byte range [offset, offset+length). Returns decoded content.

    truncated=True indicates there is more data after the requested range.
    """
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        return tool_response(
            tool="files_read_range",
            ok=False,
            data={"path": str(path), "content": ""},
            error=violation or "unknown_error",
        )
    try:
        if not p.exists() or not p.is_file():
            return tool_response(
                tool="files_read_range",
                ok=False,
                data={"path": str(p), "content": ""},
                error="not_a_file",
            )
        size = p.stat().st_size
        off = max(0, int(offset or 0))
        ln = max(0, int(length or 0))
        if ln == 0 or off >= size:
            return tool_response(
                tool="files_read_range",
                ok=True,
                data={
                    "path": str(p),
                    "offset": off,
                    "length": ln,
                    "content": "",
                    "encoding": "utf-8",
                    "size": 0,
                    "truncated": False,
                },
            )
        with open(p, "rb") as f:
            f.seek(off)
            data = f.read(ln)
        try:
            text = data.decode("utf-8", errors="replace")
            encoding = "utf-8"
        except Exception:
            try:
                text = data.decode("latin-1", errors="replace")
                encoding = "latin-1"
            except Exception:
                text = ""
                encoding = "binary"
        end_pos = off + len(data)
        truncated = end_pos < size
        return tool_response(
            tool="files_read_range",
            ok=True,
            data={
                "path": str(p),
                "offset": off,
                "length": ln,
                "content": text,
                "encoding": encoding,
                "size": len(text),
                "truncated": bool(truncated),
            },
        )
    except Exception as e:
        return tool_response(
            tool="files_read_range",
            ok=False,
            data={"path": str(p), "content": ""},
            error=f"{type(e).__name__}: {e}",
        )


@tool("files_grep")
@dispInfo("fs_grep")
def FILES_GREP_TOOL(start_dir: str, patterns: List[str], first_only: bool = False, include_globs: Optional[List[str]] = None, limit: int = 500) -> str:
    """Search recursively for regex patterns in text files.

    Returns a list of matches: {path, line_no, line, pattern}. Truncates after limit matches.
    """
    ok, violation, p = _resolve_and_guard(start_dir)
    if not ok or p is None:
        return tool_response(
            tool="files_grep",
            ok=False,
            data={"start_dir": str(start_dir), "matches": []},
            error=violation or "unknown_error",
        )
    if not p.exists() or not p.is_dir():
        return tool_response(
            tool="files_grep",
            ok=False,
            data={"start_dir": str(p), "matches": []},
            error="not_a_directory",
        )
    try:
        pats = [re.compile(pat, re.MULTILINE) for pat in (patterns or [])]
        globs = list(include_globs or [])
        matches: List[Dict[str, Any]] = []
        truncated = False
        for base, dirs, files in os.walk(p):
            base_path = Path(base)
            for name in files:
                fp = base_path / name
                if globs:
                    if not any(fnmatch.fnmatch(name, g) or fnmatch.fnmatch(str(fp), g) for g in globs):
                        continue
                # Best-effort text read
                try:
                    with open(fp, "rb") as f:
                        data = f.read()
                    try:
                        text = data.decode("utf-8", errors="replace")
                    except Exception:
                        text = data.decode("latin-1", errors="replace")
                except Exception:
                    continue
                lines = text.splitlines()
                for idx, line in enumerate(lines, start=1):
                    for pat in pats:
                        if pat.search(line):
                            matches.append({
                                "path": str(fp),
                                "line_no": idx,
                                "line": line[:400],
                                "pattern": pat.pattern,
                            })
                            if first_only:
                                return tool_response(
                                    tool="files_grep",
                                    ok=True,
                                    data={
                                        "start_dir": str(p),
                                        "matches": matches,
                                        "truncated": False,
                                        "patterns": [pt.pattern for pt in pats],
                                    },
                                )
                            if len(matches) >= limit:
                                truncated = True
                                break
                    if truncated:
                        break
                if truncated:
                    break
            if first_only or truncated:
                break
        return tool_response(
            tool="files_grep",
            ok=True,
            data={
                "start_dir": str(p),
                "matches": matches,
                "truncated": truncated,
                "patterns": [pt.pattern for pt in pats],
            },
        )
    except Exception as e:
        return tool_response(
            tool="files_grep",
            ok=False,
            data={"start_dir": str(p), "matches": []},
            error=f"{type(e).__name__}: {e}",
        )


@tool("md_outline")
@dispInfo("md_outline")
def MD_OUTLINE_TOOL(path: str) -> str:
    """Extract Markdown headings (#..######) and compute section ranges.

    Returns sections: [{level, title, line_no, start_line, end_line}].
    """
    ok, violation, p = _resolve_and_guard(path)
    if not ok or p is None:
        return tool_response(
            tool="md_outline",
            ok=False,
            data={"path": str(path), "sections": []},
            error=violation or "unknown_error",
        )
    try:
        if not p.exists() or not p.is_file():
            return tool_response(
                tool="md_outline",
                ok=False,
                data={"path": str(p), "sections": []},
                error="not_a_file",
            )
        with open(p, "rb") as f:
            data = f.read()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")
        lines = text.splitlines()
        headers: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines, start=1):
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                headers.append({"level": level, "title": title, "line_no": idx})
        # compute ranges
        sections: List[Dict[str, Any]] = []
        for i, h in enumerate(headers):
            start_ln = h["line_no"]
            end_ln = len(lines)
            for j in range(i + 1, len(headers)):
                if headers[j]["level"] <= h["level"]:
                    end_ln = headers[j]["line_no"] - 1
                    break
            sections.append({
                "level": h["level"],
                "title": h["title"],
                "line_no": h["line_no"],
                "start_line": start_ln,
                "end_line": end_ln,
            })
        return tool_response(
            tool="md_outline",
            ok=True,
            data={"path": str(p), "sections": sections, "count": len(sections)},
        )
    except Exception as e:
        return tool_response(
            tool="md_outline",
            ok=False,
            data={"path": str(p), "sections": []},
            error=f"{type(e).__name__}: {e}",
        )
