"""测试统一工具接口"""
import json
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.base import tool_response, parse_tool_response


def test_tool_response():
    """测试 tool_response 辅助函数"""
    print("测试 tool_response...")
    
    # 成功情况
    result = tool_response(
        tool="test_tool",
        ok=True,
        data={"foo": "bar", "count": 42}
    )
    parsed = json.loads(result)
    assert parsed["ok"] == True
    assert parsed["tool"] == "test_tool"
    assert parsed["data"]["foo"] == "bar"
    assert "error" not in parsed
    print("  ✓ 成功情况")
    
    # 失败情况
    result = tool_response(
        tool="test_tool",
        ok=False,
        data={"path": "/some/path"},
        error="file_not_found"
    )
    parsed = json.loads(result)
    assert parsed["ok"] == False
    assert parsed["error"] == "file_not_found"
    print("  ✓ 失败情况")


def test_parse_tool_response():
    """测试 parse_tool_response"""
    print("测试 parse_tool_response...")
    
    # 正常解析
    json_str = json.dumps({
        "ok": True,
        "tool": "files_read",
        "data": {"path": "/test", "content": "hello"}
    })
    parsed = parse_tool_response(json_str)
    assert parsed["ok"] == True
    assert parsed["tool"] == "files_read"
    assert parsed["data"]["content"] == "hello"
    print("  ✓ 正常解析")
    
    # 异常 JSON
    parsed = parse_tool_response("{invalid json")
    assert parsed["ok"] == False
    assert parsed["tool"] == "unknown"
    assert parsed["error"] == "invalid_json"
    print("  ✓ 异常 JSON 处理")


def test_files_exists():
    """测试 files_exists 工具"""
    print("测试 files_exists...")
    from tools.fs import FILES_EXISTS_TOOL
    
    # 检查当前文件（应该存在）
    result = FILES_EXISTS_TOOL(__file__)
    parsed = parse_tool_response(result)
    assert parsed["ok"] == True
    assert parsed["tool"] == "files_exists"
    assert parsed["data"]["exists"] == True
    print("  ✓ 文件存在")
    
    # 检查不存在的文件
    result = FILES_EXISTS_TOOL("nonexistent_file_12345.txt")
    parsed = parse_tool_response(result)
    assert parsed["ok"] == True
    assert parsed["tool"] == "files_exists"
    assert parsed["data"]["exists"] == False
    print("  ✓ 文件不存在")


def test_files_read():
    """测试 files_read 工具"""
    print("测试 files_read...")
    from tools.fs import FILES_READ_TOOL
    
    # 读取当前文件
    result = FILES_READ_TOOL(__file__)
    parsed = parse_tool_response(result)
    assert parsed["ok"] == True
    assert parsed["tool"] == "files_read"
    assert "测试统一工具接口" in parsed["data"]["content"]
    assert parsed["data"]["encoding"] == "utf-8"
    print("  ✓ 读取文件成功")
    
    # 读取不存在的文件
    result = FILES_READ_TOOL("nonexistent_file_12345.txt")
    parsed = parse_tool_response(result)
    assert parsed["ok"] == False
    assert parsed["tool"] == "files_read"
    assert "error" in parsed
    print("  ✓ 文件不存在错误处理")


def test_files_list():
    """测试 files_list 工具"""
    print("测试 files_list...")
    from tools.fs import FILES_LIST_TOOL
    
    # 列出 tests 目录
    result = FILES_LIST_TOOL("tests")
    parsed = parse_tool_response(result)
    assert parsed["ok"] == True
    assert parsed["tool"] == "files_list"
    assert isinstance(parsed["data"]["entries"], list)
    assert parsed["data"]["truncated"] == False
    print("  ✓ 列出目录")


def test_pyenv_python_info():
    """测试 pyenv_python_info 工具"""
    print("测试 pyenv_python_info...")
    from tools.pyenv import PYENV_PYTHON_INFO_TOOL
    
    result = PYENV_PYTHON_INFO_TOOL()
    parsed = parse_tool_response(result)
    assert parsed["ok"] == True
    assert parsed["tool"] == "pyenv_python_info"
    assert "candidates" in parsed["data"]
    print("  ✓ Python 探测")


def test_all_tools_return_unified_format():
    """验证所有工具都返回统一格式"""
    print("\n验证所有工具返回统一格式...")
    
    tools = [
        ("files_exists", lambda: __file__),
        ("files_read", lambda: __file__),
        ("files_list", lambda: "tests"),
        ("pyenv_python_info", lambda: None),
    ]
    
    for tool_name, arg_fn in tools:
        print(f"  检查 {tool_name}...", end="")
        
        # 动态导入工具
        if tool_name.startswith("files_"):
            from tools import fs
            tool_func = getattr(fs, tool_name.replace("files_", "FILES_").upper() + "_TOOL")
        elif tool_name.startswith("pyenv_"):
            from tools import pyenv
            tool_func = getattr(pyenv, tool_name.replace("pyenv_", "PYENV_").upper() + "_TOOL")
        
        # 调用工具
        arg = arg_fn()
        result = tool_func(arg) if arg else tool_func()
        
        # 解析结果
        parsed = parse_tool_response(result)
        
        # 验证必需字段
        assert "ok" in parsed, f"{tool_name} 缺少 ok 字段"
        assert "tool" in parsed, f"{tool_name} 缺少 tool 字段"
        assert "data" in parsed, f"{tool_name} 缺少 data 字段"
        assert isinstance(parsed["ok"], bool), f"{tool_name} ok 必须是 bool"
        assert isinstance(parsed["data"], dict), f"{tool_name} data 必须是 dict"
        
        print(" ✓")


if __name__ == "__main__":
    try:
        test_tool_response()
        test_parse_tool_response()
        test_files_exists()
        test_files_read()
        test_files_list()
        test_pyenv_python_info()
        test_all_tools_return_unified_format()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！统一工具接口正常工作。")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

