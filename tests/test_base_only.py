"""仅测试 tools.base 模块（不需要依赖）"""
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 直接导入 base 模块避免触发 tools/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("tools.base", project_root / "tools" / "base.py")
tools_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools_base)
tool_response = tools_base.tool_response
parse_tool_response = tools_base.parse_tool_response


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
    print("  [OK] 成功情况")
    
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
    print("  [OK] 失败情况")


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
    assert parsed["error"] is None
    print("  [OK] 正常解析")
    
    # 异常 JSON
    parsed = parse_tool_response("{invalid json")
    assert parsed["ok"] == False
    assert parsed["tool"] == "unknown"
    assert parsed["error"] == "invalid_json"
    print("  [OK] 异常 JSON 处理")
    
    # 带 error 的响应
    json_str = json.dumps({
        "ok": False,
        "tool": "test",
        "data": {},
        "error": "something wrong"
    })
    parsed = parse_tool_response(json_str)
    assert parsed["error"] == "something wrong"
    print("  [OK] 错误信息解析")


def test_unified_format_structure():
    """验证统一格式结构"""
    print("验证统一格式结构...")
    
    # 成功响应必需字段
    success = tool_response(
        tool="example",
        ok=True,
        data={"result": "success"}
    )
    parsed = json.loads(success)
    required_fields = ["ok", "tool", "data"]
    for field in required_fields:
        assert field in parsed, f"缺少必需字段: {field}"
    assert "error" not in parsed, "成功响应不应包含 error"
    print("  [OK] 成功响应格式")
    
    # 失败响应必需字段
    failure = tool_response(
        tool="example",
        ok=False,
        data={"path": "/test"},
        error="not_found"
    )
    parsed = json.loads(failure)
    for field in required_fields:
        assert field in parsed, f"缺少必需字段: {field}"
    assert "error" in parsed, "失败响应应包含 error"
    assert isinstance(parsed["error"], str), "error 应为字符串"
    print("  [OK] 失败响应格式")
    
    # 数据类型验证
    assert isinstance(parsed["ok"], bool), "ok 必须是 bool"
    assert isinstance(parsed["tool"], str), "tool 必须是字符串"
    assert isinstance(parsed["data"], dict), "data 必须是字典"
    print("  [OK] 数据类型正确")


if __name__ == "__main__":
    try:
        test_tool_response()
        test_parse_tool_response()
        test_unified_format_structure()
        
        print("\n" + "="*60)
        print("[SUCCESS] 所有基础测试通过！工具接口设计正确。")
        print("="*60)
        print("\n下一步：验证所有实际工具都已更新为统一格式")
    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

