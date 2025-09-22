#!/usr/bin/env python3
"""
测试文件，用于测试 run_single 工具的功能
"""

import sys
import os
import tempfile
from pathlib import Path
from run_single import run_single_sync, run_single
import asyncio
import time

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")

    # 直接测试命令执行，不使用LLM
    test_cases = [
        ("echo Hello World > test.txt", "创建一个名为test.txt的文件，内容是'Hello World'"),
        ("dir", "列出当前目录的内容"),
        ("echo %DATE% %TIME%", "显示当前日期和时间"),
        ("echo 2+3|set /p = & set /a result=2+3 & echo %result%", "计算2加3的结果"),
        ("mkdir test_dir", "创建一个名为test的目录")
    ]

    for i, (cmd, instruction) in enumerate(test_cases, 1):
        print(f"测试 {i}: {instruction}")
        print(f"  直接执行命令: {cmd}")
        try:
            result = run_single_sync(f"执行命令: {cmd}", timeout=10)
            print(f"  状态码: {result['exit_code']}")
            print(f"  命令: {result['command']}")
            if result['stdout']:
                print(f"  输出: {result['stdout'][:200]}{'...' if len(result['stdout']) > 200 else ''}")
            if result['stderr']:
                print(f"  错误: {result['stderr']}")
            print()
        except Exception as e:
            print(f"  错误: {e}")
            print()

def test_error_handling():
    """测试错误处理"""
    print("=== 测试错误处理 ===")

    test_cases = [
        ("xyzabc123", "运行一个不存在的命令xyzabc123"),
        ("cd nonexistent_directory", "cd 到一个不存在的目录"),
        ("type C:\\Windows\\System32\\config\\sam", "尝试访问系统文件")
    ]

    for i, (cmd, instruction) in enumerate(test_cases, 1):
        print(f"测试 {i}: {instruction}")
        print(f"  直接执行命令: {cmd}")
        try:
            result = run_single_sync(f"执行命令: {cmd}", timeout=5)
            print(f"  状态码: {result['exit_code']}")
            print(f"  命令: {result['command']}")
            if result['stdout']:
                print(f"  输出: {result['stdout']}")
            print(f"  错误: {result['stderr']}")
            print()
        except Exception as e:
            print(f"  错误: {e}")
            print()

def test_security_check():
    """测试安全检查"""
    print("=== 测试安全检查 ===")

    dangerous_commands = [
        ("del /f /s /q C:\\Windows\\*.*", "删除系统文件"),
        ("format.com D: /fs:ntfs /q /y", "格式化硬盘"),
        ("icacls C:\\Windows\\System32 /grant Everyone:F", "改变系统文件权限"),
        ("takeown /f C:\\Windows\\System32", "获取系统文件所有权")
    ]

    for i, (cmd, instruction) in enumerate(dangerous_commands, 1):
        print(f"测试 {i}: {instruction}")
        print(f"  直接执行命令: {cmd}")
        try:
            result = run_single_sync(f"执行命令: {cmd}", timeout=5)
            print(f"  状态码: {result['exit_code']}")
            print(f"  命令: {result['command']}")
            print(f"  结果: {result['stderr']}")
            print()
        except Exception as e:
            print(f"  错误: {e}")
            print()

def test_timeout():
    """测试超时功能"""
    print("=== 测试超时功能 ===")

    # 这应该会超时
    timeout_cmd = "ping -n 10 192.0.2.1"  # ping一个不存在的地址10次
    print(f"测试超时: 长时间运行的命令")
    print(f"  直接执行命令: {timeout_cmd}")

    start_time = time.time()
    try:
        result = run_single_sync(f"执行命令: {timeout_cmd}", timeout=3)
        end_time = time.time()

        print(f"  状态码: {result['exit_code']}")
        print(f"  命令: {result['command']}")
        print(f"  结果: {result['stderr']}")
        print(f"  执行时间: {end_time - start_time:.2f}秒")
    except Exception as e:
        print(f"  错误: {e}")

def test_llm_integration():
    """测试LLM集成"""
    print("=== 测试LLM集成 ===")

    # 由于LLM API有限制，我们跳过实际调用测试
    print("跳过LLM集成测试（API速率限制）")
    print("在实际使用中，LLM会将自然语言转换为具体的shell命令")
    print("例如: '创建一个包含数字1到10的文件' -> 'seq 1 10 > numbers.txt'")
    print()

    # 直接测试命令执行
    test_commands = [
        ("seq 1 10 > numbers.txt", "创建一个包含数字1到10的文件"),
        ("dir /b | find /c /v \"\"", "计算当前目录中文件的数量"),
        ("echo print('Hello World') > hello.py", "创建一个简单的Python脚本")
    ]

    for i, (cmd, instruction) in enumerate(test_commands, 1):
        print(f"测试 {i}: {instruction}")
        print(f"  直接执行命令: {cmd}")
        try:
            result = run_single_sync(f"执行命令: {cmd}", timeout=10)
            print(f"  状态码: {result['exit_code']}")
            print(f"  命令: {result['command']}")
            if result['exit_code'] == 0:
                print(f"  输出: {result['stdout'][:300]}{'...' if len(result['stdout']) > 300 else ''}")
            else:
                print(f"  错误: {result['stderr']}")
            print()
        except Exception as e:
            print(f"  错误: {e}")
            print()

def test_async_version():
    """测试异步版本"""
    print("=== 测试异步版本 ===")

    async def test_async():
        test_cmd = "echo %CD%"
        result = await run_single(f"执行命令: {test_cmd}", timeout=5)
        print("异步测试结果:")
        print(f"  状态码: {result['exit_code']}")
        print(f"  命令: {result['command']}")
        print(f"  输出: {result['stdout']}")

    # 运行异步测试
    try:
        asyncio.run(test_async())
    except Exception as e:
        print(f"异步测试错误: {e}")

def main():
    """主测试函数"""
    print("开始测试 run_single 工具...")
    print("=" * 50)

    # 检查环境
    print("Python版本:", sys.version)
    print("工作目录:", os.getcwd())
    print("临时目录:", tempfile.gettempdir())
    print()

    try:
        # 运行所有测试
        test_basic_functionality()
        test_error_handling()
        test_security_check()
        test_timeout()
        test_llm_integration()
        test_async_version()

        print("=" * 50)
        print("测试完成！")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
