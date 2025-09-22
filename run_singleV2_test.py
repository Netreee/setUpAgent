#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path

# 添加当前目录到 Python 路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent))

from run_singleV2 import run_single


async def test_persistent_shell():
    """测试持久化 shell 功能"""
    print("=== 测试持久化 Shell 功能 ===")
    
    step = 0
    token = None
    
    try:
        # ① 创建测试文件夹
        step += 1
        print(f"\nStep {step}: 在 D:\\0APython\\ 中创建测试文件夹...")
        token, result = await run_single(
            "New-Item -ItemType Directory -Path 'D:\\0APython\\test_persistent_shell' -Force",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        assert result['exit_code'] == 0, f"创建文件夹失败: {result}"
        print(f"✓ Step {step} 成功")
        
        # ② 进入测试文件夹
        step += 1
        print(f"\nStep {step}: 进入测试文件夹...")
        token, result = await run_single(
            "Set-Location 'D:\\0APython\\test_persistent_shell'",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        assert result['exit_code'] == 0, f"进入文件夹失败: {result}"
        print(f"✓ Step {step} 成功")
        
        # ③ 验证当前目录
        step += 1
        print(f"\nStep {step}: 验证当前目录...")
        token, result = await run_single(
            "Get-Location",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        print(f"工作目录: {result['work_dir']}")
        assert result['exit_code'] == 0, f"获取当前目录失败: {result}"
        current_path = result['stdout'].strip()
        expected_path = "D:\\0APython\\test_persistent_shell"
        if expected_path.lower() in current_path.lower():
            print(f"✓ Step {step} 成功 - 当前目录正确: {current_path}")
        else:
            print(f"⚠ Step {step} 警告 - 当前目录可能不正确: {current_path}")
            print(f"期望目录: {expected_path}")
        
        # ④ 创建虚拟环境
        step += 1
        print(f"\nStep {step}: 创建虚拟环境...")
        token, result = await run_single(
            "python -m venv test_venv",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        assert result['exit_code'] == 0, f"创建虚拟环境失败: {result}"
        print(f"✓ Step {step} 成功")
        
        # ⑤ 激活虚拟环境
        step += 1
        print(f"\nStep {step}: 激活虚拟环境...")
        token, result = await run_single(
            ".\\test_venv\\Scripts\\Activate.ps1",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        # 虚拟环境激活可能返回非0退出码但仍然成功，所以这里不强制检查退出码
        print(f"✓ Step {step} 完成")
        
        # ⑥ 在虚拟环境中安装 tqdm
        step += 1
        print(f"\nStep {step}: 在虚拟环境中安装 tqdm...")
        token, result = await run_single(
            "pip install tqdm",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        assert result['exit_code'] == 0, f"安装 tqdm 失败: {result}"
        print(f"✓ Step {step} 成功")
        
        # ⑦ 验证 tqdm 安装位置（应该在虚拟环境中）
        step += 1
        print(f"\nStep {step}: 验证 tqdm 安装位置...")
        token, result = await run_single(
            "python -c \"import tqdm; print(tqdm.__file__)\"",
            session_token=token
        )
        print(f"命令: {result['command']}")
        print(f"输出: {result['stdout']}")
        print(f"错误: {result['stderr']}")
        print(f"退出码: {result['exit_code']}")
        assert result['exit_code'] == 0, f"验证 tqdm 安装失败: {result}"
        
        # 检查 tqdm 是否安装在虚拟环境中
        tqdm_path = result['stdout'].strip()
        if "test_venv" in tqdm_path:
            print(f"✅ 持久化成功！tqdm 安装在虚拟环境中: {tqdm_path}")
            print("🎉 持久化 Shell 测试通过！虚拟环境状态正确保持！")
            return True
        else:
            print(f"❌ 持久化失败！tqdm 安装在全局环境中: {tqdm_path}")
            print(f"这说明虚拟环境激活状态未能在命令间保持")
            return False
        
        print(f"✓ Step {step} 成功")
            
    except Exception as e:
        print(f"✗ Step {step} 异常: {e}")
        return False


async def test_session_isolation():
    """测试会话隔离功能"""
    print("\n\n=== 测试会话隔离功能 ===")
    
    # 在会话1中设置变量
    print("\n会话1: 设置变量...")
    token1, result1 = await run_single("$session1_var = 'Session1Value'; echo $session1_var")
    print(f"会话1输出: {result1['stdout']}")
    assert "Session1Value" in result1['stdout'], "会话1变量设置失败"
    
    # 在会话2中设置不同的变量
    print("\n会话2: 设置变量...")
    token2, result2 = await run_single("$session2_var = 'Session2Value'; echo $session2_var")
    print(f"会话2输出: {result2['stdout']}")
    assert "Session2Value" in result2['stdout'], "会话2变量设置失败"
    assert token1 != token2, "会话token应该不同"
    
    # 在会话1中检查变量（应该仍然存在）
    print("\n会话1: 检查变量...")
    token1, result1 = await run_single("echo $session1_var", session_token=token1)
    print(f"会话1输出: {result1['stdout']}")
    assert "Session1Value" in result1['stdout'], "会话1变量应该仍然存在"
    
    # 在会话2中检查会话1的变量（应该不存在）
    print("\n会话2: 检查会话1的变量...")
    token2, result2 = await run_single("echo $session1_var", session_token=token2)
    print(f"会话2输出: {result2['stdout']}")
    # 会话2不应该能访问会话1的变量
    assert "Session1Value" not in result2['stdout'], "会话应该是隔离的"
    
    print("✓ 会话隔离测试通过！")


if __name__ == "__main__":
    async def main():
        success1 = await test_persistent_shell()
        await test_session_isolation()
        
        if success1:
            print("\n🎉 所有测试通过！")
        else:
            print("\n❌ 部分测试失败")
            sys.exit(1)
    
    asyncio.run(main())
