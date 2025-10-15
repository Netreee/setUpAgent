#!/usr/bin/env python3
"""Minimal script to test discover_react agent"""

from agent.discover_react import run_discover_react
import json
import os

if __name__ == "__main__":
    # Set up repo root environment variable for tools
    target_repo = r"D:\0APython\setupLab2\Real-Time-Voice-Cloning"
    os.environ["REPO_ROOT"] = target_repo
    
    goal = f"理解当前项目的 Python 环境配置，广泛收集事实，为安装与运行方案做准备"
    seed_facts = {"repo_root": target_repo}
    
    print(f"Goal: {goal}")
    print(f"Repo Root: {target_repo}\n")
    print("Running discover agent...\n")
    
    result = run_discover_react(goal, seed_facts)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(result["summary"])
    print("\n" + "=" * 80)
    print(f"Transcript: {len(result['transcript'])} turns")
    print("=" * 80)

