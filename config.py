#!/usr/bin/env.example python3
"""
项目配置文件
处理整个项目的配置信息，包括LLM相关配置
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMConfig(BaseModel):
    """LLM配置类"""
    api_key: str = Field(default="", description="LLM API密钥")
    model_name: str = Field(default="kimi-k2-0905-preview", description="使用的模型名称")
    base_url: Optional[str] = Field(default="https://api.moonshot.cn/v1", description="API基础URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=1000, gt=0, description="最大token数")
    timeout: int = Field(default=30, description="请求超时时间(秒)")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证API密钥"""
        if not v or v.strip() == '':
            raise ValueError('API密钥不能为空')
        return v.strip()


class ProjectConfig(BaseSettings):
    """项目配置类"""
    # LLM配置
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # 项目基本配置
    project_name: str = Field(default="LangGraph Agent", description="项目名称")
    version: str = Field(default="1.0.0", description="项目版本")

    # 运行时配置
    max_iterations: int = Field(default=10, gt=0, description="Agent最大迭代次数")
    debug_mode: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")

    # 路径配置
    data_dir: str = Field(default="./data", description="数据存储目录")
    cache_dir: str = Field(default="./cache", description="缓存目录")
    logs_dir: str = Field(default="./logs", description="日志目录")
    agent_work_root: str = Field(default="./agent_work", description="Agent工作目录根路径")

    # Agent配置
    default_user_message: str = Field(
        default="请帮我完成一个任务",
        description="默认用户消息"
    )

    # 观察配置
    completion_threshold: int = Field(default=3, description="完成判断阈值")

    class Config:
        """Pydantic配置"""
        env_nested_delimiter = '__'
        case_sensitive = False

    def __init__(self, **kwargs):
        # 从环境变量获取LLM配置
        env_llm_config = {}

        # 支持多种环境变量命名方式
        env_vars = {
            'api_key': ['OPENAI_API_KEY', 'LLM_API_KEY', 'MOONSHOT_API_KEY'],
            'model_name': ['OPENAI_MODEL', 'LLM_MODEL_NAME', 'MOONSHOT_MODEL'],
            'base_url': ['OPENAI_BASE_URL', 'LLM_BASE_URL', 'MOONSHOT_BASE_URL'],
            'temperature': ['OPENAI_TEMPERATURE', 'LLM_TEMPERATURE'],
            'max_tokens': ['OPENAI_MAX_TOKENS', 'LLM_MAX_TOKENS'],
            'timeout': ['OPENAI_TIMEOUT', 'LLM_TIMEOUT']
        }

        for field, env_names in env_vars.items():
            for env_name in env_names:
                if env_name in os.environ:
                    env_llm_config[field] = os.environ[env_name]
                    break

        # 如果有环境变量配置，创建LLM配置
        if env_llm_config:
            kwargs['llm'] = LLMConfig(**env_llm_config)

        # 兼容 AGENT_WORK_DIR 作为 agent_work_root 的后备环境变量
        try:
            agent_work_dir_env = os.environ.get('AGENT_WORK_DIR')
            if agent_work_dir_env and 'agent_work_root' not in kwargs:
                kwargs['agent_work_root'] = agent_work_dir_env
        except Exception:
            pass

        super().__init__(**kwargs)


class ConfigManager:
    """配置管理器"""

    _instance: Optional['ConfigManager'] = None
    _config: Optional[ProjectConfig] = None

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_config(cls) -> ProjectConfig:
        """获取配置实例"""
        if cls._config is None:
            # 从环境变量获取LLM配置
            env_llm_config = {}

            # 支持多种环境变量命名方式
            env_vars = {
                'api_key': ['OPENAI_API_KEY', 'LLM_API_KEY', 'MOONSHOT_API_KEY'],
                'model_name': ['OPENAI_MODEL', 'LLM_MODEL_NAME', 'MOONSHOT_MODEL'],
                'base_url': ['OPENAI_BASE_URL', 'LLM_BASE_URL', 'MOONSHOT_BASE_URL'],
                'temperature': ['OPENAI_TEMPERATURE', 'LLM_TEMPERATURE'],
                'max_tokens': ['OPENAI_MAX_TOKENS', 'LLM_MAX_TOKENS'],
                'timeout': ['OPENAI_TIMEOUT', 'LLM_TIMEOUT']
            }

            for field, env_names in env_vars.items():
                for env_name in env_names:
                    if env_name in os.environ:
                        env_llm_config[field] = os.environ[env_name]
                        break

            # 如果有环境变量配置，创建LLM配置
            if env_llm_config:
                cls._config = ProjectConfig(llm=LLMConfig(**env_llm_config))
            else:
                cls._config = ProjectConfig()

            # 再次确保兼容 AGENT_WORK_DIR -> agent_work_root
            try:
                agent_work_dir_env = os.environ.get('AGENT_WORK_DIR')
                if agent_work_dir_env:
                    cls._config.agent_work_root = agent_work_dir_env
            except Exception:
                pass

        return cls._config

    @classmethod
    def reload_config(cls) -> ProjectConfig:
        """重新加载配置"""
        cls._config = ProjectConfig()
        return cls._config

    @classmethod
    def get_llm_config(cls) -> LLMConfig:
        """获取LLM配置"""
        return cls.get_config().llm

    @classmethod
    def is_debug_mode(cls) -> bool:
        """检查是否为调试模式"""
        return cls.get_config().debug_mode


# 全局配置实例
config = ConfigManager.get_config()


def get_config() -> ProjectConfig:
    """获取配置的便捷函数"""
    return config


def get_llm_config() -> LLMConfig:
    """获取LLM配置的便捷函数"""
    return config.llm


if __name__ == "__main__":
    # 测试配置加载
    print("=== 项目配置 ===")
    print(f"项目名称: {config.project_name}")
    print(f"版本: {config.version}")
    print(f"调试模式: {config.debug_mode}")
    print(f"最大迭代次数: {config.max_iterations}")

    print("\n=== LLM配置 ===")
    llm_config = config.llm
    print(f"模型名称: {llm_config.model_name}")
    print(f"温度: {llm_config.temperature}")
    print(f"最大token数: {llm_config.max_tokens}")
    print(f"API密钥: {'*' * (len(llm_config.api_key) - 4) + llm_config.api_key[-4:] if llm_config.api_key else '未设置'}")

    print("\n=== 路径配置 ===")
    print(f"数据目录: {config.data_dir}")
    print(f"缓存目录: {config.cache_dir}")
    print(f"日志目录: {config.logs_dir}")
