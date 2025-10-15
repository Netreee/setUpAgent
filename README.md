# LangGraph Agent 项目

这是一个基于LangGraph的Plan-Execute-Observe Agent模板项目。

## 项目结构

```
setUpAgent/
├── config.py          # 项目配置文件
├── main.py            # 主程序文件
├── env.example        # 环境变量配置示例
├── README.md          # 项目说明文档
└── .venv/             # 虚拟环境
```

## 配置说明

### 环境变量配置

项目使用环境变量来管理配置，特别是敏感信息如API密钥。

1. 复制 `env.example` 文件为 `.env`
2. 在 `.env` 文件中设置你的实际配置值

#### Kimi/Moonshot 配置（推荐）

如果你使用Kimi LLM，推荐使用以下配置：

```bash
# 创建 .env 文件
MOONSHOT_API_KEY=sk-eaR3G6MRXwkKLBEFioreVIssEBvjJO6bOfG2eDtA2CCpa0rP
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=kimi-k2-0905-preview
```

#### OpenAI 配置

如果你使用OpenAI：

```bash
# 创建 .env 文件
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_BASE_URL=https://api.openai.com/v1
```

支持的环境变量：

#### LLM配置
- `OPENAI_API_KEY`: OpenAI API密钥
- `MOONSHOT_API_KEY`: Kimi/Moonshot API密钥（推荐）
- `LLM_API_KEY`: 备用的API密钥名称
- `OPENAI_MODEL`: OpenAI模型名称（默认: gpt-3.5-turbo）
- `MOONSHOT_MODEL`: Kimi模型名称（默认: kimi-k2-0905-preview）
- `OPENAI_BASE_URL`: OpenAI API基础URL
- `MOONSHOT_BASE_URL`: Kimi API基础URL（默认: https://api.moonshot.cn/v1）
- `LLM_MODEL_NAME`: 通用模型名称
- `LLM_BASE_URL`: 通用API基础URL
- `LLM_TEMPERATURE`: 生成温度（默认: 0.7）
- `LLM_MAX_TOKENS`: 最大token数（默认: 1000）

#### 项目配置
- `DEBUG_MODE`: 调试模式（true/false，默认: false）
- `MAX_ITERATIONS`: 最大迭代次数（默认: 10）
- `COMPLETION_THRESHOLD`: 完成判断阈值（默认: 3）

### 使用配置文件

在代码中可以通过以下方式获取配置：

```python
from config import get_config, get_llm_config

# 获取项目配置
config = get_config()

# 获取LLM配置
llm_config = get_llm_config()

# 使用配置值
max_iterations = config.max_iterations
api_key = llm_config.api_key
```

## 运行项目

1. 激活虚拟环境：
   ```bash
   .venv\Scripts\activate  # Windows
   # 或者
   source .venv/bin/activate  # Linux/Mac
   ```

2. 运行主程序：
   ```bash
   python main.py
   ```

## 配置类说明

### LLMConfig
- `api_key`: API密钥
- `model_name`: 模型名称
- `base_url`: API基础URL
- `temperature`: 生成温度（0.0-2.0）
- `max_tokens`: 最大token数
- `timeout`: 请求超时时间

### ProjectConfig
- `project_name`: 项目名称
- `version`: 项目版本
- `max_iterations`: Agent最大迭代次数
- `debug_mode`: 调试模式开关
- `log_level`: 日志级别
- `data_dir`: 数据存储目录
- `cache_dir`: 缓存目录
- `logs_dir`: 日志目录
- `default_user_message`: 默认用户消息
- `completion_threshold`: 完成判断阈值

## 开发说明

1. 配置文件支持类型验证和默认值
2. 敏感信息通过环境变量管理
3. 支持调试模式输出详细信息
4. 配置管理器使用单例模式确保配置一致性

## 注意事项

1. 确保设置了有效的API密钥
2. 调试模式下会输出更多配置信息
3. 配置文件会在项目启动时自动加载
4. 支持热重载配置（通过ConfigManager.reload_config()）
