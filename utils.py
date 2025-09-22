
import openai
from config import get_llm_config, get_config
import os
import time

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

    while retry_count <= max_retries:
        try:
            resp = client.chat.completions.create(**request_params)
            return resp.choices[0].message.content
        except openai.RateLimitError as e:
            retry_count += 1
            if retry_count <= max_retries:
                print(f"遇到请求频率限制 (429)，等待5秒后重试 ({retry_count}/{max_retries})")
                time.sleep(5)
            else:
                print(f"已达到最大重试次数 ({max_retries})，请求失败")
                raise e
        except Exception as e:
            # 对于其他错误，直接抛出
            raise e