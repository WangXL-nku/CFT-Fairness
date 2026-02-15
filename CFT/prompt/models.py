# models.py
# Unified interface for different MLLMs used in CFT.

import os
from openai import OpenAI

# Default API endpoints
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# Model name mapping
QWEN_MODEL = "Qwen3-VL-235B-A22B-Instruct"
GLM_MODEL = "glm-4v"


def create_client(model_type: str, api_key: str = None):
    """
    Create an OpenAI-compatible client for the specified model type.

    Args:
        model_type: "qwen" or "glm"
        api_key: API key; if None, read from environment variable.

    Returns:
        OpenAI client instance and the model name string.
    """
    model_type = model_type.lower()

    if model_type == "qwen":
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set.")
        client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
        )
        model_name = QWEN_MODEL

    elif model_type == "glm":
        api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY is not set.")
        client = OpenAI(
            api_key=api_key,
            base_url=ZHIPU_BASE_URL,
        )
        model_name = GLM_MODEL

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'qwen' or 'glm'.")

    return client, model_name