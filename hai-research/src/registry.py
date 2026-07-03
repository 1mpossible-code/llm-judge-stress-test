"""Backend registry for pluggable backends"""

from typing import Type

from src.judging.base import JudgeBackend
from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


def get_backend(backend_name: str) -> JudgeBackend:
    """Get backend instance by name.
    
    Args:
        backend_name: One of 'anthropic', 'openai', 'openai_compatible',
            'groq', 'gemini', 'gemini_cli', 'antigravity', 'claude_cli',
            'codex_cli', 'hf', 'mock'
    
    Returns:
        JudgeBackend instance
    """
    if backend_name == "mock":
        from src.judging.backends.mock_backend import MockBackend
        return MockBackend()
    
    elif backend_name == "anthropic":
        from src.judging.backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend()
    
    elif backend_name == "openai":
        from src.judging.backends.openai_backend import OpenAIBackend
        return OpenAIBackend()

    elif backend_name == "openai_compatible":
        from src.judging.backends.openai_backend import OpenAIBackend
        return OpenAIBackend(default_api_key_env="OPENAI_COMPATIBLE_API_KEY")

    elif backend_name == "groq":
        from src.judging.backends.openai_backend import OpenAIBackend
        return OpenAIBackend(
            default_api_key_env="GROQ_API_KEY",
            default_base_url_env="GROQ_API_BASE",
            default_base_url="https://api.groq.com/openai/v1",
        )

    elif backend_name == "gemini":
        from src.judging.backends.gemini_backend import GeminiBackend
        return GeminiBackend()

    elif backend_name == "gemini_cli":
        from src.judging.backends.gemini_cli_backend import GeminiCLIBackend
        return GeminiCLIBackend()

    elif backend_name == "antigravity":
        from src.judging.backends.antigravity_backend import AntigravityBackend
        return AntigravityBackend()

    elif backend_name == "claude_cli":
        from src.judging.backends.claude_cli_backend import ClaudeCLIBackend
        return ClaudeCLIBackend()

    elif backend_name == "codex_cli":
        from src.judging.backends.codex_cli_backend import CodexCLIBackend
        return CodexCLIBackend()
    
    elif backend_name == "hf":
        from src.judging.backends.hf_llama_backend import HuggingFaceBackend
        return HuggingFaceBackend()
    
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

