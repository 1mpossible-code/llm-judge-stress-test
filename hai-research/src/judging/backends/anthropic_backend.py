"""Anthropic Claude backend"""

import os
import time
from typing import Any

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class AnthropicBackend:
    """Anthropic Claude API backend."""
    
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY environment variable not set. "
                        "Set it or remove Anthropic models from config."
                    )
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install -e .[anthropic]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
        return self._client
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Call Anthropic API with retry logic."""
        import anthropic
        
        max_retries = 5
        backoff_base = 1.0
        
        client = self._get_client()
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model_spec.name,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    messages=[
                        {"role": "user", "content": req.prompt}
                    ],
                )
                
                # Concatenate text blocks
                raw_output = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )
                
                # Extract usage
                usage = None
                if hasattr(response, "usage"):
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                
                return JudgeResponse(
                    raw_output=raw_output,
                    usage=usage,
                    status="ok",
                )
            
            except anthropic.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue
                return JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error="Rate limit exceeded",
                )
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue
                return JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error=str(e),
                )
        
        return JudgeResponse(
            raw_output="",
            usage=None,
            status="error",
            error="Max retries exceeded",
        )

