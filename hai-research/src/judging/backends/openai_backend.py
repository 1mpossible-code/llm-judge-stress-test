"""OpenAI backend"""

import os
import time
from typing import Any

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class OpenAIBackend:
    """OpenAI-compatible chat completions backend.

    Supports OpenAI directly and OpenAI-compatible providers such as Groq,
    OpenRouter, Together, Fireworks, or a local vLLM server via model params:
    - api_key_env: environment variable containing the API key
    - base_url: provider base URL
    """
    
    def __init__(
        self,
        default_api_key_env: str = "OPENAI_API_KEY",
        default_base_url_env: str = "OPENAI_API_BASE",
        default_base_url: str | None = None,
    ):
        self.default_api_key_env = default_api_key_env
        self.default_base_url_env = default_base_url_env
        self.default_base_url = default_base_url
        self._clients = {}
    
    def _get_client(self, model_spec: ModelSpec):
        """Lazy initialization of provider-specific clients."""
        params = model_spec.params or {}
        api_key_env = params.get("api_key_env", self.default_api_key_env)
        base_url = params.get("base_url") or os.getenv(self.default_base_url_env) or self.default_base_url
        client_key = (api_key_env, base_url)

        if client_key not in self._clients:
            try:
                from openai import OpenAI
                api_key = os.getenv(api_key_env)
                if not api_key:
                    raise ValueError(
                        f"{api_key_env} environment variable not set. "
                        "Set it or remove this model from config."
                    )
                if base_url:
                    self._clients[client_key] = OpenAI(api_key=api_key, base_url=base_url)
                else:
                    self._clients[client_key] = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install -e .[openai]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI-compatible client: {e}")
        return self._clients[client_key]
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Call OpenAI-compatible API with retry logic."""
        client = self._get_client(model_spec)
        max_retries = 5
        backoff_base = 1.0
        
        # Add small delay between requests to avoid hitting rate limits
        # This is especially important with parallel workers
        time.sleep(0.15)  # 150ms delay between requests
        
        for attempt in range(max_retries):
            try:
                prompt = req.prompt + str(model_spec.params.get("prompt_suffix", ""))
                response = client.chat.completions.create(
                    model=model_spec.name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                )
                
                # Check if we got a valid response
                if not response.choices:
                    return JudgeResponse(
                        raw_output="",
                        usage=None,
                        status="error",
                        error="API returned no choices in response",
                    )
                
                raw_output = response.choices[0].message.content or ""
                
                # Warn if content is None (shouldn't happen normally)
                if response.choices[0].message.content is None:
                    print(f"WARNING: OpenAI API returned None for content (model: {model_spec.name})")
                
                # Extract usage
                usage = None
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }
                
                return JudgeResponse(
                    raw_output=raw_output,
                    usage=usage,
                    status="ok",
                )
            
            except Exception as e:
                error_str = str(e).lower()
                # Log the error for debugging
                if attempt == 0:  # Only log on first attempt to avoid spam
                    print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                
                if ("rate limit" in error_str or "429" in error_str) and attempt < max_retries - 1:
                    # Try to parse "try again in X ms" from error message
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    try:
                        import re
                        # Look for "try again in Xms" or "try again in X ms"
                        match = re.search(r"try again in (\d+)\s*ms", error_str, re.IGNORECASE)
                        if match:
                            parsed_ms = int(match.group(1))
                            wait_time = (parsed_ms / 1000.0) + 0.1  # Convert to seconds, add 100ms buffer
                            print(f"Rate limited, waiting {wait_time:.2f}s (as suggested by API)...")
                        else:
                            # Default to exponential backoff with minimum 2 seconds for rate limits
                            wait_time = max(2.0, backoff_base * (2 ** attempt))
                            print(f"Rate limited, waiting {wait_time:.2f}s (exponential backoff)...")
                    except (ValueError, AttributeError):
                        # Fallback to exponential backoff
                        wait_time = max(2.0, backoff_base * (2 ** attempt))
                    
                    time.sleep(wait_time)
                    continue
                
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue
                
                # Final attempt failed
                return JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error=f"{type(e).__name__}: {str(e)}",
                )
        
        return JudgeResponse(
            raw_output="",
            usage=None,
            status="error",
            error="Max retries exceeded",
        )

