"""Google Gemini backend"""

import os
import time
from typing import Any

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class GeminiBackend:
    """Google Gemini API backend using the google-genai SDK."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai

                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set. "
                        "Set it or remove Gemini models from config."
                    )
                self._client = genai.Client(api_key=api_key)
            except ImportError as exc:
                raise ImportError(
                    "google-genai package not installed. Install with: pip install -e .[gemini]"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Gemini client: {exc}") from exc
        return self._client

    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Call Gemini API with retry logic."""
        client = self._get_client()
        max_retries = int(model_spec.params.get("max_retries", 5))
        backoff_base = float(model_spec.params.get("backoff_base_seconds", 1.0))

        for attempt in range(max_retries):
            try:
                from google.genai import types

                response = client.models.generate_content(
                    model=model_spec.name,
                    contents=req.prompt,
                    config=types.GenerateContentConfig(
                        temperature=req.temperature,
                        max_output_tokens=req.max_tokens,
                    ),
                )

                raw_output = getattr(response, "text", None) or ""
                usage = None
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta is not None:
                    usage = {
                        "input_tokens": getattr(usage_meta, "prompt_token_count", None),
                        "output_tokens": getattr(usage_meta, "candidates_token_count", None),
                    }

                return JudgeResponse(raw_output=raw_output, usage=usage, status="ok")

            except Exception as exc:
                error_str = str(exc).lower()
                if attempt < max_retries - 1 and (
                    "rate" in error_str or "429" in error_str or "unavailable" in error_str
                ):
                    wait_time = max(2.0, backoff_base * (2 ** attempt)) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt) + (time.time() % 1)
                    time.sleep(wait_time)
                    continue

                return JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )

        return JudgeResponse(raw_output="", usage=None, status="error", error="Max retries exceeded")
