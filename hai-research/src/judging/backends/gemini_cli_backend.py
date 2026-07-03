"""Gemini command-line model backend.

The backend executes the local Gemini command in headless mode and stores raw
model outputs through the standard judgment schema.
"""

import subprocess

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class GeminiCLIBackend:
    """Run Gemini through the local `gemini` command in headless mode."""

    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Call `gemini -p` and capture stdout."""
        timeout = int(model_spec.params.get("timeout_seconds", 120))
        command = [
            "gemini",
            "--prompt",
            req.prompt,
            "--model",
            model_spec.name,
            "--output-format",
            "text",
            "--skip-trust",
        ]

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error="gemini CLI not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error=f"gemini CLI timed out after {timeout}s",
            )

        raw_output = completed.stdout.strip()
        if completed.returncode != 0:
            return JudgeResponse(
                raw_output=raw_output,
                usage=None,
                status="error",
                error=completed.stderr.strip() or f"gemini CLI exited with {completed.returncode}",
            )

        return JudgeResponse(raw_output=raw_output, usage=None, status="ok")
