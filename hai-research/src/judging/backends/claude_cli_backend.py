"""Anthropic-compatible command-line model backend.

The backend executes the local Claude command in non-interactive mode and stores
raw model outputs through the same judgment schema used by API providers.
"""

import subprocess

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class ClaudeCLIBackend:
    """Run judgments through the local `claude` command in print mode."""

    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        timeout = int(model_spec.params.get("timeout_seconds", 180))
        cost_limit_usd = str(model_spec.params.get("cost_limit_usd", "0.05"))

        system_prompt = model_spec.params.get(
            "system_prompt",
            "You are a content moderation annotation model. Follow the user's output schema exactly.",
        )
        command = [
            "claude",
            "--print",
            "--model",
            model_spec.name,
            "--system-prompt",
            system_prompt,
            "--output-format",
            "text",
            "--no-session-persistence",
            "--safe-mode",
            "--tools",
            "",
            "--max-budget-usd",
            cost_limit_usd,
            req.prompt,
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
                error="claude CLI not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error=f"claude CLI timed out after {timeout}s",
            )

        raw_output = completed.stdout.strip()
        if completed.returncode != 0:
            return JudgeResponse(
                raw_output=raw_output,
                usage=None,
                status="error",
                error=completed.stderr.strip() or f"claude CLI exited with {completed.returncode}",
            )

        return JudgeResponse(raw_output=raw_output, usage=None, status="ok")
