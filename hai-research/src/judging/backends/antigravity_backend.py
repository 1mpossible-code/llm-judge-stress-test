"""Google Antigravity command-line model backend.

The backend executes the local Antigravity CLI in non-interactive mode and
stores raw model outputs through the standard judgment schema.
"""

import subprocess

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class AntigravityBackend:
    """Run judgments through the local `agy` command."""

    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        timeout = str(model_spec.params.get("timeout", "120s"))
        command = [
            "agy",
            "-p",
            req.prompt,
            "--model",
            model_spec.name,
            "--print-timeout",
            timeout,
        ]

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_timeout_seconds(timeout),
            )
        except FileNotFoundError:
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error="agy CLI not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error=f"Antigravity CLI timed out after {timeout}",
            )

        raw_output = completed.stdout.strip()
        if completed.returncode != 0:
            return JudgeResponse(
                raw_output=raw_output,
                usage=None,
                status="error",
                error=completed.stderr.strip() or f"agy exited with {completed.returncode}",
            )

        return JudgeResponse(raw_output=raw_output, usage=None, status="ok")


def _timeout_seconds(timeout: str) -> int:
    """Parse simple agy timeout strings like 120s, 5m, or 1h."""
    timeout = timeout.strip().lower()
    if timeout.endswith("ms"):
        return max(1, int(timeout[:-2]) // 1000)
    if timeout.endswith("s"):
        return int(timeout[:-1])
    if timeout.endswith("m"):
        return int(timeout[:-1]) * 60
    if timeout.endswith("h"):
        return int(timeout[:-1]) * 3600
    return int(timeout)
