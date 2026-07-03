"""OpenAI-compatible command-line model backend.

The backend executes the local OpenAI command-line interface in non-interactive
mode and stores the final model response in the standard judgment schema.
"""

import subprocess
import tempfile
from pathlib import Path

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class CodexCLIBackend:
    """Run judgments through `codex exec`."""

    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        timeout = int(model_spec.params.get("timeout_seconds", 240))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "last_message.txt"
            command = [
                "codex",
                "exec",
                "--model",
                model_spec.name,
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--color",
                "never",
                "--output-last-message",
                str(output_file),
                "-",
            ]

            try:
                completed = subprocess.run(
                    command,
                    input=req.prompt,
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
                    error="codex CLI not found on PATH",
                )
            except subprocess.TimeoutExpired:
                return JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error=f"codex CLI timed out after {timeout}s",
                )

            raw_output = ""
            if output_file.exists():
                raw_output = output_file.read_text(encoding="utf-8").strip()
            if not raw_output:
                raw_output = completed.stdout.strip()

            if completed.returncode != 0:
                return JudgeResponse(
                    raw_output=raw_output,
                    usage=None,
                    status="error",
                    error=completed.stderr.strip() or f"codex CLI exited with {completed.returncode}",
                )

            return JudgeResponse(raw_output=raw_output, usage=None, status="ok")
