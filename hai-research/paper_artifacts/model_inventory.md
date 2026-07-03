# Model inventory for paper runs

Access date for current experiments: 2026-06-25.

## Main reported runs

| Display name | Reported model identifier | Access route | Notes / references |
| --- | --- | --- | --- |
| Llama 3.1 8B | `llama-3.1-8b-instant` | Groq-hosted model endpoint | Groq model card: https://console.groq.com/docs/model/llama-3.1-8b-instant ; Meta Llama 3.1 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md |
| Qwen3 32B | `qwen/qwen3-32b` | Groq-hosted model endpoint | Groq model card: https://console.groq.com/docs/model/qwen/qwen3-32b ; Qwen3-32B model card: https://huggingface.co/Qwen/Qwen3-32B |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Anthropic-compatible model access | Anthropic models overview lists API ID `claude-haiku-4-5-20251001`: https://platform.claude.com/docs/en/about-claude/models/overview |
| GPT-5.4 | `gpt-5.4` | OpenAI-compatible model access | OpenAI model documentation: https://developers.openai.com/api/docs/models/gpt-5.4 |
| Gemini 3.5 Flash | `gemini-3.5-flash` | Google model access | Google Gemini documentation lists model ID `gemini-3.5-flash`: https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash |

## Preliminary runs

| Display name | Exact model ID | Status |
| --- | --- | --- |
| Llama 3.3 70B | `llama-3.3-70b-versatile` | Pilot run used for early pipeline validation; not included in the main reported experiment set. |

## Reporting convention

In paper prose and result tables, use display names only: `Llama 3.1 8B`, `Qwen3 32B`, `Claude Haiku 4.5`, `GPT-5.4`, and `Gemini 3.5 Flash`. In the model inventory table, report model identifiers rather than interface/tool names. Exact implementation metadata, access date, temperature, prompt hash, and run configuration are recorded in each run directory.
