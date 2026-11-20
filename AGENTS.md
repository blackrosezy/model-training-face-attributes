# Agent notes

## Python env
Use `/home/cribpc/py-env` (activate: `source /home/cribpc/py-env/bin/activate`) and `uv pip install ...` for packages. `openai` is installed.

## Unsloth Studio (local LLM server, OpenAI-compatible)
- Runs on **Windows**, started with `unsloth studio -H 0.0.0.0 -p 8888` (must bind 0.0.0.0, otherwise WSL cannot reach it).
- From WSL use the Windows host IP, not localhost: `ip route | awk '/default/{print $3}'` (was `172.29.0.1`). Base URL: `http://<host-ip>:8888/v1`.
- Credentials are in `.env` (gitignored): `UNSLOTH_API_KEY`, `UNSLOTH_BASE_URL`. Never print or commit the key.
- **API key is invalidated whenever Studio restarts** -> 401 "Invalid or expired API key". Ask the user for a new one (Studio UI: Settings > API keys) and update `.env`.
- Restart also unloads models -> 400 "No model loaded". Load one in the Chat UI, or enable Settings > API > Model auto-switch.
- Models seen: `unsloth/Qwen3.5-9B-GGUF` (UD-Q4_K_XL, vision, reasoning), `unsloth/Qwen3.8-27B-GGUF`.
- Port 8000 is an unrelated `http.server` serving `web/`; it is not Unsloth.

## Test call
```python
import os
from openai import OpenAI
c = OpenAI(base_url=os.environ["UNSLOTH_BASE_URL"], api_key=os.environ["UNSLOTH_API_KEY"])
r = c.chat.completions.create(model="unsloth/Qwen3.5-9B-GGUF",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}], max_tokens=300)
print(r.choices[0].message.content)
```
Load env first: `set -a; source .env; set +a`.
