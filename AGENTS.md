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

## Labels, GUI and dedup (tools/)
- **Label store is SQLite**, not CSV: `~/.local/share/face-labeling/labels.db` (override: env `FACE_LABELS_DB`). It must stay on the
  WSL ext4 disk - SQLite on `/mnt/d` is ~200x slower. Images are in `data/raw/` and `data/discarded/`; `data` is a symlink to `/home/cribpc/face-data` (WSL ext4, moved off `/mnt/d` on 2026-11-21 via `tools/move_data_to_wsl.sh`; the old copy is `data.d_backup/`).
  Old CSVs are in `data/labels/csv_backup/`. `python tools/labeldb.py stats|export|migrate`.
- **Same-person links**: `images.person_id` (NULL = not clustered, treat as its own person) + table `persons`. Filled **automatically, as soon as `dedup_cli.py find` has found its groups** (torch_gpu and face_id groups only, DB only); manual: `python tools/persons.py from-dups [--apply]` (dry run by default), `stats`, `set` for manual fixes. Use it to split train/val/test by person.
- GUI thumbnail cache: `~/.cache/face-labeling/thumbs/` (safe to delete). Run the GUI with `./run_gui.sh`.
- Duplicate detection: `tools/dedup/` (pluggable finders, see its README). GUI offers `torch_gpu` (default, exact duplicates) and `face_id`; `faiss_embed` is hidden but kept.
  `face_id` = same person at any angle: InsightFace `buffalo_l` ArcFace embeddings on GPU (onnxruntime-gpu; `import torch` first so CUDA libs load), threshold 0.5, cached in `data/cache/face_id_buffalo_l.npz`. Marked `slow`: ~35 img/s, so never run it on the whole dataset (~268k images, hours) without the user asking; use `--limit`.
  `python tools/dedup_cli.py find|apply|restore|list`.
- Bulk file moves on `/mnt/d` (anything still there) from WSL are slow (~4k files/min); native Windows `powershell.exe` renames are ~1k files/s.
