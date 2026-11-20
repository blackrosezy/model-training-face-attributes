"""Label images with the Unsloth VLM and discard images with text/brand overlays, in one pass.

Legacy folder labels are never used. Resumable; results go to the label database (tools/labeldb.py).
Images flagged with an overlay are moved from data/raw/<src>/ to data/discarded/<src>/
and the database is updated to the new path.

Usage: python tools/vlm_label.py --n 100 [--seed 0] [--workers 4] [--no-move]
Env: UNSLOTH_BASE_URL, UNSLOTH_API_KEY (from .env)
"""
import argparse, base64, json, os, random, re, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeldb import LabelDB

ROOT = Path("data")
MODEL = "unsloth/Qwen3.5-9B-GGUF"
CLASSES = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())
OVERLAY = ["none", "text", "logo_brand", "watermark"]

PROMPT = (
    "Look at this image of a face. FIRST inspect it for any overlay that was ADDED ON TOP of the photo after it was taken: "
    "a watermark (even faint/semi-transparent, e.g. stock-photo names), a logo or brand name stamped "
    "on the image, or text/captions/subtitles.\n"
    "Do NOT count things that were physically part of the photographed scene, such as lettering or "
    "logos on clothing, hats, signs, badges or products.\n"
    "Then classify age, emotion, gender.\n"
    "Reply with ONLY a JSON object, keys in this order:\n"
    + json.dumps({"overlay_evidence": "short description of any added watermark/logo/text and where, or 'nothing'",
                  **{t: "one of " + json.dumps(c) for t, c in CLASSES.items()},
                  "overlay": "one of " + json.dumps(OVERLAY)}, indent=1)
)

client = OpenAI(base_url=os.environ["UNSLOTH_BASE_URL"], api_key=os.environ["UNSLOTH_API_KEY"])

def label(rel: str) -> dict:
    path = ROOT / rel
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    raw = ""
    try:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=1500, temperature=0.1,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}])
        raw = r.choices[0].message.content or ""
        d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception as e:
        return {"path": rel, "age": None, "emotion": None, "gender": None, "overlay": None,
                "valid": 0, "raw": f"ERR {type(e).__name__}: {e} | {raw}"[:500]}
    ok = all(d.get(t) in CLASSES[t] for t in CLASSES) and d.get("overlay") in OVERLAY
    return {"path": rel, "age": d.get("age"), "emotion": d.get("emotion"), "gender": d.get("gender"),
            "overlay": d.get("overlay"), "valid": int(ok), "raw": raw.replace("\n", " ")[:500]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-move", action="store_true")
    a = ap.parse_args()

    db = LabelDB()
    todo = db.candidates("raw", only_unlabeled=True)          # raw/ images that are not labeled yet
    random.Random(a.seed).shuffle(todo)
    todo = todo[: a.n]

    results, batch = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(label, todo), 1):
            results.append(row); batch.append(row)
            if len(batch) >= 5:                                 # small batches: live in the GUI, cheap on the DB
                db.set_labels(batch); batch = []
            if i % 10 == 0: print(f"{i}/{len(todo)}", flush=True)
    if batch:
        db.set_labels(batch)

    flagged = [r for r in results if r["valid"] == 1 and r["overlay"] != "none"]
    print(f"labeled {len(results)}, invalid {sum(r['valid']==0 for r in results)}, "
          f"overlay-flagged {len(flagged)}")
    if a.no_move or not flagged:
        return
    moved = {}
    for r in flagged:
        new_rel = "discarded/" + r["path"].split("/", 1)[1]      # raw/<src>/x -> discarded/<src>/x
        dst = ROOT / new_rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ROOT / r["path"], dst); moved[r["path"]] = new_rel
    db.move_paths(moved)
    print(f"moved {len(moved)} images to data/discarded/")


if __name__ == "__main__":
    main()
