"""Label images with the Unsloth VLM and discard images with text/brand overlays, in one pass.

Legacy folder labels are never used. Resumable; appends to data/labels/vlm_labels.csv.
Images flagged with an overlay are moved from data/raw/<src>/ to data/discarded/<src>/
and master.csv is updated to the new path.

Usage: python tools/vlm_label.py --n 100 [--seed 0] [--workers 4] [--no-move]
Env: UNSLOTH_BASE_URL, UNSLOTH_API_KEY (from .env)
"""
import argparse, base64, csv, json, os, random, re, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from openai import OpenAI

ROOT = Path("data")
MODEL = "unsloth/Qwen3.5-9B-GGUF"
CLASSES = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())
OVERLAY = ["none", "text", "logo_brand", "watermark"]
OUT = ROOT / "labels" / "vlm_labels.csv"
FIELDS = ["image", "vlm_age", "vlm_emotion", "vlm_gender", "vlm_overlay", "valid", "raw"]

PROMPT = (
    "Look at this image of a face.\n"
    "1. Classify the person's age group, emotion and gender.\n"
    "2. Check whether the image has any overlay added on top of the photo: visible text, captions, "
    "a logo or brand name, a watermark, stamps or UI elements. Answer \"none\" if the photo is clean.\n"
    "Reply with ONLY a JSON object, no other text:\n"
    + json.dumps({**{t: "one of " + json.dumps(c) for t, c in CLASSES.items()},
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
        return {"image": rel, "vlm_age": "", "vlm_emotion": "", "vlm_gender": "", "vlm_overlay": "",
                "valid": 0, "raw": f"ERR {type(e).__name__}: {e} | {raw}"[:500]}
    ok = all(d.get(t) in CLASSES[t] for t in CLASSES) and d.get("overlay") in OVERLAY
    return {"image": rel, "vlm_age": d.get("age"), "vlm_emotion": d.get("emotion"),
            "vlm_gender": d.get("gender"), "vlm_overlay": d.get("overlay"), "valid": int(ok),
            "raw": raw.replace("\n", " ")[:500]}

def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-move", action="store_true")
    a = ap.parse_args()

    with (ROOT / "labels/master.csv").open() as f:
        images = [r["image"] for r in csv.DictReader(f) if r["image"].startswith("raw/")]
    done = set()
    if OUT.exists():
        with OUT.open() as f:
            done = {r["image"] for r in csv.DictReader(f) if r["valid"] == "1"}
    random.Random(a.seed).shuffle(images)
    todo = [i for i in images if i not in done][: a.n]

    results = []
    new = not OUT.exists()
    with OUT.open("a", newline="") as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for i, row in enumerate(ex.map(label, todo), 1):
            w.writerow(row); f.flush(); results.append(row)
            if i % 10 == 0: print(f"{i}/{len(todo)}", flush=True)

    flagged = [r for r in results if r["valid"] == 1 and r["vlm_overlay"] != "none"]
    print(f"labeled {len(results)}, invalid {sum(r['valid']==0 for r in results)}, "
          f"overlay-flagged {len(flagged)}")
    if a.no_move or not flagged:
        return
    moved = {}
    for r in flagged:
        new_rel = "discarded/" + r["image"].split("/", 1)[1]      # raw/<src>/x -> discarded/<src>/x
        dst = ROOT / new_rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ROOT / r["image"], dst); moved[r["image"]] = new_rel
    mrows = list(csv.DictReader((ROOT / "labels/master.csv").open()))
    for r in mrows: r["image"] = moved.get(r["image"], r["image"])
    write_csv(ROOT / "labels/master.csv", mrows, list(mrows[0]))
    vrows = list(csv.DictReader(OUT.open()))
    for r in vrows: r["image"] = moved.get(r["image"], r["image"])
    write_csv(OUT, vrows, FIELDS)
    print(f"moved {len(moved)} images to data/discarded/")

if __name__ == "__main__":
    main()
