"""Duplicate-image tool (used by the GUI, also usable from a terminal).

  python tools/dedup_cli.py list
  python tools/dedup_cli.py find --finder torch_gpu --param threshold=0.95 [--limit 5000]
  python tools/dedup_cli.py apply      # keep 1 per group, move the rest to data/discarded/
  python tools/dedup_cli.py restore    # undo apply
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup import registry, store, pipeline  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def make_progress():
    last = {"t": 0.0}
    def progress(phase, done, total):
        now = time.time()
        if done >= total or now - last["t"] > 0.5:
            last["t"] = now
            print(f"PROGRESS {phase} {done} {total}", flush=True)
    return progress


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("find")
    f.add_argument("--finder", default="torch_gpu")
    f.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    f.add_argument("--limit", type=int, default=0, help="only look at N random images (0 = all)")
    sub.add_parser("apply")
    sub.add_parser("restore")
    a = ap.parse_args()

    if a.cmd == "list":
        for name, cls in registry.available().items():
            miss = cls.missing_requirements()
            print(f"{name:14s} {'OK' if not miss else 'MISSING ' + ','.join(miss):18s} {cls.description}"
                  + ("  [hidden in GUI]" if cls.hidden else ""))
            for k, p in cls.params.items():
                print(f"    {k} = {p.default!r}   {p.help}")
        for mod, err in registry.load_errors().items():
            print(f"{mod}: failed to load ({err})")
    elif a.cmd == "find":
        params = dict(kv.split("=", 1) for kv in a.param)
        t = time.time()
        res = pipeline.find(DATA, a.finder, params, a.limit, make_progress())
        print(f"RESULT groups={res['groups']} to_remove={res['to_remove']} images={res['images']} "
              f"seconds={time.time() - t:.1f}", flush=True)
    elif a.cmd == "apply":
        n = store.apply(DATA, make_progress())
        print(f"RESULT moved={n}", flush=True)
    elif a.cmd == "restore":
        n = store.restore(DATA, make_progress())
        print(f"RESULT restored={n}", flush=True)


if __name__ == "__main__":
    main()
