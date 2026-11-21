#!/usr/bin/env bash
# Move data/ from the slow /mnt/d (NTFS, 9p) to the WSL ext4 disk, leaving a symlink behind.
#
#   ./tools/move_data_to_wsl.sh            # copy -> verify -> swap in symlink
#   DST=/some/other/path ./tools/move_data_to_wsl.sh
#
# Safe by design:
#   * COPIES with rsync (original is untouched until the final, fast rename).
#   * Resumable: if interrupted, just run it again.
#   * Never deletes anything. The old folder is kept as data.d_backup; delete it yourself after checking.
# Close the GUI and any other script using data/ before running.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
SRC="$REPO/data"
DST="${DST:-$HOME/face-data}"
BACKUP="$REPO/data.d_backup"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "Pre-checks"
[ -L "$SRC" ] && die "$SRC is already a symlink -> $(readlink "$SRC"). Nothing to do."
[ -d "$SRC" ] || die "$SRC not found."
[ -e "$BACKUP" ] && die "$BACKUP already exists. Move it away first."
command -v rsync >/dev/null || die "rsync not installed."
if pgrep -f "tools/gui.py|vlm_label.py|dedup_cli.py|scan.py" >/dev/null; then
  die "GUI/label/dedup process is running. Close it first."
fi
case "$(df --output=fstype "$(dirname "$DST")" | tail -1 | tr -d ' ')" in
  9p|drvfs|fuseblk|virtiofs) die "$DST is on a Windows mount; pick a path on the WSL ext4 disk." ;;
esac

# No `du` over /mnt/d here: walking every file over 9p is very slow. data/ was measured at ~13 GB,
# so just require a fixed free-space floor (override with MIN_FREE_GB).
MIN_FREE_GB="${MIN_FREE_GB:-30}"
avail_kb=$(df --output=avail -k "$(dirname "$DST")" | tail -1)
echo "free space: $((avail_kb / 1024 / 1024)) GB at $(dirname "$DST") (need >= ${MIN_FREE_GB} GB)"
[ "$avail_kb" -gt $((MIN_FREE_GB * 1024 * 1024)) ] || die "Not enough free space."
echo "source    : $SRC"
echo "target    : $DST"

say "1/4 Copying (starts immediately; overall % is an estimate that firms up as it scans; Ctrl+C and re-run is safe)"
mkdir -p "$DST"
# Incremental recursion (default): copy while scanning. A full pre-scan over 9p is silent and very slow.
rsync -a --info=progress2,stats2 --human-readable "$SRC"/ "$DST"/

say "2/4 Verifying (dry-run compare by size + mtime; should list nothing)"
diffs=$(rsync -anr --itemize-changes -t "$SRC"/ "$DST"/ | grep -v '^\.d' || true)
if [ -n "$diffs" ]; then
  echo "$diffs" | head -20
  die "Copy differs from source (see above). Re-run the script; nothing was swapped."
fi
src_n=$(find "$SRC" -type f | wc -l)
dst_n=$(find "$DST" -type f | wc -l)
echo "files: source=$src_n  copy=$dst_n"
[ "$src_n" -eq "$dst_n" ] || die "File counts differ. Nothing was swapped."

say "3/4 Swapping in symlink"
mv "$SRC" "$BACKUP"            # same filesystem -> instant rename
ln -s "$DST" "$SRC"
ls -ld "$SRC"

say "4/4 Done"
cat <<EOF
data -> $DST   (images now on ext4)
Original kept at: $BACKUP   (still on /mnt/d, NOT deleted)

Next:
  1. Run ./run_gui.sh and check everything works.
  2. When happy, free the space yourself:  rm -rf "$BACKUP"
     (or on Windows: powershell.exe -c "Remove-Item -Recurse -Force 'D:\\bazilah\\model-training-face-attributes\\data.d_backup'" - much faster)
To undo:  rm "$SRC" && mv "$BACKUP" "$SRC"
EOF
