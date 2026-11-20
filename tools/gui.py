"""Simple PyQt6 GUI: run the VLM labeling script with buttons and browse the results.

Run from anywhere:  python tools/gui.py
Needs UNSLOTH_BASE_URL / UNSLOTH_API_KEY in .env (same as the CLI).
"""
import hashlib, html, os, shutil, sys, time, traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import dedup` works from anywhere
from dedup import registry as dedup_registry, store as dedup_store
from labeldb import LabelDB
from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QIcon, QImage, QImageReader, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QSizePolicy, QCheckBox, QDoubleSpinBox, QGridLayout, QComboBox, QLineEdit, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
CLASSES = yaml.safe_load((DATA / "configs" / "classes.yaml").read_text())
OVERLAYS = ["text", "logo_brand", "watermark"]
PAGE = 120
LOG_FILE = REPO / "gui.log"
LOG_COLORS = {"INFO": "#222", "OK": "#1b7a2b", "WARN": "#b26a00", "ERROR": "#c62828", "PROC": "#555"}
THUMB = 150
THUMB_CACHE = Path.home() / ".cache" / "face-labeling" / "thumbs"     # fast WSL disk, not /mnt/d

ASSETS = (REPO / "tools" / "assets").as_posix()
ACCENT = "#0078d7"

STYLE = """
* { font-family: "Segoe UI", "Open Sans", "Noto Sans", "DejaVu Sans", sans-serif; font-size: 13px; color: #1a1a1a; }
QMainWindow, QWidget#content, QWidget#page, QTabWidget::pane { background: white; border: none; }
QToolTip { background: #f2f2f2; border: 1px solid #767676; color: #1a1a1a; padding: 4px; }

/* sidebar */
QWidget#sidebar { background: #f2f2f2; }
QLabel#brand { font-size: 14px; padding: 20px 22px 14px 22px; background: transparent; }
QListWidget#nav { background: transparent; border: none; outline: 0; }
QListWidget#nav::item { padding: 11px 20px; border: none; border-left: 3px solid transparent; color: #1a1a1a; }
QListWidget#nav::item:hover { background: #e6e6e6; }
QListWidget#nav::item:selected { background: #e1e1e1; border: none; border-left: 3px solid #0078d7; color: #000; }
QLabel#sideLabel { color: #444; font-size: 12px; }
QProgressBar#sideBar { background: #d4d4d4; border: none; max-height: 4px; min-height: 4px; }
QProgressBar#sideBar::chunk { background: #0078d7; }

/* text */
QLabel#pageTitle { font-size: 30px; font-weight: 300; }
QLabel#muted { color: #767676; }
QLabel#section { font-size: 15px; }
QLabel#preview { background: #f2f2f2; color: #767676; }

/* buttons */
QPushButton { background: #e6e6e6; border: 1px solid transparent; padding: 6px 18px; min-height: 18px; }
QPushButton:hover { border: 1px solid #7a7a7a; }
QPushButton:pressed { background: #cccccc; }
QPushButton:disabled { background: #f2f2f2; color: #a0a0a0; border: 1px solid transparent; }
QPushButton#start { background: #0078d7; color: white; font-weight: 600; padding: 7px 24px; }
QPushButton#start:hover { background: #106ebe; border: 1px solid #106ebe; }
QPushButton#start:pressed { background: #005a9e; }
QPushButton#start:disabled { background: #b3d6f2; color: white; }
QPushButton#stop:enabled { color: #c42b1c; }
QPushButton#danger { background: white; border: 1px solid #c42b1c; color: #c42b1c; }
QPushButton#danger:hover { background: #fdf0ef; }
QPushButton#danger:disabled { border: 1px solid #e0e0e0; color: #a0a0a0; background: #f7f7f7; }
QPushButton#safe { background: white; border: 1px solid #0078d7; color: #0078d7; }
QPushButton#safe:hover { background: #eef6fd; }
QPushButton#safe:disabled { border: 1px solid #e0e0e0; color: #a0a0a0; background: #f7f7f7; }

/* inputs */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: white; border: 1px solid #8a8a8a; padding: 4px 8px; min-height: 20px; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: #333; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #0078d7; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url("ASSETS/chevron-down.svg"); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: white; border: 1px solid #8a8a8a; outline: 0;
    selection-background-color: #cce4f7; selection-color: #000; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; background: transparent; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("ASSETS/chevron-up.svg"); width: 8px; height: 8px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("ASSETS/chevron-down.svg"); width: 8px; height: 8px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #333; background: white; }
QCheckBox::indicator:hover { border-color: #0078d7; }
QCheckBox::indicator:checked { background: #0078d7; border-color: #0078d7; image: url("ASSETS/check.svg"); }
QProgressBar { background: #e6e6e6; border: none; max-height: 4px; min-height: 4px; }
QProgressBar::chunk { background: #0078d7; }

/* lists / grids */
QListWidget { background: white; border: none; outline: 0; }
QListWidget::item { color: #333; border: 1px solid transparent; }
QListWidget::item:hover { background: #f0f0f0; }
QListWidget::item:selected { background: #cce4f7; color: #000; border: 1px solid #0078d7; }

/* group boxes = section headings with an underline */
QGroupBox { border: none; border-top: 1px solid #d0d0d0; margin-top: 14px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 0; padding: 0 6px 0 0;
    font-size: 15px; }
QWidget#card { background: #0078d7; }
QWidget#card QLabel { color: white; background: transparent; }
QLabel#cardNum { font-size: 28px; font-weight: 300; }

/* console, status bar, splitters */
QPlainTextEdit#console { background: white; border: none; border-top: 1px solid #e0e0e0;
    font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 12px; }
QStatusBar { background: #f2f2f2; border-top: 1px solid #e0e0e0; }
QStatusBar QLabel, QStatusBar::item { border: none; color: #444; }
QSplitter::handle { background: #e6e6e6; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }

/* thin Windows 10 scrollbars */
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #c1c1c1; min-height: 30px; margin: 0 2px; }
QScrollBar::handle:vertical:hover { background: #a0a0a0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #c1c1c1; min-width: 30px; margin: 2px 0; }
QScrollBar::handle:horizontal:hover { background: #a0a0a0; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
""".replace("ASSETS", ASSETS)


def load_env() -> dict:
    env = {}
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_pixmap(path: Path, size: int) -> QPixmap:
    r = QImageReader(str(path))
    r.setAutoTransform(True)
    s = r.size()
    if s.isValid():
        s.scale(size, size, Qt.AspectRatioMode.KeepAspectRatio)
        r.setScaledSize(s)
    img = r.read()
    return QPixmap() if img.isNull() else QPixmap.fromImage(img)


def fmt_duration(sec: float) -> str:
    sec = int(max(0, round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def progress_text(done: int, total: int, started: float, unit: str = "", job_started: float = None) -> str:
    """'37 / 100 images · elapsed 15s · 2.4/s · ~26s left'.
    The estimate is based on `started` (this step); elapsed counts from `job_started` (whole job) if given."""
    now = time.monotonic()
    step = now - started
    text = f"{done:,} / {total:,}{unit}  ·  elapsed {fmt_duration(now - (job_started or started))}"
    if done > 0 and step >= 2 and done < total:
        rate = done / step
        text += f"  ·  {rate:.1f}/s  ·  ~{fmt_duration((total - done) / rate)} left"
    return text


def short_label(r: dict) -> str:
    g = {"female": "F", "male": "M"}.get(r["gender"], "?")
    return f"{r['age']} · {r['emotion']} · {g}"


# ---------------------------------------------------------------------------------------------
# Thumbnails: loaded by a thread pool, cached on the fast WSL disk, so the window never waits.
# ---------------------------------------------------------------------------------------------
def thumb_path(rel: str, size: int) -> Path:
    stable = rel.split("/", 1)[1] if "/" in rel else rel          # same cache entry in raw/ and discarded/
    h = hashlib.sha1(f"{stable}|{size}".encode()).hexdigest()
    return THUMB_CACHE / h[:2] / f"{h}.jpg"


def make_thumb(rel: str, size: int) -> QImage:
    cached = thumb_path(rel, size)
    if cached.exists():
        img = QImage(str(cached))
        if not img.isNull():
            return img
    r = QImageReader(str(DATA / rel))
    r.setAutoTransform(True)
    s = r.size()
    if s.isValid():
        s.scale(size, size, Qt.AspectRatioMode.KeepAspectRatio)
        r.setScaledSize(s)
    img = r.read()
    if not img.isNull():
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            img.convertToFormat(QImage.Format.Format_RGB32).save(str(cached), "JPEG", 88)
        except OSError:
            pass
    return img


class Batch:
    """Lets a page cancel all its still-queued thumbnail jobs when the user moves on."""
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Signals(QObject):
    done = pyqtSignal(str, int, QImage)


class _Task(QRunnable):
    def __init__(self, rel, size, batch, sig):
        super().__init__()
        self.rel, self.size, self.batch, self.sig = rel, size, batch, sig

    def run(self):
        if self.batch.cancelled:
            return
        try:
            img = make_thumb(self.rel, self.size)
        except Exception:
            img = QImage()
        self.sig.done.emit(self.rel, self.size, img)


class ThumbLoader:
    def __init__(self):
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(8)
        self.sig = _Signals()

    def request(self, rel: str, size: int, batch: Batch):
        self.pool.start(_Task(rel, size, batch, self.sig))


_loader = None


def thumbs() -> ThumbLoader:
    global _loader
    if _loader is None:
        _loader = ThumbLoader()
    return _loader


_placeholder = {}


def placeholder(size: int) -> QIcon:
    if size not in _placeholder:
        pm = QPixmap(size, size)
        pm.fill(QColor("#ececec"))
        _placeholder[size] = QIcon(pm)
    return _placeholder[size]


class PreviewLabel(QLabel):
    """Shows one image scaled to whatever space is available (re-scales when the window is resized)."""

    def __init__(self, text: str):
        super().__init__(text)
        self._pm = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, pm: QPixmap):
        self._pm = pm
        self._fit()

    def clear_image(self, text: str):
        self._pm = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(text)

    def _fit(self):
        if not self._pm.isNull():
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def sizeHint(self):
        return QSize(340, 340)


class Gallery(QWidget):
    """Filterable, paged thumbnail grid for one location (raw = kept, discarded). Data comes from SQLite."""

    def __init__(self, loc: str, win):
        super().__init__()
        self.loc, self.win = loc, win
        self.page, self.items, self.batch, self._last, self._quiet = 0, {}, None, None, False

        self.f_age, self.f_emo, self.f_gen = (QComboBox() for _ in range(3))
        for box, key in ((self.f_age, "age"), (self.f_emo, "emotion"), (self.f_gen, "gender")):
            box.addItems(["All"] + CLASSES[key])
        self.f_src = QComboBox(); self.f_src.addItem("All")
        self.f_ovl = QComboBox(); self.f_ovl.addItems(["All"] + OVERLAYS)
        for box in (self.f_age, self.f_emo, self.f_gen, self.f_src, self.f_ovl):
            box.currentIndexChanged.connect(self._filters_changed)

        top = QHBoxLayout()
        pairs = [("Age", self.f_age), ("Emotion", self.f_emo), ("Gender", self.f_gen), ("Source", self.f_src)]
        if loc == "discarded":
            pairs.append(("Overlay", self.f_ovl))
        for text, box in pairs:
            top.addWidget(QLabel(text)); top.addWidget(box)
        top.addStretch()

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setGridSize(QSize(THUMB + 46, THUMB + 42))
        self.grid.setWordWrap(True)
        self.grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.grid.itemSelectionChanged.connect(self._sel_changed)

        self.prev, self.next = QPushButton("◀ Prev"), QPushButton("Next ▶")
        self.prev.clicked.connect(lambda: self._go(-1))
        self.next.clicked.connect(lambda: self._go(1))
        self.info = QLabel()
        bottom = QHBoxLayout()
        bottom.addWidget(self.prev); bottom.addWidget(self.info, 1, Qt.AlignmentFlag.AlignCenter)
        bottom.addWidget(self.next)

        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addWidget(self.grid, 1); lay.addLayout(bottom)
        thumbs().sig.done.connect(self._thumb_ready)

    # ---- filters / paging ----
    def filters(self) -> dict:
        return {"age": self.f_age.currentText(), "emotion": self.f_emo.currentText(),
                "gender": self.f_gen.currentText(), "source": self.f_src.currentText(),
                "overlay": self.f_ovl.currentText()}

    def _filters_changed(self):
        self.page = 0
        self.refresh(force=True)

    def _go(self, d):
        self.page += d
        self.refresh(force=True)

    def _sel_changed(self):
        if not self._quiet:
            self.win.show_detail(self)

    def _sync_sources(self):
        names = ["All"] + self.win.db.sources(self.loc)
        if names != [self.f_src.itemText(i) for i in range(self.f_src.count())]:
            cur = self.f_src.currentText()
            self.f_src.blockSignals(True)
            self.f_src.clear(); self.f_src.addItems(names)
            if cur in names:
                self.f_src.setCurrentText(cur)
            self.f_src.blockSignals(False)

    # ---- data: SQL query for ONE page; in-place update so scroll position and selection survive ----
    def refresh(self, force: bool = False):
        db = self.win.db
        self._sync_sources()
        f = self.filters()
        total = db.count(self.loc, f)
        pages = max(1, -(-total // PAGE))
        self.page = min(self.page, pages - 1)
        rows = db.page(self.loc, f, PAGE, self.page * PAGE)
        key = (tuple(sorted(f.items())), self.page)
        self._quiet = True
        self.grid.setUpdatesEnabled(False)
        try:
            if force or key != self._last:
                self._rebuild(rows)
            else:
                self._diff(rows)
        finally:
            self.grid.setUpdatesEnabled(True)
            self._quiet = False
        self._last = key
        self.info.setText(f"{total:,} images  ·  page {self.page + 1} / {pages}")
        self.prev.setEnabled(self.page > 0)
        self.next.setEnabled(self.page + 1 < pages)
        self._sel_changed()

    def _add(self, r: dict):
        it = QListWidgetItem(placeholder(THUMB), short_label(r))
        it.setData(Qt.ItemDataRole.UserRole, r)
        it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.grid.addItem(it)
        self.items[r["path"]] = it
        thumbs().request(r["path"], THUMB, self.batch)

    def _rebuild(self, rows):
        if self.batch:
            self.batch.cancel()
        self.batch = Batch()
        self.grid.clear(); self.items = {}
        for r in rows:
            self._add(r)

    def _diff(self, rows):
        new = {r["path"] for r in rows}
        for path in [p for p in self.items if p not in new]:     # moved away / relabeled
            self.grid.takeItem(self.grid.row(self.items.pop(path)))
        for r in rows:                                           # freshly labeled images are appended
            if r["path"] not in self.items:
                self._add(r)

    def _thumb_ready(self, rel: str, size: int, img: QImage):
        it = self.items.get(rel)
        if it is None or size != THUMB or img.isNull():
            return
        try:
            it.setIcon(QIcon(QPixmap.fromImage(img)))
        except RuntimeError:                                     # item deleted meanwhile
            pass

    def selected(self) -> list:
        return [i.data(Qt.ItemDataRole.UserRole) for i in self.grid.selectedItems()]

    def shutdown(self):
        if self.batch:
            self.batch.cancel()


class BarChart(QWidget):
    """Horizontal bar chart: one row per class (fixed order, zero counts included)."""
    ROW = 26

    def __init__(self, order: list, color: str):
        super().__init__()
        self.order, self.color, self.counts = order, QColor(color), {}
        self.setMinimumHeight(self.ROW * len(order) + 8)
        self.setMinimumWidth(320)

    def set_counts(self, counts: dict):
        self.counts = dict(counts)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.palette().color(QPalette.ColorRole.WindowText)
        track = QColor(text); track.setAlpha(25)
        vals = [self.counts.get(k, 0) for k in self.order]
        total, top = sum(vals), max(max(vals, default=0), 1)
        label_w, value_w = 82, 112
        bar_w = max(40, self.width() - label_w - value_w - 8)
        for i, (k, v) in enumerate(zip(self.order, vals)):
            y = 4 + i * self.ROW
            p.setPen(text)
            p.drawText(0, y, label_w, self.ROW - 6, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, k)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(track); p.drawRect(label_w, y + 3, bar_w, self.ROW - 12)
            if v:
                p.setBrush(self.color)
                p.drawRect(label_w, y + 3, max(4, int(bar_w * v / top)), self.ROW - 12)
            p.setPen(text)
            pct = f"{100 * v / total:.1f}%" if total else "-"
            p.drawText(label_w + bar_w + 8, y, value_w, self.ROW - 6,
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{v:,}  ({pct})")


class StatsTab(QWidget):
    """Live label statistics; updated every time the labels are reloaded."""

    def __init__(self):
        super().__init__()
        self.scope = QComboBox()
        self.scope.addItems(["Kept images only", "Kept + discarded"])
        self.scope.currentIndexChanged.connect(lambda _: self._redraw())
        self.updated = QLabel(); self.updated.setObjectName("muted")
        head = QHBoxLayout()
        head.addWidget(QLabel("Count:")); head.addWidget(self.scope); head.addStretch(); head.addWidget(self.updated)

        self.cards = {}
        cards = QHBoxLayout(); cards.setSpacing(8)
        for key in ("Total images", "Labeled", "Kept", "Discarded", "Failed / invalid", "Unlabeled"):
            tile = QWidget(); tile.setObjectName("card")
            tl = QVBoxLayout(tile); tl.setContentsMargins(14, 10, 14, 12); tl.setSpacing(0)
            num = QLabel("0"); num.setObjectName("cardNum")
            tl.addWidget(QLabel(key)); tl.addWidget(num)
            self.cards[key] = num; cards.addWidget(tile)

        self.charts = {}
        grid = QGridLayout()
        specs = (("Age", "age", CLASSES["age"], "#0078d7"), ("Emotion", "emotion", CLASSES["emotion"], "#0063b1"),
                 ("Gender", "gender", CLASSES["gender"], "#2b88d8"),
                 ("Overlay (text / brand)", "overlay", ["none"] + OVERLAYS, "#c42b1c"))
        for i, (title, key, order, color) in enumerate(specs):
            box = QGroupBox(title); chart = BarChart(order, color)
            QVBoxLayout(box).addWidget(chart)
            self.charts[key] = chart
            grid.addWidget(box, i // 2, i % 2)

        lay = QVBoxLayout(self)
        lay.addLayout(head); lay.addLayout(cards); lay.addLayout(grid); lay.addStretch()
        self.db = None

    def update_data(self, db):
        self.db = db
        self._redraw()

    def _redraw(self):
        db, t = self.db, self.db.totals()
        for key, val in (("Total images", t["total"]), ("Labeled", t["labeled"]), ("Kept", t["kept"]),
                         ("Discarded", t["discarded"]), ("Failed / invalid", t["failed"]),
                         ("Unlabeled", t["unlabeled"])):
            self.cards[key].setText(f"{val:,}")
        kept_only = self.scope.currentIndex() == 0
        for key, chart in self.charts.items():
            chart.set_counts(db.distribution(key, kept_only and key != "overlay"))   # overlay images are never "kept"
        counted = t["kept"] if kept_only else t["labeled"]
        self.updated.setText(f"updated {datetime.now().strftime('%H:%M:%S')}  ·  {counted:,} images counted")


class DupTab(QWidget):
    """Find near-duplicate images (pluggable back-end), review the groups, keep 1 per group."""

    def __init__(self, win):
        super().__init__()
        self.win, self.proc, self.groups = win, None, {}
        self.pending_remove = self.n_moved = 0
        self.widgets, self._titems, self._tbatch = {}, {}, None
        self._job_t0 = self._phase_t0 = time.monotonic()
        self._phase, self._pdone, self._ptotal, self._label = "", 0, 0, ""
        self._tick = QTimer(self); self._tick.timeout.connect(self._tick_status)

        self.finder = QComboBox()
        offered = {n: c for n, c in dedup_registry.available().items() if not c.hidden}
        for i, (name, cls) in enumerate(offered.items()):
            self.finder.addItem(name, name)
            miss = cls.missing_requirements()
            tip = cls.description + (f"\nMissing: {', '.join(miss)}. {cls.install_hint}" if miss else "")
            self.finder.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
            if miss:
                self.finder.model().item(i).setEnabled(False)
        self.finder.currentIndexChanged.connect(self._build_params)
        self.param_box = QHBoxLayout(); self.param_box.setSpacing(8)
        self.limit = QSpinBox(); self.limit.setRange(0, 200000); self.limit.setFixedWidth(92)
        self.limit.setToolTip("Only look at N random images (0 = all). Handy for a quick test.")
        self.find_btn = QPushButton("Find duplicates"); self.find_btn.setObjectName("start")
        self.find_btn.clicked.connect(self.find)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setObjectName("stop"); self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(lambda: self.proc and self.proc.kill())

        top = QHBoxLayout(); top.setSpacing(8)
        lab = QLabel("Method"); lab.setObjectName("muted"); top.addWidget(lab); top.addWidget(self.finder)
        lab = QLabel("Limit"); lab.setObjectName("muted"); top.addWidget(lab); top.addWidget(self.limit)
        top.addStretch(); top.addWidget(self.find_btn); top.addWidget(self.stop_btn)
        self.param_box.addStretch()

        self.summary = QLabel(); self.summary.setObjectName("muted")
        self.progress = QProgressBar(); self.progress.setTextVisible(False)
        self.status = QLabel("Idle"); self.status.setObjectName("muted")
        prow = QHBoxLayout(); prow.addWidget(self.progress, 1); prow.addWidget(self.status)

        self.glist = QListWidget(); self.glist.setFixedWidth(260)
        self.glist.currentRowChanged.connect(lambda _: self._show_group())
        self.thumbs = QListWidget()
        self.thumbs.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbs.setMovement(QListWidget.Movement.Static)
        self.thumbs.setIconSize(QSize(THUMB + 30, THUMB + 30))
        self.thumbs.setGridSize(QSize(THUMB + 54, THUMB + 84))
        self.thumbs.setWordWrap(True)
        self.thumbs.itemSelectionChanged.connect(self._update_buttons)
        self.thumbs.itemDoubleClicked.connect(lambda _: self.make_keeper())
        thumbs().sig.done.connect(self._thumb_ready)
        body = QSplitter(); body.addWidget(self.glist); body.addWidget(self.thumbs)
        body.setStretchFactor(1, 1); body.setChildrenCollapsible(False)

        self.keep_btn = QPushButton("Keep this one instead")
        self.keep_btn.setToolTip("Select an image in the group (or double-click it) to make it the survivor.")
        self.keep_btn.clicked.connect(self.make_keeper)
        self.restore_btn = QPushButton("Restore moved duplicates"); self.restore_btn.setObjectName("safe")
        self.restore_btn.clicked.connect(self.restore)
        self.apply_btn = QPushButton("Move duplicates to discarded"); self.apply_btn.setObjectName("danger")
        self.apply_btn.clicked.connect(self.apply)
        bottom = QHBoxLayout(); bottom.setSpacing(8)
        bottom.addWidget(self.keep_btn); bottom.addStretch(); bottom.addWidget(self.restore_btn); bottom.addWidget(self.apply_btn)

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 8, 0, 0); lay.setSpacing(10)
        lay.addLayout(top); lay.addLayout(self.param_box); lay.addWidget(self.summary); lay.addLayout(prow)
        lay.addWidget(body, 1); lay.addLayout(bottom)
        self._build_params()
        self.loaded = False                 # the (large) group list is built on first visit

    # ---- method parameters, generated from the finder's declaration ----
    def _build_params(self):
        while self.param_box.count() > 1:                    # keep the trailing stretch
            it = self.param_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.widgets = {}
        cls = dedup_registry.available().get(self.finder.currentData())
        if not cls:
            return
        for key, spec in cls.params.items():
            if isinstance(spec.default, float):
                w = QDoubleSpinBox(); w.setDecimals(3)
                w.setRange(spec.min if spec.min is not None else 0, spec.max if spec.max is not None else 1e6)
                w.setSingleStep(spec.step or 0.01); w.setValue(spec.default); w.setFixedWidth(96)
            elif isinstance(spec.default, int):
                w = QSpinBox()
                w.setRange(int(spec.min if spec.min is not None else 0), int(spec.max if spec.max is not None else 10**9))
                w.setSingleStep(int(spec.step or 1)); w.setValue(spec.default); w.setFixedWidth(84)
            else:
                w = QLineEdit(str(spec.default)); w.setFixedWidth(110)
            w.setToolTip(spec.help)
            label = QLabel(key.replace("_", " ").capitalize()); label.setObjectName("muted")
            label.setToolTip(spec.help)
            at = max(0, self.param_box.count() - 1)
            self.param_box.insertWidget(at, label); self.param_box.insertWidget(at + 1, w)
            self.widgets[key] = w

    def _params(self) -> list:
        out = []
        for key, w in self.widgets.items():
            v = w.text() if isinstance(w, QLineEdit) else w.value()
            out += ["--param", f"{key}={v}"]
        return out

    # ---- data ----
    @property
    def busy(self) -> bool:
        return bool(self.proc and self.proc.state() != QProcess.ProcessState.NotRunning)

    def reload(self):
        self.loaded = True
        rows = dedup_store.read_report()
        self.groups = {}
        for r in rows:
            self.groups.setdefault(int(r["group_id"]), []).append(r)
        cur = self.glist.currentItem().data(Qt.ItemDataRole.UserRole) if self.glist.currentItem() else None
        self.glist.blockSignals(True); self.glist.clear()
        n_pend = n_moved = 0
        for gid in sorted(self.groups):
            rs = self.groups[gid]
            states = {r["status"] for r in rs}
            state = "pending" if "pending" in states else ("moved" if "moved" in states else "done")
            n_pend += sum(r["keeper"] == "0" and r["status"] == "pending" for r in rs)
            n_moved += sum(r["status"] == "moved" for r in rs)
            it = QListWidgetItem(f"#{gid}   {len(rs)} images   sim {float(rs[0]['score']):.3f}   {state}")
            it.setData(Qt.ItemDataRole.UserRole, gid)
            if state != "pending":
                it.setForeground(QColor("#888"))
            self.glist.addItem(it)
        self.glist.blockSignals(False)
        self.pending_remove, self.n_moved = n_pend, n_moved
        self.summary.setText(f"{len(self.groups):,} groups  ·  {n_pend:,} duplicate images waiting to be moved  ·  "
                             f"{n_moved:,} already moved  ·  one image per group is kept")
        self.apply_btn.setText(f"Move {n_pend:,} duplicates to discarded" if n_pend else "Move duplicates to discarded")
        row = 0
        for i in range(self.glist.count()):
            if self.glist.item(i).data(Qt.ItemDataRole.UserRole) == cur:
                row = i
        if self.glist.count():
            self.glist.setCurrentRow(row)
        self._show_group()

    def _thumb_ready(self, rel, size, img):
        it = self._titems.get(rel)
        if it is None or size != THUMB + 30 or img.isNull():
            return
        try:
            it.setIcon(QIcon(QPixmap.fromImage(img)))
        except RuntimeError:
            pass

    def _show_group(self):
        if self._tbatch:
            self._tbatch.cancel()
        self._tbatch, self._titems = Batch(), {}
        self.thumbs.clear()
        it = self.glist.currentItem()
        if it:
            for r in self.groups.get(it.data(Qt.ItemDataRole.UserRole), []):
                keep = r["keeper"] == "1"
                moved = r["status"] == "moved"
                head = "✔ KEEP" if keep else ("✕ moved" if moved else "✕ remove")
                size = f"{r['width']}×{r['height']} · {int(r['bytes']) // 1024:,} KB"
                t = QListWidgetItem(placeholder(THUMB + 30), f"{head}\n{size}\n{r['image'].split('/', 1)[1]}")
                rel = r["moved_to"] if moved else r["image"]
                self._titems[rel] = t
                thumbs().request(rel, THUMB + 30, self._tbatch)
                t.setData(Qt.ItemDataRole.UserRole, r)
                t.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
                if keep:
                    t.setBackground(QColor("#e3f1fb"))
                self.thumbs.addItem(t)
        self._update_buttons()

    def _update_buttons(self):
        free = not self.busy and not (self.win.proc and self.win.proc.state() != QProcess.ProcessState.NotRunning)
        sel = self.thumbs.selectedItems()
        can_keep = bool(sel) and sel[0].data(Qt.ItemDataRole.UserRole)["status"] == "pending" \
            and sel[0].data(Qt.ItemDataRole.UserRole)["keeper"] == "0"
        self.keep_btn.setEnabled(free and can_keep)
        self.apply_btn.setEnabled(free and self.pending_remove > 0)
        self.restore_btn.setEnabled(free and getattr(self, "n_moved", 0) > 0)
        self.find_btn.setEnabled(free); self.stop_btn.setEnabled(self.busy)

    def make_keeper(self):
        sel = self.thumbs.selectedItems()
        if not sel or not self.keep_btn.isEnabled():
            return
        r = sel[0].data(Qt.ItemDataRole.UserRole)
        dedup_store.set_keeper(DATA, r["group_id"], r["image"])
        self.win.log(f"group {r['group_id']}: keeper set to {r['image']}")
        self.reload()

    # ---- actions run through the command-line tool, so the GUI never freezes ----
    def find(self):
        if self.limit.value() == 0 and self.pending_remove:
            if QMessageBox.question(self, "Replace results?", f"{self.pending_remove:,} pending duplicates from the "
                    "previous search will be replaced by the new search. Continue?") != QMessageBox.StandardButton.Yes:
                return
        args = ["find", "--finder", self.finder.currentData(), "--limit", str(self.limit.value())] + self._params()
        self._run(args, "Searching for duplicates")

    def apply(self):
        if QMessageBox.question(self, "Move duplicates", f"Keep one image per group and move the other "
                f"{self.pending_remove:,} image(s) to data/discarded/?\n\nYou can bring them back with "
                "'Restore moved duplicates'.") == QMessageBox.StandardButton.Yes:
            self._run(["apply"], "Moving duplicates")

    def restore(self):
        if QMessageBox.question(self, "Restore", f"Move {self.n_moved:,} duplicate image(s) back to data/raw/?") \
                == QMessageBox.StandardButton.Yes:
            self._run(["restore"], "Restoring duplicates")

    def _run(self, args, label):
        if self.busy or (self.win.proc and self.win.proc.state() != QProcess.ProcessState.NotRunning):
            QMessageBox.information(self, "Busy", "Another job is running. Wait for it or press Stop.")
            return
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._output)
        self.proc.finished.connect(lambda code, _s: self._finished(code, label))
        self._job_t0 = self._phase_t0 = time.monotonic(); self._phase, self._pdone, self._ptotal = "", 0, 0
        self._label = label; self._tick.start(1000)
        self.progress.setRange(0, 0); self.status.setText(label + "…")
        self.win.log(f"{label}: python tools/dedup_cli.py " + " ".join(args))
        self.proc.start(sys.executable, ["-u", "tools/dedup_cli.py"] + args)
        self._update_buttons()

    def _tick_status(self):
        if not self.busy:
            return
        if self._ptotal:
            self.status.setText(f"{self._phase}   " + progress_text(
                self._pdone, self._ptotal, self._phase_t0, job_started=self._job_t0))
        else:
            self.status.setText(f"{self._label}…  elapsed {fmt_duration(time.monotonic() - self._job_t0)}")

    def _output(self):
        for line in bytes(self.proc.readAllStandardOutput()).decode(errors="replace").splitlines():
            if line.startswith("PROGRESS "):
                _, phase, done, total = line.split()
                if phase != self._phase:                          # new step: restart its timer
                    self._phase, self._phase_t0 = phase, time.monotonic()
                self._pdone, self._ptotal = int(done), int(total)
                self.progress.setRange(0, self._ptotal); self.progress.setValue(self._pdone)
                self._tick_status()
            elif line.strip():
                bad = "Traceback" in line or "Error" in line
                self.win.log("[dedup] " + line, "ERROR" if bad else ("WARN" if line.startswith("WARN") else "PROC"))

    def _finished(self, code, label):
        self.progress.setRange(0, 1); self.progress.setValue(1 if code == 0 else 0)
        self._tick.stop()
        took = fmt_duration(time.monotonic() - self._job_t0)
        self.status.setText(f"Done in {took}" if code == 0 else f"Stopped / failed after {took}")
        self.win.log(f"{label} finished after {took} (exit code {code})", "OK" if code == 0 else "WARN")
        self.win.refresh(force=True)    # image paths in the database may have changed
        self._update_buttons()


class MainWindow(QMainWindow):
    NAV = ["✔   Verified", "✕   Discarded", "▥   Statistics", "▣   Duplicates"]
    TITLES = ["Verified images", "Discarded images", "Statistics", "Duplicate images"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face attribute labeling")
        self.resize(1400, 860)
        self.setMinimumSize(1180, 680)
        self.proc = None
        self.target_n = 0
        self.db = LabelDB()
        self._grid_stale = False
        self._rev, self._last_refresh, self._shown = -1, 0.0, None
        self._t0, self._done = time.monotonic(), 0
        self._tick = QTimer(self); self._tick.timeout.connect(self._tick_status)

        # ---- labeling controls ----
        self.n = QSpinBox(); self.n.setRange(1, 200000); self.n.setValue(100)
        self.n.setToolTip("How many NEW (not yet labeled) images to process.")
        self.seed = QSpinBox(); self.seed.setRange(0, 999999)
        self.seed.setToolTip("Random order seed. Already-labeled images are always skipped.")
        self.workers = QSpinBox(); self.workers.setRange(1, 16); self.workers.setValue(4)
        self.workers.setToolTip("Parallel requests to the VLM server.")
        for sp in (self.n, self.seed, self.workers):
            sp.setFixedWidth(92)
        self.move_chk = QCheckBox("Discard overlay images"); self.move_chk.setChecked(True)
        self.start = QPushButton("Start labeling"); self.start.setObjectName("start")
        self.stop = QPushButton("Stop"); self.stop.setObjectName("stop"); self.stop.setEnabled(False)
        self.start.clicked.connect(self.start_run)
        self.stop.clicked.connect(self.stop_run)
        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.setValue(0)
        self.status = QLabel("Idle"); self.status.setObjectName("muted")
        self.log_view = QPlainTextEdit(); self.log_view.setObjectName("console")
        self.log_view.setReadOnly(True); self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Console log")

        self.refresh_btn = QPushButton("⟳  Refresh")
        self.refresh_btn.setShortcut("F5")
        self.refresh_btn.setToolTip("Reload labels from disk (F5)")
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        self.scan_btn = QPushButton("Scan new images")
        self.scan_btn.setToolTip("Find images you added to data/raw/ and register them as unlabeled.\n"
                                 "Fast: only folders that changed since the last scan are re-listed.")
        self.scan_btn.clicked.connect(self.scan_new)
        self.scan_proc = None
        self.auto_chk = QCheckBox("Auto refresh images")
        self.auto_chk.setChecked(True)
        self.auto_chk.setToolTip("Only affects the image grids (Verified / Discarded). Statistics, counts and the "
                                 "progress bars always update by themselves.")
        self.auto_chk.toggled.connect(lambda on: on and self._grid_stale and self.refresh())
        self.poll = QTimer(self); self.poll.timeout.connect(self._poll); self.poll.start(1000)

        # ---- pages (tab bar hidden; the sidebar is the navigation) ----
        self.kept = Gallery("raw", self)
        self.disc = Gallery("discarded", self)
        self.stats = StatsTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.kept, "Verified (kept)")
        self.tabs.addTab(self.disc, "Discarded")
        self.tabs.addTab(self.stats, "Statistics")
        self.dups = DupTab(self)
        self.tabs.addTab(self.dups, "Duplicates")
        self.tabs.tabBar().hide()
        for page in (self.kept, self.disc, self.stats, self.dups):
            page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.tabs.currentChanged.connect(self._on_tab)

        # ---- sidebar ----
        brand = QLabel("☰    Face labeling"); brand.setObjectName("brand")
        self.nav = QListWidget(); self.nav.setObjectName("nav")
        for text in self.NAV:
            it = QListWidgetItem(text); it.setSizeHint(QSize(0, 42)); self.nav.addItem(it)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.side_text = QLabel("Labeled"); self.side_text.setObjectName("sideLabel")
        self.side_count = QLabel("0 / 0"); self.side_count.setObjectName("sideLabel")
        self.side_bar = QProgressBar(); self.side_bar.setObjectName("sideBar"); self.side_bar.setTextVisible(False)
        srow = QHBoxLayout(); srow.addWidget(self.side_text); srow.addStretch(); srow.addWidget(self.side_count)
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(230)
        sl = QVBoxLayout(side); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(0)
        sl.addWidget(brand); sl.addWidget(self.nav, 1)
        foot = QVBoxLayout(); foot.setContentsMargins(22, 0, 22, 20); foot.setSpacing(6)
        foot.addLayout(srow); foot.addWidget(self.side_bar)
        sl.addLayout(foot)

        # ---- header: title + primary action, options, progress ----
        self.title = QLabel(self.TITLES[0]); self.title.setObjectName("pageTitle")
        head = QHBoxLayout(); head.setSpacing(8)
        head.addWidget(self.title); head.addStretch(); head.addWidget(self.start); head.addWidget(self.stop)
        opts = self.opts = QHBoxLayout(); opts.setSpacing(8)
        for label, w in (("Images", self.n), ("Seed", self.seed), ("Workers", self.workers)):
            lab = QLabel(label); lab.setObjectName("muted"); opts.addWidget(lab); opts.addWidget(w); opts.addSpacing(8)
        opts.addWidget(self.move_chk); opts.addStretch(); opts.addWidget(self.auto_chk)
        opts.addWidget(self.scan_btn); opts.addWidget(self.refresh_btn)
        prow = self.prow = QHBoxLayout(); prow.addWidget(self.progress, 1); prow.addWidget(self.status)

        # ---- detail panel ----
        self.preview = PreviewLabel("Select an image"); self.preview.setObjectName("preview")
        self.d = {k: QLabel("-") for k in ("Age", "Emotion", "Gender", "Overlay", "Source", "File")}
        for v in self.d.values():
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); v.setWordWrap(True)
        dform = QFormLayout(); dform.setVerticalSpacing(6); dform.setContentsMargins(0, 8, 0, 0)
        for k, v in self.d.items():
            lab = QLabel(k); lab.setObjectName("muted"); dform.addRow(lab, v)
        self.act = QPushButton(); self.act.clicked.connect(self.move_selected)
        self.open_btn = QPushButton("Open folder"); self.open_btn.clicked.connect(self.open_folder)
        self.hint = QLabel("Tip: Ctrl/Shift-click to select several images.")
        self.hint.setObjectName("muted"); self.hint.setWordWrap(True)
        dl = QVBoxLayout(); dl.setContentsMargins(16, 0, 0, 0); dl.setSpacing(10)
        dl.addWidget(self.preview, 1); dl.addLayout(dform); dl.addWidget(self.act)
        dl.addWidget(self.open_btn); dl.addWidget(self.hint)
        detail = self.detail = QWidget(); detail.setLayout(dl); detail.setMinimumWidth(386); detail.setMaximumWidth(436)

        split = QSplitter(); split.addWidget(self.tabs); split.addWidget(detail)
        split.setStretchFactor(0, 1); split.setChildrenCollapsible(False)
        clear_btn = QPushButton("Clear"); clear_btn.clicked.connect(self.log_view.clear)
        clear_btn.setToolTip("Clear the console (the file gui.log keeps the full history)")
        chead = QHBoxLayout(); chead.setContentsMargins(0, 8, 0, 6); chead.setSpacing(12)
        ctitle = QLabel("Console"); ctitle.setObjectName("section")
        chint = QLabel(f"also saved to {LOG_FILE.name}"); chint.setObjectName("muted")
        chead.addWidget(ctitle); chead.addWidget(chint); chead.addStretch(); chead.addWidget(clear_btn)
        console = QWidget(); cv = QVBoxLayout(console); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(0)
        cv.addLayout(chead); cv.addWidget(self.log_view)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(split); vsplit.addWidget(console)
        vsplit.setStretchFactor(0, 4); vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([600, 170]); vsplit.setCollapsible(0, False)

        content = QWidget(); content.setObjectName("content")
        cl = QVBoxLayout(content); cl.setContentsMargins(32, 22, 28, 12); cl.setSpacing(12)
        cl.addLayout(head); cl.addLayout(opts); cl.addLayout(prow); cl.addWidget(vsplit, 1)
        central = QWidget(); root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(side); root.addWidget(content, 1)
        self.setCentralWidget(central)

        sys.excepthook = self._on_exception
        self.log(f"GUI started. data dir: {DATA}")
        self.refresh(force=True)

    def _on_tab(self, i: int):
        self.nav.blockSignals(True); self.nav.setCurrentRow(i); self.nav.blockSignals(False)
        self.title.setText(self.TITLES[i])
        on_dups = i == 3                      # labeling controls are irrelevant there
        if on_dups and not self.dups.loaded:
            self.dups.reload()
        keep = (self.auto_chk, self.refresh_btn, self.scan_btn)
        for k in range(self.opts.count()):
            w = self.opts.itemAt(k).widget()
            if w is not None and w not in keep:
                w.setVisible(not on_dups)
        for w in (self.start, self.stop):
            w.setVisible(not on_dups)
        for k in range(self.prow.count()):
            w = self.prow.itemAt(k).widget()
            if w is not None:
                w.setVisible(not on_dups)
        self.show_detail(self.tabs.currentWidget())

    # ---- logging ----
    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendHtml(
            f'<span style="color:#999">{ts}</span> '
            f'<span style="color:{LOG_COLORS.get(level, "#222")}; white-space:pre-wrap">'
            f'{"" if level in ("INFO", "PROC") else "<b>" + level + "</b> "}'
            f'{html.escape(msg).replace(chr(10), "<br>")}</span>')
        try:
            with LOG_FILE.open("a") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')} {level} {msg}\n")
        except OSError:
            pass

    def _on_exception(self, etype, value, tb):
        self.log("Unhandled exception:\n" + "".join(traceback.format_exception(etype, value, tb)).rstrip(), "ERROR")

    # ---- data ----
    def _poll(self):
        """Cheap check (one tiny SQL query); only refreshes when a writer committed something new."""
        if time.monotonic() - self._last_refresh < 1.0:
            return
        if self.db.rev() != self._rev:
            self.refresh(grids=self.auto_chk.isChecked())      # stats / counts always; image grids only if enabled

    def _gallery(self):
        w = self.tabs.currentWidget()
        return w if isinstance(w, Gallery) else self.kept

    def refresh(self, force: bool = False, grids: bool = True):
        """Update from the database. Statistics, counts and progress bars always update; the image grids
        only when `grids` (auto-refresh on) or `force` (Refresh button). Grids update in place."""
        self._rev = self.db.rev()
        self._last_refresh = time.monotonic()
        if grids or force:
            self.kept.refresh(force=force); self.disc.refresh(force=force)
            self._grid_stale = False
        else:
            self._grid_stale = True
        self.refresh_btn.setText("⟳  Refresh  •" if self._grid_stale else "⟳  Refresh")
        self.refresh_btn.setToolTip("Reload the image grids from the database (F5)" +
                                    ("\nThe grids are out of date." if self._grid_stale else ""))
        self.stats.update_data(self.db)
        if isinstance(self.tabs.currentWidget(), DupTab) or (force and self.dups.loaded):
            self.dups.reload()
        t = self.db.totals()
        n_dup = self.db.dups_pending_count()
        self.nav.item(0).setText(f"{self.NAV[0]}   ({t['kept']:,})")
        self.nav.item(1).setText(f"{self.NAV[1]}   ({t['discarded']:,})")
        self.nav.item(3).setText(self.NAV[3] + (f"   ({n_dup:,})" if n_dup else ""))
        self.side_bar.setRange(0, max(t["total"], 1)); self.side_bar.setValue(t["labeled"])
        self.side_count.setText(f"{t['labeled']:,} / {t['total']:,}")
        self.statusBar().showMessage(f"{t['labeled']:,} labeled of {t['total']:,} images  ·  {t['unlabeled']:,} remaining")
        if force:
            self.log(f"Refreshed: {t['labeled']:,} labeled ({t['kept']:,} kept, {t['discarded']:,} discarded), "
                     f"{t['failed']:,} failed, {t['unlabeled']:,} unlabeled")

    # ---- scan for new images ----
    def scan_new(self):
        if self.scan_proc and self.scan_proc.state() != QProcess.ProcessState.NotRunning:
            return
        self.scan_btn.setEnabled(False)
        self.status.setText("Scanning data/raw/ …")
        self.log("Scanning data/raw/ for new images")
        self.scan_proc = QProcess(self)
        self.scan_proc.setWorkingDirectory(str(REPO))
        self.scan_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.scan_proc.readyReadStandardOutput.connect(self._scan_output)
        self.scan_proc.finished.connect(self._scan_finished)
        self.scan_proc.start(sys.executable, ["-u", "tools/scan.py"])

    def _scan_output(self):
        for line in bytes(self.scan_proc.readAllStandardOutput()).decode(errors="replace").splitlines():
            if line.strip():
                self.log(line.strip(), "ERROR" if ("Traceback" in line or "Error" in line) else "PROC")

    def _scan_finished(self, code, _status):
        self.scan_btn.setEnabled(True)
        self.status.setText("Scan finished" if code == 0 else "Scan failed (see console)")
        self.log("Scan finished" if code == 0 else f"Scan failed (exit code {code})", "OK" if code == 0 else "ERROR")
        self.refresh(force=True)

    # ---- labeling process ----
    def start_run(self):
        if self.dups.busy:
            QMessageBox.information(self, "Busy", "A duplicate job is running. Wait for it or press Stop.")
            return
        env = load_env()
        if "UNSLOTH_API_KEY" not in env or "UNSLOTH_BASE_URL" not in env:
            self.log("UNSLOTH_API_KEY / UNSLOTH_BASE_URL not found in .env", "ERROR")
            QMessageBox.warning(self, "Missing settings", "UNSLOTH_API_KEY / UNSLOTH_BASE_URL not found in .env")
            return
        pe = QProcessEnvironment.systemEnvironment()
        for k, v in env.items():
            pe.insert(k, v)
        args = ["-u", "tools/vlm_label.py", "--n", str(self.n.value()), "--seed", str(self.seed.value()),
                "--workers", str(self.workers.value())]
        if not self.move_chk.isChecked():
            args.append("--no-move")
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(pe); self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_output)
        self.proc.finished.connect(self.on_finished)
        self.target_n = self.n.value()
        self._t0, self._done = time.monotonic(), 0
        self._tick.start(1000)
        self.progress.setRange(0, self.target_n); self.progress.setValue(0)
        self.status.setText("Running…")
        self.set_running(True)
        self.log("Starting: python " + " ".join(args))
        self.proc.start(sys.executable, args)

    def set_running(self, running: bool):
        self.start.setEnabled(not running); self.stop.setEnabled(running)
        for w in (self.n, self.seed, self.workers, self.move_chk, self.act):
            w.setEnabled(not running)
        self.dups._update_buttons()

    def on_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.log(line, "ERROR" if ("Traceback" in line or "Error" in line) else "PROC")
            if "/" in line and line.split("/")[0].strip().isdigit():
                self._done = int(line.split("/")[0])
                self.progress.setValue(self._done)
                self._tick_status()

    def _tick_status(self):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.status.setText(progress_text(self._done, self.target_n, self._t0, " images"))

    def stop_run(self):
        if self.proc:
            self.status.setText("Stopping…")
            self.log("Stop requested - killing the labeling process (finished results are kept)", "WARN")
            self.proc.kill()

    def on_finished(self, code, _status):
        self.set_running(False)
        self._tick.stop()
        took = fmt_duration(time.monotonic() - self._t0)
        self.status.setText(f"Done in {took}" if code == 0 else f"Stopped after {took} (results so far are kept)")
        self.log(f"Labeling process finished after {took} (exit code {code})", "OK" if code == 0 else "WARN")
        if code == 0:
            self.progress.setValue(self.target_n)
        self.refresh(force=True)

    # ---- detail / actions ----
    def show_detail(self, gallery):
        if gallery is not self.tabs.currentWidget():
            return
        self.detail.setVisible(isinstance(gallery, Gallery))
        if not isinstance(gallery, Gallery):
            return
        sel = gallery.selected()
        sig = (sel[0]["path"], len(sel)) if sel else None
        in_disc = gallery is self.disc
        self.act.setObjectName("safe" if in_disc else "danger")
        self.act.setStyle(self.act.style())
        n = len(sel)
        self.act.setText(("↩  Restore to kept" if in_disc else "✕  Move to discarded") + (f" ({n})" if n else ""))
        self.act.setEnabled(n > 0 and not (self.proc and self.proc.state() != QProcess.ProcessState.NotRunning))
        self.open_btn.setEnabled(n > 0)
        if sig == self._shown:
            return                                              # same selection as before: nothing to reload
        self._shown = sig
        if not sel:
            self.preview.clear_image("Select an image")
            for v in self.d.values():
                v.setText("-")
            return
        r = sel[0]
        pm = load_pixmap(DATA / r["path"], 520)
        if pm.isNull():
            self.preview.clear_image("(cannot load image)")
            self.log(f"cannot load image {r['path']}", "WARN")
        else:
            self.preview.set_image(pm)
        self.d["Age"].setText(r["age"]); self.d["Emotion"].setText(r["emotion"])
        self.d["Gender"].setText(r["gender"]); self.d["Overlay"].setText(r["overlay"] or "-")
        self.d["Source"].setText(r["source"]); self.d["File"].setText(r["path"])
        if n > 1:
            self.d["File"].setText(f"{r['path']}  (+{n - 1} more selected)")

    def open_folder(self):
        sel = self._gallery().selected()
        if sel:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str((DATA / sel[0]["path"]).parent)))

    def move_selected(self):
        gallery = self._gallery()
        sel = gallery.selected()
        if not sel:
            return
        to_disc = gallery is self.kept
        mapping = {}
        missing = [r["path"] for r in sel if not (DATA / r["path"]).exists()]
        for r in sel:
            old = r["path"]
            if old in missing:
                continue
            new = ("discarded/" if to_disc else "raw/") + old.split("/", 1)[1]
            try:
                (DATA / new).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(DATA / old, DATA / new)
                mapping[old] = new
                self.log(f"moved {old} -> {new}")
            except OSError as e:
                self.log(f"could not move {old}: {e}", "ERROR")
        for m in missing:
            self.log(f"skipped {m}: file no longer exists", "WARN")
        if mapping:
            self.db.move_paths(mapping)
            self.log(f"database updated ({len(mapping)} paths)", "OK")
        self.refresh()
        msg = f"Moved {len(mapping)} image(s) to {'discarded' if to_disc else 'kept'}"
        if missing:
            msg += f"  ·  {len(missing)} skipped (file was already moved; list refreshed)"
        self.statusBar().showMessage(msg, 6000)

    def closeEvent(self, e):
        for pr in (self.proc, self.dups.proc):
            if pr and pr.state() != QProcess.ProcessState.NotRunning:
                pr.kill(); pr.waitForFinished(2000)
        for g in (self.kept, self.disc):
            g.shutdown()
        if self.dups._tbatch:
            self.dups._tbatch.cancel()
        thumbs().pool.waitForDone(1500)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
