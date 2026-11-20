"""Simple PyQt6 GUI: run the VLM labeling script with buttons and browse the results.

Run from anywhere:  python tools/gui.py
Needs UNSLOTH_BASE_URL / UNSLOTH_API_KEY in .env (same as the CLI).
"""
import csv, html, os, shutil, sys, traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from PyQt6.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QIcon, QImageReader, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QGridLayout, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
VLM_CSV = DATA / "labels" / "vlm_labels.csv"
MASTER_CSV = DATA / "labels" / "master.csv"
CLASSES = yaml.safe_load((DATA / "configs" / "classes.yaml").read_text())
OVERLAYS = ["text", "logo_brand", "watermark"]
PAGE = 120
LOG_FILE = REPO / "gui.log"
LOG_COLORS = {"INFO": "#222", "OK": "#1b7a2b", "WARN": "#b26a00", "ERROR": "#c62828", "PROC": "#555"}
THUMB = 150

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
QComboBox, QSpinBox { background: white; border: 1px solid #8a8a8a; padding: 4px 8px; min-height: 20px; }
QComboBox:hover, QSpinBox:hover { border-color: #333; }
QComboBox:focus, QSpinBox:focus { border-color: #0078d7; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url("ASSETS/chevron-down.svg"); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: white; border: 1px solid #8a8a8a; outline: 0;
    selection-background-color: #cce4f7; selection-color: #000; }
QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: none; background: transparent; }
QSpinBox::up-arrow { image: url("ASSETS/chevron-up.svg"); width: 8px; height: 8px; }
QSpinBox::down-arrow { image: url("ASSETS/chevron-down.svg"); width: 8px; height: 8px; }
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


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rewrite_paths(path: Path, mapping: dict) -> None:
    rows = read_rows(path)
    if not rows:
        return
    for r in rows:
        r["image"] = mapping.get(r["image"], r["image"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def short_label(r: dict) -> str:
    g = {"female": "F", "male": "M"}.get(r["vlm_gender"], "?")
    return f"{r['vlm_age']} · {r['vlm_emotion']} · {g}"


class Gallery(QWidget):
    """Filterable, paged thumbnail grid for one folder (raw = kept, discarded)."""

    def __init__(self, prefix: str, on_select):
        super().__init__()
        self.prefix, self.on_select = prefix, on_select
        self.rows, self.filtered, self.page = [], [], 0
        self._queue, self._timer = [], QTimer(self)
        self._timer.timeout.connect(self._load_chunk)

        self.f_age, self.f_emo, self.f_gen = (QComboBox() for _ in range(3))
        for box, key in ((self.f_age, "age"), (self.f_emo, "emotion"), (self.f_gen, "gender")):
            box.addItems(["All"] + CLASSES[key])
        self.f_src = QComboBox()
        self.f_ovl = QComboBox()
        self.f_ovl.addItems(["All"] + OVERLAYS)
        for box in (self.f_age, self.f_emo, self.f_gen, self.f_src, self.f_ovl):
            box.currentIndexChanged.connect(self._filters_changed)

        top = QHBoxLayout()
        pairs = [("Age", self.f_age), ("Emotion", self.f_emo), ("Gender", self.f_gen), ("Source", self.f_src)]
        if prefix == "discarded":
            pairs.append(("Overlay", self.f_ovl))
        for text, box in pairs:
            top.addWidget(QLabel(text)); top.addWidget(box)
        top.addStretch()

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setGridSize(QSize(THUMB + 24, THUMB + 42))
        self.grid.setWordWrap(True)
        self.grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.grid.itemSelectionChanged.connect(lambda: self.on_select(self))

        self.prev, self.next = QPushButton("◀ Prev"), QPushButton("Next ▶")
        self.prev.clicked.connect(lambda: self._go(-1))
        self.next.clicked.connect(lambda: self._go(1))
        self.info = QLabel()
        bottom = QHBoxLayout()
        bottom.addWidget(self.prev); bottom.addWidget(self.info, 1, Qt.AlignmentFlag.AlignCenter)
        bottom.addWidget(self.next)

        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addWidget(self.grid, 1); lay.addLayout(bottom)

    # -- data -----------------------------------------------------------
    def set_rows(self, all_rows: list):
        self.rows = [r for r in all_rows if r["valid"] == "1" and r["image"].startswith(self.prefix + "/")]
        sources = sorted({r["image"].split("/")[1] for r in self.rows})
        cur = self.f_src.currentText()
        self.f_src.blockSignals(True)
        self.f_src.clear(); self.f_src.addItems(["All"] + sources)
        if cur in sources:
            self.f_src.setCurrentText(cur)
        self.f_src.blockSignals(False)
        self._apply()

    def _filters_changed(self):
        self.page = 0
        self._apply()

    def _apply(self):
        def ok(r):
            return ((self.f_age.currentText() in ("All", r["vlm_age"]))
                    and (self.f_emo.currentText() in ("All", r["vlm_emotion"]))
                    and (self.f_gen.currentText() in ("All", r["vlm_gender"]))
                    and (self.f_src.currentText() in ("All", r["image"].split("/")[1]))
                    and (self.f_ovl.currentText() in ("All", r["vlm_overlay"])))
        self.filtered = [r for r in self.rows if ok(r)]
        self.page = min(self.page, max(0, (len(self.filtered) - 1) // PAGE))
        self._show_page()

    def _go(self, d):
        self.page += d
        self._show_page()

    def _show_page(self):
        self._timer.stop()
        self.grid.clear()
        chunk = self.filtered[self.page * PAGE:(self.page + 1) * PAGE]
        blank = QIcon(QPixmap(THUMB, THUMB))
        self._queue = []
        for r in chunk:
            it = QListWidgetItem(blank, short_label(r))
            it.setData(Qt.ItemDataRole.UserRole, r)
            it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.grid.addItem(it)
            self._queue.append(it)
        pages = max(1, -(-len(self.filtered) // PAGE))
        self.info.setText(f"{len(self.filtered):,} images  ·  page {self.page + 1} / {pages}")
        self.prev.setEnabled(self.page > 0)
        self.next.setEnabled(self.page + 1 < pages)
        self._timer.start(0)

    def _load_chunk(self):
        for _ in range(8):
            if not self._queue:
                self._timer.stop(); return
            it = self._queue.pop(0)
            try:
                r = it.data(Qt.ItemDataRole.UserRole)
                pm = load_pixmap(DATA / r["image"], THUMB)
                if not pm.isNull():
                    it.setIcon(QIcon(pm))
            except RuntimeError:  # item deleted meanwhile
                self._queue.clear(); return

    def selected(self) -> list:
        return [i.data(Qt.ItemDataRole.UserRole) for i in self.grid.selectedItems()]

    def select_paths(self, paths: set):
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.ItemDataRole.UserRole)["image"] in paths:
                it.setSelected(True)


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
        self.rows, self.total = [], 0

    def update_data(self, rows: list, total: int):
        self.rows, self.total = rows, total
        self._redraw()

    def _redraw(self):
        ok = [r for r in self.rows if r["valid"] == "1"]
        kept = [r for r in ok if r["image"].startswith("raw/")]
        disc = [r for r in ok if r["image"].startswith("discarded/")]
        for key, val in (("Total images", self.total), ("Labeled", len(ok)), ("Kept", len(kept)),
                         ("Discarded", len(disc)), ("Failed / invalid", len(self.rows) - len(ok)),
                         ("Unlabeled", max(0, self.total - len(ok)))):
            self.cards[key].setText(f"{val:,}")
        use = kept if self.scope.currentIndex() == 0 else ok
        for key, chart in self.charts.items():
            rows = ok if key == "overlay" else use  # overlay images are never in "kept"
            chart.set_counts(Counter(r.get("vlm_" + key, "") for r in rows))
        self.updated.setText(f"updated {datetime.now().strftime('%H:%M:%S')}  ·  {len(use):,} images counted")


class MainWindow(QMainWindow):
    NAV = ["✔   Verified", "✕   Discarded", "▥   Statistics"]
    TITLES = ["Verified images", "Discarded images", "Statistics"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face attribute labeling")
        self.resize(1400, 860)
        self.proc = None
        self.target_n = 0
        self.rows = []

        # ---- labeling controls ----
        self.n = QSpinBox(); self.n.setRange(1, 200000); self.n.setValue(100)
        self.n.setToolTip("How many NEW (not yet labeled) images to process.")
        self.seed = QSpinBox(); self.seed.setRange(0, 999999)
        self.seed.setToolTip("Random order seed. Already-labeled images are always skipped.")
        self.workers = QSpinBox(); self.workers.setRange(1, 16); self.workers.setValue(4)
        self.workers.setToolTip("Parallel requests to the VLM server.")
        for sp in (self.n, self.seed, self.workers):
            sp.setFixedWidth(92)
        self.move_chk = QCheckBox("Move overlay images to discarded/"); self.move_chk.setChecked(True)
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
        self.refresh_btn.clicked.connect(lambda: (self.log("manual refresh"), self.refresh()))
        self.auto_chk = QCheckBox("Auto refresh")
        self.auto_chk.setChecked(True)
        self.auto_chk.setToolTip("Reload automatically when vlm_labels.csv changes (e.g. a command-line run).")
        self._sig = self._csv_sig()
        self.poll = QTimer(self); self.poll.timeout.connect(self._poll); self.poll.start(2000)

        # ---- pages (tab bar hidden; the sidebar is the navigation) ----
        self.kept = Gallery("raw", self.show_detail)
        self.disc = Gallery("discarded", self.show_detail)
        self.stats = StatsTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.kept, "Verified (kept)")
        self.tabs.addTab(self.disc, "Discarded")
        self.tabs.addTab(self.stats, "Statistics")
        self.tabs.tabBar().hide()
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
        opts = QHBoxLayout(); opts.setSpacing(8)
        for label, w in (("Images", self.n), ("Seed", self.seed), ("Workers", self.workers)):
            lab = QLabel(label); lab.setObjectName("muted"); opts.addWidget(lab); opts.addWidget(w); opts.addSpacing(8)
        opts.addWidget(self.move_chk); opts.addStretch(); opts.addWidget(self.auto_chk); opts.addWidget(self.refresh_btn)
        prow = QHBoxLayout(); prow.addWidget(self.progress, 1); prow.addWidget(self.status)

        # ---- detail panel ----
        self.preview = QLabel("Select an image"); self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(340, 340)
        self.d = {k: QLabel("-") for k in ("Age", "Emotion", "Gender", "Overlay", "Source", "File")}
        for v in self.d.values():
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); v.setWordWrap(True)
        dform = QFormLayout(); dform.setVerticalSpacing(6)
        for k, v in self.d.items():
            lab = QLabel(k); lab.setObjectName("muted"); dform.addRow(lab, v)
        self.act = QPushButton(); self.act.clicked.connect(self.move_selected)
        self.open_btn = QPushButton("Open folder"); self.open_btn.clicked.connect(self.open_folder)
        self.hint = QLabel("Tip: Ctrl/Shift-click to select several images.")
        self.hint.setObjectName("muted"); self.hint.setWordWrap(True)
        dl = QVBoxLayout(); dl.setContentsMargins(16, 0, 0, 0); dl.setSpacing(10)
        dl.addWidget(self.preview); dl.addLayout(dform); dl.addWidget(self.act)
        dl.addWidget(self.open_btn); dl.addWidget(self.hint); dl.addStretch()
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
        self.reload()
        self.show_detail(self.kept)

    def _on_tab(self, i: int):
        self.nav.blockSignals(True); self.nav.setCurrentRow(i); self.nav.blockSignals(False)
        self.title.setText(self.TITLES[i])
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
    def reload(self):
        self._sig = self._csv_sig()
        self.rows = read_rows(VLM_CSV)
        self.kept.set_rows(self.rows); self.disc.set_rows(self.rows)
        total = max(0, sum(1 for _ in MASTER_CSV.open()) - 1) if MASTER_CSV.exists() else 0
        self.stats.update_data(self.rows, total)
        ok = [r for r in self.rows if r["valid"] == "1"]
        n_disc = sum(r["image"].startswith("discarded/") for r in ok)
        self.nav.item(0).setText(f"{self.NAV[0]}   ({len(ok) - n_disc:,})")
        self.nav.item(1).setText(f"{self.NAV[1]}   ({n_disc:,})")
        self.side_bar.setRange(0, max(total, 1)); self.side_bar.setValue(len(ok))
        self.side_count.setText(f"{len(ok):,} / {total:,}")
        self.statusBar().showMessage(f"{len(ok):,} labeled of {total:,} images  ·  {total - len(ok):,} remaining")
        self.log(f"Loaded labels: {len(ok):,} valid ({len(ok) - n_disc:,} kept, {n_disc:,} discarded), "
                 f"{len(self.rows) - len(ok):,} invalid, {total - len(ok):,} of {total:,} images unlabeled")

    def _csv_sig(self):
        try:
            st = VLM_CSV.stat()
            return (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return None

    def _poll(self):
        if self.auto_chk.isChecked() and self._csv_sig() != self._sig:
            self.log("labels file changed -> auto refresh")
            self.refresh()

    def _gallery(self):
        w = self.tabs.currentWidget()
        return w if isinstance(w, Gallery) else self.kept

    def refresh(self):
        """Reload from disk, keeping the current page and selection."""
        gallery = self._gallery()
        keep = {r["image"] for r in gallery.selected()}
        self._sig = self._csv_sig()
        self.reload()
        gallery.select_paths(keep)
        self.show_detail(gallery)

    # ---- labeling process ----
    def start_run(self):
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
        self.progress.setRange(0, self.target_n); self.progress.setValue(0)
        self.status.setText("Running…")
        self.set_running(True)
        self.log("Starting: python " + " ".join(args))
        self.proc.start(sys.executable, args)

    def set_running(self, running: bool):
        self.start.setEnabled(not running); self.stop.setEnabled(running)
        for w in (self.n, self.seed, self.workers, self.move_chk, self.act):
            w.setEnabled(not running)

    def on_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.log(line, "ERROR" if ("Traceback" in line or "Error" in line) else "PROC")
            if "/" in line and line.split("/")[0].strip().isdigit():
                self.progress.setValue(int(line.split("/")[0]))
                self.status.setText(f"{line.strip()} images")

    def stop_run(self):
        if self.proc:
            self.status.setText("Stopping…")
            self.log("Stop requested - killing the labeling process (finished results are kept)", "WARN")
            self.proc.kill()

    def on_finished(self, code, _status):
        self.set_running(False)
        self.status.setText("Done" if code == 0 else "Stopped (results so far are kept)")
        self.log(f"Labeling process finished (exit code {code})", "OK" if code == 0 else "WARN")
        if code == 0:
            self.progress.setValue(self.target_n)
        self.reload()
        self.show_detail(self.tabs.currentWidget())

    # ---- detail / actions ----
    def show_detail(self, gallery):
        if gallery is not self.tabs.currentWidget():
            return
        self.detail.setVisible(isinstance(gallery, Gallery))
        if not isinstance(gallery, Gallery):
            return
        sel = gallery.selected()
        in_disc = gallery is self.disc
        self.act.setObjectName("safe" if in_disc else "danger")
        self.act.setStyle(self.act.style())
        n = len(sel)
        self.act.setText(("↩  Restore to kept" if in_disc else "✕  Move to discarded") + (f" ({n})" if n else ""))
        self.act.setEnabled(n > 0 and not (self.proc and self.proc.state() != QProcess.ProcessState.NotRunning))
        self.open_btn.setEnabled(n > 0)
        if not sel:
            self.preview.setPixmap(QPixmap()); self.preview.setText("Select an image")
            for v in self.d.values():
                v.setText("-")
            return
        r = sel[0]
        pm = load_pixmap(DATA / r["image"], 520)
        if pm.isNull():
            self.preview.setText("(cannot load image)")
            self.log(f"cannot load image {r['image']}", "WARN")
        else:
            self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
        self.d["Age"].setText(r["vlm_age"]); self.d["Emotion"].setText(r["vlm_emotion"])
        self.d["Gender"].setText(r["vlm_gender"]); self.d["Overlay"].setText(r["vlm_overlay"])
        self.d["Source"].setText(r["image"].split("/")[1]); self.d["File"].setText(r["image"])
        if n > 1:
            self.d["File"].setText(f"{r['image']}  (+{n - 1} more selected)")

    def open_folder(self):
        sel = self._gallery().selected()
        if sel:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str((DATA / sel[0]["image"]).parent)))

    def move_selected(self):
        gallery = self._gallery()
        sel = gallery.selected()
        if not sel:
            return
        to_disc = gallery is self.kept
        mapping = {}
        missing = [r["image"] for r in sel if not (DATA / r["image"]).exists()]
        sel = [r for r in sel if r["image"] not in missing]
        for r in sel:
            old = r["image"]
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
            rewrite_paths(VLM_CSV, mapping); rewrite_paths(MASTER_CSV, mapping)
            self.log(f"updated vlm_labels.csv and master.csv ({len(mapping)} paths)", "OK")
        self.reload()
        self.show_detail(gallery)
        msg = f"Moved {len(mapping)} image(s) to {'discarded' if to_disc else 'kept'}"
        if missing:
            msg += f"  ·  {len(missing)} skipped (file was already moved; list refreshed)"
        self.statusBar().showMessage(msg, 6000)

    def closeEvent(self, e):
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill(); self.proc.waitForFinished(2000)
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
