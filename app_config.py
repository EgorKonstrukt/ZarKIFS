from __future__ import annotations
import json
from pathlib import Path
from PyQt5.QtCore import QByteArray
from PyQt5.QtWidgets import QApplication

CONFIG_DIR  = Path(__file__).parent / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"
THEMES_DIR  = CONFIG_DIR / "themes"

THEME_NATIVE_FUSION  = "fusion"
THEME_NATIVE_WINDOWS = "windows"
THEME_DARK_BLUE      = "dark_blue"

BUILTIN_THEMES = {
    THEME_NATIVE_FUSION:  "Fusion",
    THEME_NATIVE_WINDOWS: "Windows",
    THEME_DARK_BLUE:      "Dark Blue",
}

_DEFAULT = {
    "theme": THEME_DARK_BLUE,
    "windows": {},
}


def _ensure_dirs():
    CONFIG_DIR.mkdir(exist_ok=True)
    THEMES_DIR.mkdir(exist_ok=True)


def _load_raw() -> dict:
    try:
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            merged = dict(_DEFAULT)
            merged.update(data)
            return merged
    except Exception:
        pass
    return dict(_DEFAULT)


def _save_raw(data: dict):
    _ensure_dirs()
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_cfg: dict = _load_raw()
_theme_change_callbacks: list = []


def register_theme_change_callback(cb):
    if cb not in _theme_change_callbacks:
        _theme_change_callbacks.append(cb)


def unregister_theme_change_callback(cb):
    if cb in _theme_change_callbacks:
        _theme_change_callbacks.remove(cb)


def get(key: str, default=None):
    return _cfg.get(key, default)


def set(key: str, value):
    _cfg[key] = value
    _save_raw(_cfg)


def get_theme() -> str:
    return _cfg.get("theme", THEME_DARK_BLUE)


def set_theme(name: str):
    _cfg["theme"] = name
    _save_raw(_cfg)


def save_window_geometry(name: str, widget):
    wins = _cfg.setdefault("windows", {})
    geom = widget.geometry()
    wins[name] = {
        "x": geom.x(),
        "y": geom.y(),
        "w": geom.width(),
        "h": geom.height(),
    }
    _save_raw(_cfg)


def restore_window_geometry(name: str, widget, default_w=None, default_h=None):
    wins = _cfg.get("windows", {})
    entry = wins.get(name)
    if not entry:
        if default_w and default_h:
            widget.resize(default_w, default_h)
        return
    try:
        screen = QApplication.primaryScreen().availableGeometry()
        x = int(entry.get("x", 100))
        y = int(entry.get("y", 100))
        w = int(entry.get("w", default_w or 800))
        h = int(entry.get("h", default_h or 600))
        w = max(200, min(w, screen.width()))
        h = max(150, min(h, screen.height()))
        x = max(screen.left(), min(x, screen.right() - 100))
        y = max(screen.top(),  min(y, screen.bottom() - 50))
        widget.setGeometry(x, y, w, h)
    except Exception:
        if default_w and default_h:
            widget.resize(default_w, default_h)


def list_qss_themes() -> list[str]:
    _ensure_dirs()
    _write_builtin_qss()
    names = []
    for path in sorted(THEMES_DIR.glob("*.qss")):
        stem = path.stem
        names.append(stem)
    return names


def load_qss(name: str) -> str:
    path = THEMES_DIR / f"{name}.qss"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def apply_theme(app: QApplication, name: str):
    if name in (THEME_NATIVE_FUSION, THEME_NATIVE_WINDOWS):
        style_name = "Fusion" if name == THEME_NATIVE_FUSION else "Windows"
        app.setStyle(style_name)
        app.setStyleSheet("")
        app.setPalette(app.style().standardPalette())
        for widget in app.allWidgets():
            widget.setStyleSheet("")
            widget.setPalette(app.style().standardPalette())
            widget.update()
    else:
        app.setStyle("Fusion")
        qss = load_qss(name)
        app.setStyleSheet(qss)
        for widget in app.allWidgets():
            widget.update()
    for cb in list(_theme_change_callbacks):
        try:
            cb(name)
        except Exception:
            pass


def get_theme_display_name(name: str) -> str:
    if name in BUILTIN_THEMES:
        return BUILTIN_THEMES[name]
    return name.replace("_", " ").title()


def _write_builtin_qss():
    _write_dark_blue_qss()


def _write_dark_blue_qss():
    path = THEMES_DIR / f"{THEME_DARK_BLUE}.qss"
    qss = """
QWidget {
    background-color: #0f0f1e;
    color: #e0e0ff;
    font-family: "Segoe UI";
    font-size: 9pt;
}

QMainWindow, QDialog {
    background-color: #0f0f1e;
}

QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {
    background-color: #0f0f1e;
    border: none;
}

QScrollBar:vertical {
    background: #12122a;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #2d2d5e;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #7b68ee;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #12122a;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #2d2d5e;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #7b68ee;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}

QGroupBox {
    border: 1px solid #2d2d5e;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    background-color: #12122a;
    color: #e0e0ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    top: -2px;
    color: #7b68ee;
    font-weight: bold;
}

QPushButton {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    border-radius: 4px;
    padding: 4px 10px;
}
QPushButton:hover {
    background-color: #3a2d80;
    border-color: #7b68ee;
}
QPushButton:pressed {
    background-color: #5a4fc8;
}
QPushButton:disabled {
    color: #5050a0;
    background-color: #1a1a2e;
}
QPushButton:checked {
    background-color: #3a2d80;
    border-color: #7b68ee;
}

QLabel {
    background: transparent;
    color: #e0e0ff;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #12122a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    border-radius: 4px;
    padding: 2px 6px;
    selection-background-color: #7b68ee;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #7b68ee;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #2d2d5e;
    border: none;
    width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #7b68ee;
}

QComboBox {
    background-color: #12122a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    border-radius: 4px;
    padding: 2px 6px;
}
QComboBox:hover {
    border-color: #7b68ee;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    selection-background-color: #3a2d80;
}

QCheckBox {
    color: #e0e0ff;
    spacing: 6px;
    background: transparent;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2d2d5e;
    border-radius: 3px;
    background: #12122a;
}
QCheckBox::indicator:checked {
    background: #7b68ee;
    border-color: #7b68ee;
}
QCheckBox::indicator:hover {
    border-color: #7b68ee;
}

QRadioButton {
    color: #e0e0ff;
    spacing: 6px;
    background: transparent;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2d2d5e;
    border-radius: 7px;
    background: #12122a;
}
QRadioButton::indicator:checked {
    background: #7b68ee;
    border-color: #7b68ee;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #2d2d5e;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #7b68ee;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #9d8fff;
}
QSlider::sub-page:horizontal {
    background: #7b68ee;
    border-radius: 2px;
}

QTabWidget::pane {
    border: 1px solid #2d2d5e;
    background: #0f0f1e;
}
QTabBar::tab {
    background: #12122a;
    color: #8888cc;
    padding: 5px 10px;
    border: 1px solid #2d2d5e;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #1e1e3a;
    color: #7b68ee;
    border-bottom-color: #1e1e3a;
}
QTabBar::tab:hover:!selected {
    background: #1a1a2e;
    color: #aaaacc;
}

QListWidget {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    border-radius: 4px;
}
QListWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #12122a;
}
QListWidget::item:selected {
    background-color: #3a2d80;
    color: #e0e0ff;
}
QListWidget::item:hover {
    background-color: #252545;
}

QTreeWidget {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    border-radius: 4px;
}
QTreeWidget::item:selected {
    background-color: #3a2d80;
}
QTreeWidget::item:hover {
    background-color: #252545;
}
QHeaderView::section {
    background-color: #12122a;
    color: #8888cc;
    border: none;
    border-right: 1px solid #2d2d5e;
    padding: 4px 6px;
}

QMenu {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
}
QMenu::item {
    padding: 4px 20px;
}
QMenu::item:selected {
    background-color: #3a2d80;
}
QMenu::separator {
    height: 1px;
    background: #2d2d5e;
    margin: 2px 0;
}

QProgressBar {
    background-color: #2d2d5e;
    border-radius: 3px;
    border: none;
    text-align: center;
    color: #e0e0ff;
}
QProgressBar::chunk {
    background-color: #7b68ee;
    border-radius: 3px;
}

QToolBar {
    background-color: #12122a;
    border-bottom: 1px solid #2d2d5e;
    spacing: 4px;
    padding: 2px 4px;
}
QToolButton {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #2d2d5e;
    border-radius: 3px;
    padding: 3px 6px;
    margin: 1px;
}
QToolButton:hover {
    background-color: #3a2d80;
    border-color: #7b68ee;
}
QToolButton:pressed {
    background-color: #5a4fc8;
}
QToolButton:disabled {
    color: #5050a0;
    background-color: #1a1a2e;
}
QToolButton:checked {
    background-color: #3a2d80;
    border-color: #7b68ee;
}

QStatusBar {
    background-color: #12122a;
    color: #8888cc;
    border-top: 1px solid #2d2d5e;
}
QStatusBar QLabel {
    color: #8888cc;
    background: transparent;
}

QSplitter::handle {
    background: #2d2d5e;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}

QFrame[frameShape="4"],
QFrame[frameShape="5"] {
    color: #2d2d5e;
}

QToolTip {
    background-color: #1e1e3a;
    color: #e0e0ff;
    border: 1px solid #7b68ee;
    padding: 4px;
    border-radius: 4px;
}

QMessageBox {
    background-color: #1a1a2e;
}
QMessageBox QLabel {
    color: #e0e0ff;
}

QInputDialog {
    background-color: #1a1a2e;
}

QFileDialog {
    background-color: #1a1a2e;
}

QDialogButtonBox QPushButton {
    min-width: 70px;
}
"""
    try:
        _ensure_dirs()
        path.write_text(qss.strip(), encoding="utf-8")
    except Exception:
        pass