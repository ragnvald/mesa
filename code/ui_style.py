# -*- coding: utf-8 -*-
"""Shared Qt stylesheet for all MESA windows (same warm palette as mesa.py).

Deliberately lightweight: imports only PySide6 + stdlib, NO GIS stack. Helpers
that need only the look-and-feel (e.g. the Classification config UI) import from
here instead of asset_manage, so they don't drag fiona/geopandas/shapely into
their packaged exe. See learning.md "Stylesheet import dragged GIS into helpers".
"""

import os

from PySide6.QtCore import Qt

# =====================================================================
# Shared stylesheet (same warm palette as mesa.py)
# =====================================================================
ASSET_STYLESHEET = """
QMainWindow {
    background-color: #f3ecdf;
}
QWidget#CentralHost {
    background-color: #f3ecdf;
}
QWidget {
    color: #3f3528;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}
QTabWidget::pane {
    border: 1px solid #cbb791;
    border-top: none;
    background: #f3ecdf;
}
QTabBar {
    background: #e6dac2;
}
QTabBar::tab {
    background: #e6dac2;
    color: #5c4a2f;
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid #c6b089;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-weight: 500;
    font-size: 10pt;
}
QTabBar::tab:selected {
    background: #f8f3e9;
    color: #3f3528;
    font-weight: 600;
}
QTabBar::tab:!selected {
    margin-top: 2px;
}
QTabBar::tab:hover:!selected {
    background: #efe3cc;
}
QGroupBox {
    font-weight: 600;
    font-size: 10pt;
    background: #faf6ee;
    border: 1px solid #d5c3a4;
    border-radius: 8px;
    margin-top: 12px;
    padding: 18px 16px 14px 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 3px 10px;
    color: #715a36;
}
QPushButton {
    background-color: #f2e7d3;
    color: #4d4029;
    border: 1px solid #c7b18c;
    border-radius: 7px;
    padding: 6px 16px;
    font-weight: 500;
    min-width: 80px;
}
QPushButton:hover {
    background-color: #eadbbd;
    border-color: #b99763;
}
QPushButton:pressed {
    background-color: #ddc89f;
}
QPushButton:disabled {
    background-color: #eee5d7;
    color: #a28f71;
    border-color: #d4c6af;
}
QPushButton[role="primary"] {
    background-color: #d9bd7d;
    color: #3f3018;
    border: 1px solid #9b7c3d;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background-color: #e1c78d;
    border-color: #8c6d31;
}
QPushButton[role="primary"]:pressed {
    background-color: #cfb06f;
}
QPushButton[role="success"] {
    background-color: #e6ecd8;
    color: #34482a;
    border: 1px solid #9cad83;
}
QPushButton[role="success"]:hover {
    background-color: #edf2e3;
    border-color: #899b72;
}
QPushButton[role="success"]:pressed {
    background-color: #d9e3c7;
}
QLabel {
    background: transparent;
    color: #3f3528;
}
QLineEdit {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 10pt;
    selection-background-color: #d7bb7f;
    selection-color: #2f2517;
}
QLineEdit:focus {
    border-color: #b99763;
}
QPlainTextEdit, QTextEdit {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    padding: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 9pt;
    selection-background-color: #d7bb7f;
    selection-color: #2f2517;
}
QPlainTextEdit:focus, QTextEdit:focus {
    border-color: #b99763;
}
QComboBox {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 10pt;
    min-width: 60px;
}
QComboBox:hover { border-color: #b99763; }
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    selection-background-color: #d7bb7f;
    selection-color: #2f2517;
}
QTableWidget {
    background: #fffdf8;
    border: 1px solid #d9cab1;
    border-radius: 6px;
    gridline-color: #e2d5bf;
    font-size: 9pt;
    alternate-background-color: #f6efdf;
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background: #eee5d7;
    color: #5c4a2f;
    border: 1px solid #d5c3a4;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 9pt;
}
QScrollArea {
    border: none;
    background: transparent;
}
QSplitter::handle {
    background: #d5c3a4;
}
QCheckBox {
    spacing: 6px;
    font-size: 10pt;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked {
    background: #f5edd8;
    border: 1.5px solid #9a8260;
}
QCheckBox::indicator:unchecked:hover {
    border-color: #715a36;
    background: #efe3cc;
}
QCheckBox::indicator:checked {
    background: #715a36;
    border: 1.5px solid #513912;
}
QCheckBox::indicator:checked:hover {
    background: #8a6d3a;
}
QCheckBox::indicator:disabled {
    background: #e5dcc9;
    border-color: #c4b699;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
}
QRadioButton::indicator:unchecked {
    background: #f5edd8;
    border: 1.5px solid #9a8260;
}
QRadioButton::indicator:checked {
    background: #715a36;
    border: 1.5px solid #513912;
}
QProgressBar {
    background: #e7dbc4;
    border: 1px solid #d3c29f;
    border-radius: 5px;
    height: 14px;
    text-align: center;
    color: #4f4129;
    font-size: 8pt;
}
QProgressBar::chunk {
    background: #b79b67;
    border-radius: 4px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #d0bc97;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #b99763;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #d0bc97;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #b99763;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QToolTip {
    background: #3f3528;
    color: #faf6ee;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 9pt;
}
"""


def _generate_indicator_stylesheet() -> str:
    """Generate QSS with checkmark/radio-dot images for checked indicators."""
    import tempfile as _tmpmod
    from PySide6.QtGui import QPainter, QPen, QPixmap, QColor

    _dir = os.path.join(_tmpmod.gettempdir(), "mesa_indicators")
    os.makedirs(_dir, exist_ok=True)
    _check = os.path.join(_dir, "check.png")
    _dot = os.path.join(_dir, "dot.png")

    # White checkmark on transparent
    pm = QPixmap(16, 16)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#ffffff"), 2.2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.drawLine(3, 8, 6, 12)
    p.drawLine(6, 12, 13, 4)
    p.end()
    pm.save(_check, "PNG")

    # White dot on transparent
    pm2 = QPixmap(16, 16)
    pm2.fill(QColor(0, 0, 0, 0))
    p2 = QPainter(pm2)
    p2.setRenderHint(QPainter.Antialiasing)
    p2.setPen(Qt.NoPen)
    p2.setBrush(QColor("#ffffff"))
    p2.drawEllipse(4, 4, 8, 8)
    p2.end()
    pm2.save(_dot, "PNG")

    cu = _check.replace("\\", "/")
    du = _dot.replace("\\", "/")
    return f'''
QCheckBox::indicator:checked {{ image: url("{cu}"); }}
QRadioButton::indicator:checked {{ image: url("{du}"); }}
'''


def apply_shared_stylesheet(app) -> None:
    """Apply ASSET_STYLESHEET plus generated indicator images to a QApplication."""
    css = ASSET_STYLESHEET + _generate_indicator_stylesheet()
    app.setStyleSheet(css)
