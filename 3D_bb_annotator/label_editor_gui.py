import sys
import os
import re
import json
import copy
import random

# Allow sibling-module imports (pose_estimation_pipeline, auto_bbox_dialog)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from PIL import Image as _PIL_Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit, QLabel,
    QFileDialog, QMessageBox, QSplitter, QGroupBox, QGridLayout,
    QDialog, QComboBox, QDialogButtonBox, QFormLayout, QDoubleSpinBox,
    QStyle, QCheckBox, QSizePolicy, QScrollArea, QFrame, QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

pv.set_plot_theme("dark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH     = os.path.expanduser("~/.3d_label_editor.json")
MAX_RECENT      = 5
YOLO_MODEL_PATH = "/home/tumwfml-ubunt6/3D_object_detection/instance_segmentation/YOLOv11-Seg/train/weights/best.pt"

PALLET_CLASSES = {"pallet", "pallet truck", "pallet_truck"}

BBOX_LINES = [
    [0,1], [1,2], [2,3], [3,0],
    [4,5], [5,6], [6,7], [7,4],
    [0,4], [1,5], [2,6], [3,7],
]

CLASS_COLORS = {
    "forklift":           (0.0, 1.0, 0.0),
    "pallet_truck":       (1.0, 0.5, 0.0),
    "pallet":             (0.0, 1.0, 1.0),
    "small_load_carrier": (1.0, 0.0, 1.0),
    "stillage":           (1.0, 1.0, 0.0),
    "person":             (1.0, 1.0, 0.0),
}
DEFAULT_COLOR = (0.0, 0.5, 1.0)

FIELD_KEYS = [
    ("centroid",   "x"), ("centroid",   "y"), ("centroid",   "z"),
    ("dimensions", "length"), ("dimensions", "width"), ("dimensions", "height"),
    ("rotations",  "x"), ("rotations",  "y"), ("rotations",  "z"),
]
FIELD_LABELS = [
    "centroid x", "centroid y", "centroid z",
    "dim length",  "dim width",  "dim height",
    "rot x",       "rot y",      "rot z",
]

DROPDOWN_OPTIONS = ["pallet", "KLT small", "KLT large", "stillage", "forklift", "pallet truck", "Custom..."]

DROPDOWN_NAME_MAP = {
    "pallet":       "pallet",
    "KLT small":    "klt",
    "KLT large":    "klt",
    "stillage":     "stillage",
    "forklift":     "forklift",
    "pallet truck": "pallet truck",
}

DEFAULT_DIMENSIONS = {
    "pallet":       {"length": 1.200, "width": 0.800, "height": 0.144},
    "KLT small":    {"length": 0.400, "width": 0.300, "height": 0.147},
    "KLT large":    {"length": 0.600, "width": 0.400, "height": 0.147},
    "stillage":     {"length": 1.200, "width": 0.800, "height": 0.970},
    "forklift":     {"length": 2.800, "width": 1.300, "height": 2.150},
    "pallet truck": {"length": 1.800, "width": 0.550, "height": 1.200},
}

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-size: 12px;
}
QPushButton {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555555;
    padding: 5px 10px;
    border-radius: 4px;
    min-height: 22px;
}
QPushButton:hover  { background-color: #4c4c4c; border-color: #888; }
QPushButton:pressed { background-color: #2a2a2a; }
QListWidget {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    outline: none;
}
QListWidget::item:selected { background-color: #094771; color: #ffffff; }
QListWidget::item:hover    { background-color: #2a2d2e; }
QLineEdit {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 4px;
}
QLineEdit:focus { border-color: #007acc; }
QGroupBox {
    color: #9cdcfe;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
QLabel { color: #d4d4d4; }
QScrollArea { background-color: #1e1e1e; border: none; }
QScrollBar:vertical {
    background: #252526; width: 10px; border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #555; border-radius: 5px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #888; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background-color: #007acc; color: #ffffff; font-weight: bold; }
QSplitter::handle { background-color: #3c3c3c; width: 2px; }
"""

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"recent_projects": []}


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def push_recent_project(cfg: dict, name: str, pcd_dir: str, labels_dir: str,
                        rgb_dir: str = "", depth_dir: str = "", camera_params_dir: str = ""):
    # Remove any existing entry with the same folders (regardless of name)
    recent = [p for p in cfg.get("recent_projects", [])
              if not (p.get("pcd_dir") == pcd_dir and p.get("labels_dir") == labels_dir)]
    recent.insert(0, {
        "name":              name,
        "pcd_dir":           pcd_dir,
        "labels_dir":        labels_dir,
        "rgb_dir":           rgb_dir,
        "depth_dir":         depth_dir,
        "camera_params_dir": camera_params_dir,
    })
    cfg["recent_projects"] = recent[:MAX_RECENT]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _rotation_y(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def build_pv_bbox(obj) -> tuple[pv.PolyData, tuple]:
    cx, cy, cz = obj["centroid"]["x"], obj["centroid"]["y"], obj["centroid"]["z"]
    center = np.array([cx, cy, cz])
    L = obj["dimensions"]["length"]
    W = obj["dimensions"]["width"]
    H = obj["dimensions"]["height"]

    local_corners = np.array([
        [-L/2, -H/2, -W/2], [ L/2, -H/2, -W/2],
        [ L/2,  H/2, -W/2], [-L/2,  H/2, -W/2],
        [-L/2, -H/2,  W/2], [ L/2, -H/2,  W/2],
        [ L/2,  H/2,  W/2], [-L/2,  H/2,  W/2],
    ])

    R = _rotation_y(np.deg2rad(obj["rotations"]["y"]))
    bbox_points = (R @ local_corners.T).T + center

    lines_conn = []
    for a, b in BBOX_LINES:
        lines_conn.extend([2, a, b])

    mesh = pv.PolyData()
    mesh.points = bbox_points
    mesh.lines = np.array(lines_conn)

    color = CLASS_COLORS.get(obj["name"].replace(" ", "_"), DEFAULT_COLOR)
    return mesh, color



_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def _find_rgb_image(rgb_dir: str, frame_id: str) -> str | None:
    # Fast path: exact name match with common suffixes
    for suffix in ("", "_color"):
        for ext in _IMAGE_EXTS:
            path = os.path.join(rgb_dir, f"{frame_id}{suffix}{ext}")
            if os.path.exists(path):
                return path

    # Fallback: match by numeric ID only (handles different naming conventions,
    # e.g. pointcloud_000 <-> rgb_000, frame_0019 <-> image_19)
    numbers = re.findall(r"\d+", frame_id)
    if not numbers:
        return None
    target = int(numbers[-1])  # use last number, compare as int (ignores zero-padding)

    for fname in sorted(os.listdir(rgb_dir)):
        if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
            continue
        file_nums = re.findall(r"\d+", fname)
        if file_nums and int(file_nums[-1]) == target:
            return os.path.join(rgb_dir, fname)

    return None


# Spinbox config: (min, max, step, decimals) per field section
_SPIN_CFG = {
    "centroid":   (-50.0, 50.0,  0.01, 3),
    "dimensions": ( 0.001, 10.0, 0.01, 3),
    "rotations":  (-360.0, 360.0, 1.0, 1),
}

# ---------------------------------------------------------------------------
# Project editor dialog (used for both New and Edit)
# ---------------------------------------------------------------------------
class ProjectEditorDialog(QDialog):

    def __init__(self, name: str = "", pcd_dir: str = "", labels_dir: str = "",
                 rgb_dir: str = "", depth_dir: str = "", camera_params_dir: str = "",
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Folders")
        self.setModal(True)
        self.setMinimumWidth(540)
        self._name              = name
        self._pcd_dir           = pcd_dir
        self._labels_dir        = labels_dir
        self._rgb_dir           = rgb_dir
        self._depth_dir         = depth_dir
        self._camera_params_dir = camera_params_dir
        self._build_ui()

    def _build_ui(self):
        ROW_H = 30

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Name field — full width
        name_form = QFormLayout()
        name_form.setSpacing(6)
        self._name_edit = QLineEdit(self._name)
        self._name_edit.setPlaceholderText("e.g. Azure Kinect — warehouse run 1")
        self._name_edit.setFixedHeight(ROW_H)
        name_form.addRow("Project name:", self._name_edit)
        layout.addLayout(name_form)

        # Grid: col 0 = path display (stretches), col 1 = buttons (fixed, aligned)
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setColumnStretch(0, 1)

        def _path_edit(placeholder, text):
            e = QLineEdit(text)
            e.setPlaceholderText(placeholder)
            e.setReadOnly(True)
            e.setFixedHeight(ROW_H)
            return e

        def _browse_btn(text, slot):
            b = QPushButton(text)
            b.setFixedHeight(ROW_H)
            b.clicked.connect(slot)
            return b

        self._pcd_edit = _path_edit("Point cloud folder (.pcd / .ply)…", self._pcd_dir)
        pcd_btn = _browse_btn("Browse PCD folder…", self._browse_pcd)
        grid.addWidget(self._pcd_edit, 0, 0)
        grid.addWidget(pcd_btn,        0, 1)

        self._lbl_edit = _path_edit("3D labels folder (.json)…", self._labels_dir)
        lbl_btn = _browse_btn("Browse Labels folder…", self._browse_labels)
        grid.addWidget(self._lbl_edit, 1, 0)
        grid.addWidget(lbl_btn,        1, 1)

        self._rgb_edit = _path_edit("RGB images folder (.jpg / .png)…", self._rgb_dir)
        rgb_btn = _browse_btn("Browse RGB folder…", self._browse_rgb)
        grid.addWidget(self._rgb_edit, 2, 0)
        grid.addWidget(rgb_btn,        2, 1)

        self._depth_edit = _path_edit("Depth maps folder (.npy)…", self._depth_dir)
        depth_btn = _browse_btn("Browse Depth folder…", self._browse_depth)
        grid.addWidget(self._depth_edit, 3, 0)
        grid.addWidget(depth_btn,        3, 1)

        self._params_edit = _path_edit("Camera params folder (.npz / .json)…", self._camera_params_dir)
        params_btn = _browse_btn("Browse Params folder…", self._browse_params)
        grid.addWidget(self._params_edit, 4, 0)
        grid.addWidget(params_btn,        4, 1)

        # Cancel + Confirm in col 1 — same column as Browse buttons → same width
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        cancel_btn  = QPushButton("Cancel")
        confirm_btn = QPushButton("Confirm")
        cancel_btn.setFixedHeight(ROW_H)
        confirm_btn.setFixedHeight(ROW_H)
        cancel_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        confirm_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOkButton))
        cancel_btn.clicked.connect(self.reject)
        confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(confirm_btn)
        grid.addLayout(btn_row, 5, 1)

        layout.addLayout(grid)

    def _browse_pcd(self):
        path = QFileDialog.getExistingDirectory(self, "Select PCD / PLY folder",
                                                self._pcd_edit.text() or os.path.expanduser("~"))
        if path:
            self._pcd_edit.setText(path)

    def _browse_labels(self):
        path = QFileDialog.getExistingDirectory(self, "Select 3D Labels folder",
                                                self._lbl_edit.text() or os.path.expanduser("~"))
        if path:
            self._lbl_edit.setText(path)

    def _browse_rgb(self):
        path = QFileDialog.getExistingDirectory(self, "Select RGB Images folder",
                                                self._rgb_edit.text() or os.path.expanduser("~"))
        if path:
            self._rgb_edit.setText(path)

    def _browse_depth(self):
        path = QFileDialog.getExistingDirectory(self, "Select Depth Maps folder",
                                                self._depth_edit.text() or os.path.expanduser("~"))
        if path:
            self._depth_edit.setText(path)

    def _browse_params(self):
        path = QFileDialog.getExistingDirectory(self, "Select Camera Parameters folder",
                                                self._params_edit.text() or os.path.expanduser("~"))
        if path:
            self._params_edit.setText(path)

    def _on_confirm(self):
        if not self._pcd_edit.text().strip():
            QMessageBox.warning(self, "Missing folder", "Please select a PCD / PLY folder.")
            return
        self.accept()

    def get_project(self) -> dict:
        return {
            "name":              self._name_edit.text().strip(),
            "pcd_dir":           self._pcd_edit.text().strip(),
            "labels_dir":        self._lbl_edit.text().strip(),
            "rgb_dir":           self._rgb_edit.text().strip(),
            "depth_dir":         self._depth_edit.text().strip(),
            "camera_params_dir": self._params_edit.text().strip(),
        }


# ---------------------------------------------------------------------------
# Load Project dialog
# ---------------------------------------------------------------------------
class LoadProjectDialog(QDialog):

    def __init__(self, recent_projects: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Project")
        self.setModal(True)
        self.setMinimumWidth(540)
        self._recent = list(recent_projects)
        self._selected: dict | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Recent projects:"))
        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(lambda _: self._on_accept())
        layout.addWidget(self._list)

        # Action buttons row — equal fixed width so they all match
        action_row = QHBoxLayout()
        new_btn = QPushButton("+ New")
        new_btn.clicked.connect(self._on_new)
        self._edit_btn   = QPushButton("Edit")
        self._edit_btn.clicked.connect(self._on_edit)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        for b in (new_btn, self._edit_btn, self._delete_btn):
            b.setFixedWidth(80)
            action_row.addWidget(b)
        action_row.addStretch()
        layout.addLayout(action_row)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._detail)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Load")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate list only after _detail exists
        self._refresh_list()
        self._update_button_state()

    def _refresh_list(self, keep_row: int = 0):
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._recent:
            name = p.get("name", "").strip()
            label = name if name else os.path.basename(p["pcd_dir"])
            self._list.addItem(f"  {label}")
        if not self._recent:
            self._list.addItem("No recent projects — click '+ New' to add one")
        self._list.blockSignals(False)
        row = min(keep_row, max(0, len(self._recent) - 1))
        self._list.setCurrentRow(row)
        self._on_row_changed(row)
        self._update_button_state()

    def _update_button_state(self):
        has_selection = 0 <= self._list.currentRow() < len(self._recent)
        self._edit_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self._recent):
            p = self._recent[row]
            lbl = p.get("labels_dir") or "—"
            self._detail.setText(f"PCD:    {p['pcd_dir']}\nLabels: {lbl}")
        else:
            self._detail.setText("")
        self._update_button_state()

    def _on_new(self):
        dlg = ProjectEditorDialog(parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted:
            return
        entry = dlg.get_project()
        self._recent = [p for p in self._recent if p != entry]
        self._recent.insert(0, entry)
        self._recent = self._recent[:MAX_RECENT]
        self._refresh_list(keep_row=0)

    def _on_edit(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._recent):
            return
        p = self._recent[row]
        dlg = ProjectEditorDialog(name=p.get("name", ""),
                                  pcd_dir=p.get("pcd_dir", ""),
                                  labels_dir=p.get("labels_dir", ""),
                                  rgb_dir=p.get("rgb_dir", ""),
                                  depth_dir=p.get("depth_dir", ""),
                                  camera_params_dir=p.get("camera_params_dir", ""),
                                  parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted:
            return
        self._recent[row] = dlg.get_project()
        self._refresh_list(keep_row=row)

    def _on_delete(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._recent):
            return
        name = os.path.basename(self._recent[row]["pcd_dir"])
        reply = QMessageBox.question(self, "Delete project",
                                     f"Remove '{name}' from recent projects?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._recent.pop(row)
        self._refresh_list(keep_row=max(0, row - 1))

    def _on_accept(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._recent):
            return
        self._selected = self._recent[row]
        self.accept()

    def get_project(self) -> dict | None:
        return self._selected

    def get_updated_recent(self) -> list:
        return self._recent


# ---------------------------------------------------------------------------
# Per-object editor widget
# ---------------------------------------------------------------------------
class ObjectFieldWidget(QGroupBox):

    def __init__(self, index: int, obj: dict, on_change=None, parent=None):
        super().__init__(f"[{index}]  {obj['name']}", parent)
        self._obj = obj
        self.fields: dict[tuple, QDoubleSpinBox] = {}

        CELL_H = 24

        grid = QGridLayout()
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(3)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)

        def _spin(section, key):
            lo, hi, step, dec = _SPIN_CFG[section]
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setFixedHeight(CELL_H)
            s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            s.setValue(float(obj[section][key]))
            if on_change:
                s.valueChanged.connect(lambda _: on_change())
            return s

        def _btn(text, slot):
            b = QPushButton(text)
            b.setFixedHeight(CELL_H)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet("min-height: 0; padding: 0;")
            b.clicked.connect(slot)
            if on_change:
                b.clicked.connect(on_change)
            return b

        for row_idx, ((section, key), label_text) in enumerate(zip(FIELD_KEYS, FIELD_LABELS)):
            spin = _spin(section, key)
            self.fields[(section, key)] = spin

            grid.addWidget(QLabel(label_text), row_idx, 0)
            grid.addWidget(spin,               row_idx, 1)

            if (section, key) == ("dimensions", "length"):
                grid.addWidget(_btn("L↔W", self.swap_lw), row_idx, 2)
                grid.addWidget(_btn("L↔H", self.swap_lh), row_idx, 3)
            elif (section, key) == ("dimensions", "width"):
                grid.addWidget(_btn("W↔H", self.swap_wh), row_idx, 2)

        self.setLayout(grid)

    def get_values(self) -> dict | None:
        result = copy.deepcopy(self._obj)
        for (section, key), spin in self.fields.items():
            result[section][key] = spin.value()
        return result

    def clear_highlights(self):
        pass

    def swap_wh(self):
        tmp = self.fields[("dimensions", "width")].value()
        self.fields[("dimensions", "width")].setValue(self.fields[("dimensions", "height")].value())
        self.fields[("dimensions", "height")].setValue(tmp)

    def swap_lw(self):
        tmp = self.fields[("dimensions", "length")].value()
        self.fields[("dimensions", "length")].setValue(self.fields[("dimensions", "width")].value())
        self.fields[("dimensions", "width")].setValue(tmp)

    def swap_lh(self):
        tmp = self.fields[("dimensions", "length")].value()
        self.fields[("dimensions", "length")].setValue(self.fields[("dimensions", "height")].value())
        self.fields[("dimensions", "height")].setValue(tmp)


# ---------------------------------------------------------------------------
# Add-object dialog
# ---------------------------------------------------------------------------
class AddObjectDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Object")
        self.setModal(True)
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        form = QFormLayout()
        form.setSpacing(6)

        self._combo = QComboBox()
        self._combo.addItems(DROPDOWN_OPTIONS)
        self._combo.currentTextChanged.connect(self._on_type_changed)
        form.addRow("Object type:", self._combo)

        self._custom_name = QLineEdit()
        self._custom_name.setPlaceholderText("Enter custom name…")
        self._custom_name.hide()
        form.addRow("Custom name:", self._custom_name)

        self._f_length = QLineEdit("0.000")
        self._f_width  = QLineEdit("0.000")
        self._f_height = QLineEdit("0.000")
        form.addRow("Length (m):", self._f_length)
        form.addRow("Width  (m):", self._f_width)
        form.addRow("Height (m):", self._f_height)

        self._f_cx = QLineEdit("0.000")
        self._f_cy = QLineEdit("0.000")
        self._f_cz = QLineEdit("0.000")
        form.addRow("Centroid X:", self._f_cx)
        form.addRow("Centroid Y:", self._f_cy)
        form.addRow("Centroid Z:", self._f_cz)

        self._f_rx = QLineEdit("0.000")
        self._f_ry = QLineEdit("0.000")
        self._f_rz = QLineEdit("0.000")
        form.addRow("Rotation X:", self._f_rx)
        form.addRow("Rotation Y:", self._f_ry)
        form.addRow("Rotation Z:", self._f_rz)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add")
        buttons.accepted.connect(self._on_add)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_type_changed(DROPDOWN_OPTIONS[0])

    def _on_type_changed(self, text: str):
        is_custom = (text == "Custom...")
        self._custom_name.setVisible(is_custom)
        if is_custom:
            self._f_length.clear()
            self._f_width.clear()
            self._f_height.clear()
        else:
            dims = DEFAULT_DIMENSIONS.get(text, {})
            self._f_length.setText(f"{dims.get('length', 0.0):.3f}")
            self._f_width.setText(f"{dims.get('width',  0.0):.3f}")
            self._f_height.setText(f"{dims.get('height', 0.0):.3f}")

    def _on_add(self):
        fields = [
            self._f_length, self._f_width, self._f_height,
            self._f_cx, self._f_cy, self._f_cz,
            self._f_rx, self._f_ry, self._f_rz,
        ]
        invalid = False
        for edit in fields:
            try:
                float(edit.text().strip())
                edit.setStyleSheet("")
            except ValueError:
                edit.setStyleSheet("background-color: #7a2020; color: #ffcccc;")
                invalid = True

        if self._combo.currentText() == "Custom..." and not self._custom_name.text().strip():
            self._custom_name.setStyleSheet("background-color: #7a2020; color: #ffcccc;")
            invalid = True

        if invalid:
            QMessageBox.warning(self, "Invalid values", "Please fix the highlighted fields.")
            return
        self.accept()

    def get_object(self) -> dict:
        label = self._combo.currentText()
        if label == "Custom...":
            name = self._custom_name.text().strip()
        else:
            name = DROPDOWN_NAME_MAP.get(label, label)

        return {
            "name": name,
            "centroid": {
                "x": float(self._f_cx.text()),
                "y": float(self._f_cy.text()),
                "z": float(self._f_cz.text()),
            },
            "dimensions": {
                "length": float(self._f_length.text()),
                "width":  float(self._f_width.text()),
                "height": float(self._f_height.text()),
            },
            "rotations": {
                "x": float(self._f_rx.text()),
                "y": float(self._f_ry.text()),
                "z": float(self._f_rz.text()),
            },
        }


# ---------------------------------------------------------------------------
# Background worker for the batch processing loop in _on_auto_bbox
# ---------------------------------------------------------------------------
class _BatchWorker(QThread):
    frame_started   = pyqtSignal(int, str)   # (frame_idx, frame_id)
    object_progress = pyqtSignal(int, int)   # (det_idx, total)
    batch_done      = pyqtSignal(list, int)  # (current_frame_objects, total_saved)

    def __init__(self, run_fn):
        super().__init__()
        self._run_fn = run_fn

    def run(self):
        self._run_fn(self)


# ---------------------------------------------------------------------------
# Dual progress dialog used by _on_auto_bbox
# ---------------------------------------------------------------------------
class _AutoBBoxProgressDialog(QDialog):
    def __init__(self, n_frames: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autonomous 3D BB Generation")
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setFixedWidth(440)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._frame_label = QLabel("Starting…")
        layout.addWidget(self._frame_label)
        self._frame_bar = QProgressBar()
        self._frame_bar.setRange(0, n_frames)
        self._frame_bar.setValue(0)
        layout.addWidget(self._frame_bar)

        self._obj_label = QLabel("Objects: —")
        layout.addWidget(self._obj_label)
        self._obj_bar = QProgressBar()
        self._obj_bar.setRange(0, 1)
        self._obj_bar.setValue(0)
        layout.addWidget(self._obj_bar)

    def set_frame(self, idx: int, frame_id: str):
        total = self._frame_bar.maximum()
        self._frame_label.setText(f"Frame {idx + 1} / {total}:  {frame_id}")
        self._frame_bar.setValue(idx + 1)
        self._obj_bar.setValue(0)
        self._obj_bar.setRange(0, 1)
        self._obj_label.setText("Objects: detecting…")
        QApplication.processEvents()

    def set_object(self, idx: int, total: int):
        self._obj_bar.setRange(0, total)
        self._obj_bar.setValue(idx + 1)
        self._obj_label.setText(f"Object {idx + 1} / {total}")
        QApplication.processEvents()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class LabelEditorWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.pcd_dir:            str | None = None
        self.labels_dir:         str | None = None
        self.rgb_dir:            str | None = None
        self.depth_dir:          str | None = None
        self.camera_params_dir:  str | None = None
        self.current_frame_id:   str | None = None
        self.current_label_data: dict | None = None
        self.current_objects:    list = []
        self._selected_obj_idx:  int = -1
        self._active_widget:     ObjectFieldWidget | None = None
        self._dirty = False
        self._cfg = load_config()
        self._orig_pixmap = None

        self.setWindowTitle("3D Label Editor")
        self.resize(1500, 900)
        self._build_ui()
        self._auto_load_last_project()

    def _auto_load_last_project(self):
        recent = self._cfg.get("recent_projects", [])
        if recent:
            self._apply_project(recent[0])

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([220, 800, 360])
        self.setCentralWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        load_btn = QPushButton("Load Project…")
        load_btn.clicked.connect(self._on_load_project)
        layout.addWidget(load_btn)

        self._project_label = QLabel("No project loaded")
        self._project_label.setWordWrap(True)
        self._project_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._project_label)

        layout.addWidget(QLabel("PCD / PLY files:"))
        self._select_all_cb = QCheckBox("Select all")
        self._select_all_cb.setTristate(True)
        self._select_all_cb.clicked.connect(self._on_select_all_clicked)
        layout.addWidget(self._select_all_cb)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        self.file_list.itemChanged.connect(self._on_item_check_changed)
        layout.addWidget(self.file_list)

        self._auto_btn = QPushButton("Autonomous 3D BB Generation")
        self._auto_btn.clicked.connect(self._on_auto_bbox)
        self._auto_btn.setEnabled(False)
        self._auto_btn.setToolTip("Check one or more files in the list to enable")
        layout.addWidget(self._auto_btn)

        return w

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plotter = QtInteractor(parent=w)
        self.plotter.set_background("#1a1a2e", top="#0d0d1a")
        layout.addWidget(self.plotter, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 4, 4, 4)

        prev_btn = QPushButton("←")
        prev_btn.clicked.connect(self._on_prev_frame)
        btn_row.addWidget(prev_btn)

        rand_btn = QPushButton("Random Frame")
        rand_btn.clicked.connect(self._on_random_frame)
        btn_row.addWidget(rand_btn)

        next_btn = QPushButton("→")
        next_btn.clicked.connect(self._on_next_frame)
        btn_row.addWidget(next_btn)

        layout.addLayout(btn_row)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w_layout = QVBoxLayout(w)
        w_layout.setContentsMargins(8, 8, 8, 8)
        w_layout.setSpacing(0)

        self._right_splitter = QSplitter(Qt.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.setHandleWidth(2)

        # ── Panel 1: Objects ───────────────────────────────────────────────
        self._obj_panel = QWidget()
        obj_layout = QVBoxLayout(self._obj_panel)
        obj_layout.setContentsMargins(0, 0, 0, 0)
        obj_layout.setSpacing(4)

        obj_layout.addWidget(QLabel("Objects:"))
        self.object_list = QListWidget()
        self.object_list.currentRowChanged.connect(self._on_object_selected)
        obj_layout.addWidget(self.object_list)

        obj_btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Object")
        add_btn.clicked.connect(self._on_add_object)
        obj_btn_row.addWidget(add_btn)
        remove_btn = QPushButton("− Remove Object")
        remove_btn.clicked.connect(self._on_remove_object)
        obj_btn_row.addWidget(remove_btn)
        obj_layout.addLayout(obj_btn_row)

        self._right_splitter.addWidget(self._obj_panel)

        # ── Panel 2: BB dimensions ─────────────────────────────────────────
        self._fields_panel = QWidget()
        fields_outer = QVBoxLayout(self._fields_panel)
        fields_outer.setContentsMargins(0, 0, 0, 0)
        fields_outer.setSpacing(0)

        self._fields_container = QWidget()
        self._fields_layout = QVBoxLayout(self._fields_container)
        self._fields_layout.setAlignment(Qt.AlignTop)
        self._fields_layout.setSpacing(8)

        fields_scroll = QScrollArea()
        fields_scroll.setWidget(self._fields_container)
        fields_scroll.setWidgetResizable(True)
        fields_scroll.setFrameShape(QFrame.NoFrame)
        fields_outer.addWidget(fields_scroll)

        self._right_splitter.addWidget(self._fields_panel)

        # ── Panel 3: Save buttons + RGB image ──────────────────────────────
        self._bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(self._bottom_panel)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(4)

        bottom_row = QHBoxLayout()
        swap_all_btn = QPushButton("Swap All W↔H")
        swap_all_btn.clicked.connect(self._on_swap_all_pallets)
        bottom_row.addWidget(swap_all_btn)
        save_btn = QPushButton("Save JSON")
        save_btn.clicked.connect(self._on_save)
        bottom_row.addWidget(save_btn)
        bottom_layout.addLayout(bottom_row)

        bottom_layout.addWidget(QLabel("RGB image:"))
        self._image_label = QLabel("No image loaded")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setStyleSheet(
            "background-color: #252526; border: 1px solid #3c3c3c; color: #555;"
        )
        bottom_layout.addWidget(self._image_label, stretch=1)

        self._right_splitter.addWidget(self._bottom_panel)

        w_layout.addWidget(self._right_splitter)
        return w

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------
    def _on_load_project(self):
        recent = self._cfg.get("recent_projects", [])
        dlg = LoadProjectDialog(recent, parent=self)
        dlg.setStyleSheet(self.styleSheet())
        result = dlg.exec_()
        # Always persist edits/deletions, even if the user cancels
        self._cfg["recent_projects"] = dlg.get_updated_recent()
        save_config(self._cfg)
        if result != QDialog.Accepted:
            return
        project = dlg.get_project()
        if not project:
            return
        self._apply_project(project)

    def _apply_project(self, project: dict):
        pcd_dir    = project.get("pcd_dir", "")
        labels_dir = project.get("labels_dir", "")
        if not os.path.isdir(pcd_dir):
            QMessageBox.warning(self, "Folder not found", f"PCD folder not found:\n{pcd_dir}")
            return
        self.pcd_dir    = pcd_dir
        self.labels_dir = labels_dir if os.path.isdir(labels_dir) else None
        rgb_dir         = project.get("rgb_dir", "")
        self.rgb_dir    = rgb_dir if os.path.isdir(rgb_dir) else None
        depth_dir       = project.get("depth_dir", "")
        self.depth_dir  = depth_dir if os.path.isdir(depth_dir) else None
        cam_dir         = project.get("camera_params_dir", "")
        self.camera_params_dir = cam_dir if os.path.isdir(cam_dir) else None

        project_name = project.get("name", "").strip()
        pcd_name  = os.path.basename(pcd_dir)
        lbl_name  = os.path.basename(labels_dir) if self.labels_dir else "—"
        rgb_name  = os.path.basename(rgb_dir) if self.rgb_dir else "—"
        dep_name  = os.path.basename(depth_dir) if self.depth_dir else "—"
        cam_name  = os.path.basename(cam_dir) if self.camera_params_dir else "—"
        header = f"{project_name}\n" if project_name else ""
        self._project_label.setText(
            f"{header}PCD: {pcd_name}\nLabels: {lbl_name}\nRGB: {rgb_name}\n"
            f"Depth: {dep_name}\nParams: {cam_name}"
        )

        # Save this project as most-recent (preserving name)
        push_recent_project(self._cfg, project.get("name", ""), pcd_dir, labels_dir,
                            rgb_dir, depth_dir, cam_dir)
        save_config(self._cfg)

        self._populate_file_list()
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _populate_file_list(self):
        if not self.pcd_dir:
            return
        files = sorted(f for f in os.listdir(self.pcd_dir)
                       if f.endswith(".pcd") or f.endswith(".ply"))
        self._select_all_cb.blockSignals(True)
        self._select_all_cb.setCheckState(Qt.Unchecked)
        self._select_all_cb.blockSignals(False)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for f in files:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self._update_status(f"Found {len(files)} point cloud files")

    def _on_file_selected(self, item: QListWidgetItem | None):
        if item is None:
            return
        self._load_frame(os.path.splitext(item.text())[0])

    def _load_frame(self, frame_id: str):
        if self._dirty and not self._check_unsaved():
            return

        if not self.pcd_dir:
            self._update_status("Please load a project first")
            return

        pcd_path = next(
            (os.path.join(self.pcd_dir, f"{frame_id}{ext}") for ext in (".pcd", ".ply")
             if os.path.exists(os.path.join(self.pcd_dir, f"{frame_id}{ext}"))),
            None,
        )
        if pcd_path is None:
            QMessageBox.warning(self, "Missing file", f"No .pcd or .ply found for:\n{frame_id}")
            return

        if self.labels_dir:
            label_path = os.path.join(self.labels_dir, f"{frame_id}.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    self.current_label_data = json.load(f)
                self.current_objects = copy.deepcopy(self.current_label_data.get("objects", []))
            else:
                self._update_status(f"No JSON for {frame_id} — showing point cloud only")
                self.current_label_data = None
                self.current_objects = []
        else:
            self.current_label_data = None
            self.current_objects = []

        self.current_frame_id = frame_id
        self._dirty = False
        self._rebuild_object_panel()
        self.setWindowTitle(f"3D Label Editor — {frame_id}")
        self._sync_file_list_selection(frame_id)
        self._render_scene()
        self._load_rgb_image(frame_id)
        self._update_status(f"{frame_id} — {len(self.current_objects)} object(s)")

    def _rebuild_object_panel(self):
        self.object_list.blockSignals(True)
        self.object_list.clear()
        for i, obj in enumerate(self.current_objects):
            self.object_list.addItem(f"[{i}]  {obj['name']}")
        self.object_list.blockSignals(False)

        self._clear_active_widget()
        self._selected_obj_idx = -1

        if self.current_objects:
            self.object_list.setCurrentRow(0)

    def _clear_active_widget(self):
        if self._active_widget is not None:
            self._fields_layout.removeWidget(self._active_widget)
            self._active_widget.setParent(None)
            self._active_widget.deleteLater()
            self._active_widget = None

    def _sync_active(self) -> str | None:
        if self._active_widget is None or self._selected_obj_idx < 0:
            return None
        result = self._active_widget.get_values()
        if result is None:
            return f"Object [{self._selected_obj_idx}]: invalid numeric value in one or more fields"
        self.current_objects[self._selected_obj_idx] = result
        return None

    def _on_object_selected(self, row: int):
        if row < 0 or row >= len(self.current_objects):
            return
        self._sync_active()
        self._selected_obj_idx = row
        self._clear_active_widget()
        widget = ObjectFieldWidget(row, self.current_objects[row], on_change=self._on_regenerate)
        self._fields_layout.addWidget(widget)
        self._active_widget = widget
        self._render_scene()

    def _on_regenerate(self):
        if self.current_frame_id is None:
            return
        err = self._sync_active()
        if err:
            QMessageBox.warning(self, "Invalid field values", err)
            return
        self._dirty = True
        if self._active_widget:
            self._active_widget.clear_highlights()
        self._render_scene()

    def _render_scene(self, reset_camera: bool = False):
        cam_pos = self.plotter.camera_position if not reset_camera else None

        self.plotter.clear()
        if self.current_frame_id is None or not self.pcd_dir:
            return

        pcd_path = next(
            (os.path.join(self.pcd_dir, f"{self.current_frame_id}{ext}") for ext in (".pcd", ".ply")
             if os.path.exists(os.path.join(self.pcd_dir, f"{self.current_frame_id}{ext}"))),
            None,
        )
        if pcd_path is None:
            return
        try:
            pcd_o3d = o3d.io.read_point_cloud(pcd_path)
            pts = np.asarray(pcd_o3d.points)
            if pts.shape[0] > 0:
                cloud = pv.PolyData(pts)
                cols = np.asarray(pcd_o3d.colors)
                if cols.shape[0] == pts.shape[0]:
                    cloud["RGB"] = (cols * 255).astype(np.uint8)
                    self.plotter.add_mesh(cloud, scalars="RGB", rgb=True,
                                         point_size=2, style="points",
                                         render_points_as_spheres=False)
                else:
                    self.plotter.add_mesh(cloud, color="white", point_size=2, style="points")
        except Exception as e:
            self._update_status(f"Error loading point cloud: {e}")
            return

        for i, obj in enumerate(self.current_objects):
            mesh, color = build_pv_bbox(obj)
            selected = (i == self._selected_obj_idx)
            self.plotter.add_mesh(mesh, color=color,
                                  line_width=4 if selected else 2,
                                  opacity=1.0 if selected else 0.3)

        self.plotter.add_axes()
        if reset_camera or cam_pos is None:
            self.plotter.reset_camera()
        else:
            self.plotter.camera_position = cam_pos

    def _on_add_object(self):
        if not self.current_frame_id:
            QMessageBox.warning(self, "No frame loaded",
                                "Please load a point cloud before adding objects.")
            return
        if not self.labels_dir:
            QMessageBox.warning(self, "No labels folder",
                                "Please load a project with a labels folder first.")
            return

        dlg = AddObjectDialog(parent=self)
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted:
            return

        new_obj = dlg.get_object()
        self._sync_active()
        self.current_objects.append(new_obj)

        new_idx = len(self.current_objects) - 1
        self.object_list.blockSignals(True)
        self.object_list.addItem(f"[{new_idx}]  {new_obj['name']}")
        self.object_list.blockSignals(False)

        self._dirty = True
        self.object_list.setCurrentRow(new_idx)
        self._update_status(f"Added '{new_obj['name']}' — {len(self.current_objects)} object(s)")

    def _on_remove_object(self):
        row = self.object_list.currentRow()
        if row < 0:
            return
        self._sync_active()
        removed_name = self.current_objects[row]["name"]
        del self.current_objects[row]
        self.object_list.blockSignals(True)
        self.object_list.takeItem(row)
        for i in range(self.object_list.count()):
            self.object_list.item(i).setText(f"[{i}]  {self.current_objects[i]['name']}")
        self.object_list.blockSignals(False)
        # Reset stale widget/index BEFORE setCurrentRow so _sync_active is a no-op
        self._active_widget = None
        self._selected_obj_idx = -1
        self._dirty = True
        new_row = min(row, self.object_list.count() - 1)
        self.object_list.setCurrentRow(new_row)
        self._render_scene()
        self._update_status(f"Removed '{removed_name}' — {len(self.current_objects)} object(s)")

    def _on_save(self):
        if self.current_frame_id is None or not self.labels_dir:
            self._update_status("No frame loaded or labels folder not set")
            return
        err = self._sync_active()
        if err:
            QMessageBox.warning(self, "Invalid field values", err)
            return

        is_new = self.current_label_data is None
        if is_new:
            pcd_filename = f"{self.current_frame_id}.pcd"
            pcd_path = os.path.join(self.pcd_dir, pcd_filename) if self.pcd_dir else ""
            save_data = {
                "folder":   os.path.basename(self.labels_dir),
                "filename": f"{self.current_frame_id}.ply",
                "path":     pcd_path,
                "objects":  self.current_objects,
            }
        else:
            save_data = copy.deepcopy(self.current_label_data)
            save_data["objects"] = self.current_objects

        out_path = os.path.join(self.labels_dir, f"{self.current_frame_id}.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent="\t")

        self.current_label_data = save_data
        self._dirty = False
        if is_new:
            self._update_status(f"New label file created: {self.current_frame_id}.json")
        else:
            self._update_status(f"Saved {self.current_frame_id}.json successfully")

    def _on_swap_all_pallets(self):
        self._sync_active()
        for obj in self.current_objects:
            obj["dimensions"]["width"], obj["dimensions"]["height"] = \
                obj["dimensions"]["height"], obj["dimensions"]["width"]
        if 0 <= self._selected_obj_idx < len(self.current_objects):
            self._clear_active_widget()
            widget = ObjectFieldWidget(self._selected_obj_idx,
                                       self.current_objects[self._selected_obj_idx],
                                       on_change=self._on_regenerate)
            self._fields_layout.addWidget(widget)
            self._active_widget = widget
        self._dirty = True
        self._render_scene()
        self._update_status(f"Swapped W↔H for all {len(self.current_objects)} object(s)")

    def _on_random_frame(self):
        if not self.pcd_dir:
            self._update_status("Please load a project first")
            return
        available = [os.path.splitext(f)[0] for f in os.listdir(self.pcd_dir)
                     if f.endswith(".pcd") or f.endswith(".ply")]
        if available:
            self._load_frame(random.choice(available))

    def _check_unsaved(self) -> bool:
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            f"Save changes to '{self.current_frame_id}' before leaving?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._on_save()
        return True

    def _on_prev_frame(self):
        if not self._check_unsaved():
            return
        row = self.file_list.currentRow()
        if row > 0:
            self.file_list.setCurrentRow(row - 1)

    def _on_next_frame(self):
        if not self._check_unsaved():
            return
        row = self.file_list.currentRow()
        if row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_rgb_image(self, frame_id: str):
        self._orig_pixmap = None
        if not self.rgb_dir:
            self._image_label.setText("No RGB folder set")
            self._image_label.setPixmap(QPixmap())
            return
        path = _find_rgb_image(self.rgb_dir, frame_id)
        if path is None:
            self._image_label.setText(f"Image not found for {frame_id}")
            self._image_label.setPixmap(QPixmap())
            return
        self._orig_pixmap = QPixmap(path)
        self._image_label.setText("")
        self._refresh_image_pixmap()

    def _refresh_image_pixmap(self):
        if self._orig_pixmap is None:
            return
        w = self._image_label.width()
        h = self._image_label.height()
        if w > 0 and h > 0:
            self._image_label.setPixmap(
                self._orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def showEvent(self, event):
        super().showEvent(event)
        h = self._right_splitter.height()
        self._right_splitter.setSizes([h * 3 // 10, h * 4 // 10, h * 3 // 10])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_image_pixmap()

    def _on_select_all_clicked(self, checked: bool):
        new_state = Qt.Checked if checked else Qt.Unchecked
        self._select_all_cb.blockSignals(True)
        self._select_all_cb.setCheckState(new_state)
        self._select_all_cb.blockSignals(False)
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(new_state)
        self.file_list.blockSignals(False)
        self._auto_btn.setEnabled(checked and self.file_list.count() > 0)

    def _on_item_check_changed(self, _item):
        count = self.file_list.count()
        if count == 0:
            return
        n_checked = sum(
            1 for i in range(count)
            if self.file_list.item(i).checkState() == Qt.Checked
        )
        self._select_all_cb.blockSignals(True)
        if n_checked == 0:
            self._select_all_cb.setCheckState(Qt.Unchecked)
        elif n_checked == count:
            self._select_all_cb.setCheckState(Qt.Checked)
        else:
            self._select_all_cb.setCheckState(Qt.PartiallyChecked)
        self._select_all_cb.blockSignals(False)
        self._auto_btn.setEnabled(n_checked > 0)

    def get_checked_frame_ids(self) -> list:
        return [
            os.path.splitext(self.file_list.item(i).text())[0]
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]

    def _sync_file_list_selection(self, frame_id: str):
        for ext in (".pcd", ".ply"):
            target = f"{frame_id}{ext}"
            for i in range(self.file_list.count()):
                if self.file_list.item(i).text() == target:
                    self.file_list.blockSignals(True)
                    self.file_list.setCurrentRow(i)
                    self.file_list.blockSignals(False)
                    return

    # ------------------------------------------------------------------
    # Autonomous 3D BB generation
    # ------------------------------------------------------------------
    def _on_auto_bbox(self):
        # 1. Collect target frames (checked, or fall back to current frame)
        frame_ids = self.get_checked_frame_ids()
        if not frame_ids:
            if self.current_frame_id:
                frame_ids = [self.current_frame_id]
            else:
                QMessageBox.warning(self, "No frame", "Please load a frame first.")
                return

        # 2. Validate required folders
        missing = []
        if not self.rgb_dir:
            missing.append("RGB images folder")
        if not self.depth_dir:
            missing.append("Depth maps folder")
        if not self.camera_params_dir:
            missing.append("Camera parameters folder")
        if not self.labels_dir:
            missing.append("Labels folder")
        if missing:
            QMessageBox.warning(self, "Missing folders",
                                "Please configure these project folders first:\n• " +
                                "\n• ".join(missing))
            return

        from pose_estimation_pipeline import (
            find_depth_file, find_camera_params_file,
            load_intrinsics, apply_hist_depth_filter, estimate_3d_pose, make_label_object,
        )
        from auto_bbox_dialog import AutoBBoxValidationDialog

        # 3. Process first frame with validation dialog
        first_id = frame_ids[0]
        rgb_path    = _find_rgb_image(self.rgb_dir, first_id) if self.rgb_dir else None
        depth_path  = find_depth_file(self.depth_dir, first_id)
        params_path = find_camera_params_file(self.camera_params_dir, first_id)

        if not rgb_path:
            QMessageBox.warning(self, "Missing file", f"No RGB image found for: {first_id}")
            return
        if not depth_path:
            QMessageBox.warning(self, "Missing file", f"No depth file found for: {first_id}")
            return
        if not params_path:
            QMessageBox.warning(self, "Missing file",
                                f"No camera parameters file found for: {first_id}")
            return

        try:
            rgb_img   = np.array(_PIL_Image.open(rgb_path).convert("RGB"))
            depth_img = np.load(depth_path)
            fx, fy, cx, cy = load_intrinsics(params_path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        # Detect Z-axis convention from point cloud (Z-backward = all-negative Z, e.g. ZED2 PLY)
        # Use PyVista (already imported) instead of Open3D — handles more PLY variants.
        z_backward = False
        if self.pcd_dir:
            for _ext in (".pcd", ".ply"):
                _pc_path = os.path.join(self.pcd_dir, f"{first_id}{_ext}")
                if os.path.exists(_pc_path):
                    try:
                        _pts = pv.read(_pc_path).points
                        if len(_pts) > 0 and _pts[:, 2].max() < 0:
                            z_backward = True
                    except Exception:
                        pass
                    break

        # Show validation dialog (segmentation runs inside the dialog)
        dlg = AutoBBoxValidationDialog(
            rgb_img, depth_img, fx, fy, cx, cy,
            default_yolo_path=YOLO_MODEL_PATH, parent=self
        )
        dlg.setStyleSheet(self.styleSheet())
        if dlg.exec_() != QDialog.Accepted or dlg.result is None:
            return

        detections = dlg._detections

        hdf_params     = dlg._hdf_params
        conf_threshold = dlg.conf_threshold

        # ── helper: run pipeline on one detection, return label dict or None ──
        def _process_detection(box, det_mask, dep_full, fx_i, fy_i, cx_i, cy_i, cls_name):
            x1, y1, x2, y2 = box.astype(int)
            dep_crop = dep_full[y1:y2, x1:x2].astype(float)
            if dep_crop.size == 0:
                return None
            raw_mask = (det_mask.astype(np.uint8)[y1:y2, x1:x2]
                        if det_mask is not None
                        else np.ones(dep_crop.shape, dtype=np.uint8))
            # Normalize to mm
            valid_px = dep_crop[dep_crop > 0]
            if valid_px.size > 0 and valid_px.max() <= 100:
                dep_crop = dep_crop * 1000.0
            dep_masked = np.where(raw_mask, dep_crop, 0)
            filtered, *_ = apply_hist_depth_filter(
                dep_masked,
                resolution=hdf_params["resolution"],
                max_height_percent=hdf_params["max_height_percent"],
            )
            mask_crop = (filtered > 0).astype(np.uint8)
            try:
                _, center, dims, yaw_deg, _ = estimate_3d_pose(
                    filtered, mask_crop, x1, y1, fx_i, fy_i, cx_i, cy_i
                )
            except Exception:
                return None
            if z_backward:
                center[1] = -center[1]
                center[2] = -center[2]
                yaw_deg   = -yaw_deg
            return make_label_object(cls_name, center, dims, yaw_deg)

        def _get_class_names(frame_detections):
            if "class_name" in frame_detections.data:
                return [str(n) for n in frame_detections.data["class_name"]]
            return ["object"] * len(frame_detections)

        def _get_masks(frame_detections):
            if frame_detections.mask is not None:
                return list(frame_detections.mask)
            return [None] * len(frame_detections)

        # ── batch worker runs all frame processing in a background thread ────
        save_fn          = self._save_auto_result_to_file
        current_fid_cap  = self.current_frame_id
        rgb_dir_cap      = self.rgb_dir
        depth_dir_cap    = self.depth_dir
        params_dir_cap   = self.camera_params_dir

        def _run_batch(worker):
            current_frame_objs: list = []
            total = 0
            for frame_idx, fid in enumerate(frame_ids):
                worker.frame_started.emit(frame_idx, fid)
                try:
                    if fid == first_id:
                        rgb_f, dep_f = rgb_img, depth_img
                        fx_f, fy_f, cx_f, cy_f = fx, fy, cx, cy
                        frame_det = detections
                    else:
                        r_path = _find_rgb_image(rgb_dir_cap, fid)
                        d_path = find_depth_file(depth_dir_cap, fid)
                        p_path = find_camera_params_file(params_dir_cap, fid)
                        if not r_path or not d_path or not p_path:
                            continue
                        rgb_f  = np.array(_PIL_Image.open(r_path).convert("RGB"))
                        dep_f  = np.load(d_path)
                        fx_f, fy_f, cx_f, cy_f = load_intrinsics(p_path)
                        frame_det = dlg.run_on_image(rgb_f)

                    if frame_det.confidence is not None:
                        frame_det = frame_det[frame_det.confidence >= conf_threshold]
                    if len(frame_det) == 0:
                        continue

                    cls_names  = _get_class_names(frame_det)
                    masks      = _get_masks(frame_det)
                    frame_objs: list = []

                    for det_idx, (box, det_mask, cls_name) in enumerate(
                            zip(frame_det.xyxy, masks, cls_names)):
                        worker.object_progress.emit(det_idx, len(frame_det))
                        obj = _process_detection(
                            box, det_mask, dep_f, fx_f, fy_f, cx_f, cy_f, cls_name)
                        if obj is not None:
                            frame_objs.append(obj)

                    if fid == current_fid_cap:
                        current_frame_objs.extend(frame_objs)
                    else:
                        for obj in frame_objs:
                            save_fn(fid, obj)

                    total += len(frame_objs)
                except Exception:
                    continue

            worker.batch_done.emit(current_frame_objs, total)

        self._batch_prog    = _AutoBBoxProgressDialog(len(frame_ids), parent=self)
        self._batch_n_frames = len(frame_ids)
        self._batch_prog.show()

        self._batch_worker = _BatchWorker(_run_batch)
        self._batch_worker.frame_started.connect(self._batch_prog.set_frame)
        self._batch_worker.object_progress.connect(self._batch_prog.set_object)
        self._batch_worker.batch_done.connect(self._finish_batch)
        self._batch_worker.start()

    def _finish_batch(self, current_frame_objs: list, total_saved: int):
        """Slot called on the main thread when _BatchWorker finishes."""
        self._batch_prog.close()
        self._sync_active()
        self.object_list.blockSignals(True)
        for obj in current_frame_objs:
            self.current_objects.append(obj)
            self.object_list.addItem(f"[{len(self.current_objects) - 1}]  {obj['name']}")
        self.object_list.blockSignals(False)
        if current_frame_objs:
            self._dirty = True
        self._render_scene()
        self._update_status(
            f"Done — {total_saved} object(s) saved across {self._batch_n_frames} frame(s)"
        )

    def _save_auto_result_to_file(self, frame_id: str, obj: dict):
        """Append obj to the label JSON for frame_id (creates file if needed)."""
        if not self.labels_dir:
            return
        os.makedirs(self.labels_dir, exist_ok=True)
        label_path = os.path.join(self.labels_dir, f"{frame_id}.json")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                data = json.load(f)
            data.setdefault("objects", []).append(obj)
        else:
            data = {
                "folder":   os.path.basename(self.labels_dir),
                "filename": f"{frame_id}.pcd",
                "path":     os.path.join(self.pcd_dir or "", f"{frame_id}.pcd"),
                "objects":  [obj],
            }
        with open(label_path, "w") as f:
            json.dump(data, f, indent="\t")

    def _update_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    win = LabelEditorWindow()
    win.show()
    sys.exit(app.exec_())
