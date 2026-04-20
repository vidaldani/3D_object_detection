import sys
import os
import json
import copy
import random

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit, QLabel,
    QFileDialog, QMessageBox, QSplitter, QGroupBox, QGridLayout,
)
from PyQt5.QtCore import Qt

pv.set_plot_theme("dark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
# Geometry helpers (pure numpy — no Open3D dependency in math)
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


# ---------------------------------------------------------------------------
# Per-object editor widget
# ---------------------------------------------------------------------------
class ObjectFieldWidget(QGroupBox):

    def __init__(self, index: int, obj: dict, on_change=None, parent=None):
        super().__init__(f"[{index}]  {obj['name']}", parent)
        self._obj = obj
        self._on_change = on_change
        self.fields: dict[tuple, QLineEdit] = {}

        CELL_W = 100  # uniform width for edits and buttons
        CELL_H = 24   # uniform height for edits and buttons

        grid = QGridLayout()
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(3)
        grid.setColumnMinimumWidth(0, 72)
        grid.setColumnMinimumWidth(1, CELL_W)
        grid.setColumnMinimumWidth(2, CELL_W)
        grid.setColumnMinimumWidth(3, CELL_W)

        def _btn(text, slot):
            b = QPushButton(text)
            b.setFixedSize(CELL_W, CELL_H)
            b.setStyleSheet("min-height: 0; padding: 0;")
            b.clicked.connect(slot)
            if on_change:
                b.clicked.connect(on_change)
            return b

        for row_idx, ((section, key), label_text) in enumerate(zip(FIELD_KEYS, FIELD_LABELS)):
            lbl = QLabel(label_text)
            val = obj[section][key]
            edit = QLineEdit(f"{val:.3f}" if isinstance(val, float) else str(val))
            edit.setFixedSize(CELL_W, CELL_H)
            if on_change:
                edit.editingFinished.connect(on_change)
            self.fields[(section, key)] = edit

            grid.addWidget(lbl,  row_idx, 0)
            grid.addWidget(edit, row_idx, 1)

            if (section, key) == ("dimensions", "length"):
                grid.addWidget(_btn("L↔W", self.swap_lw), row_idx, 2)
                grid.addWidget(_btn("L↔H", self.swap_lh), row_idx, 3)
            elif (section, key) == ("dimensions", "width"):
                grid.addWidget(_btn("W↔H", self.swap_wh), row_idx, 2)

        self.setLayout(grid)

    def get_values(self) -> dict | None:
        result = copy.deepcopy(self._obj)
        invalid = False
        for (section, key), edit in self.fields.items():
            try:
                result[section][key] = float(edit.text().strip())
                edit.setStyleSheet("")
            except ValueError:
                edit.setStyleSheet("background-color: #7a2020; color: #ffcccc;")
                invalid = True
        return None if invalid else result

    def clear_highlights(self):
        for edit in self.fields.values():
            edit.setStyleSheet("")

    def swap_wh(self):
        w_edit = self.fields[("dimensions", "width")]
        h_edit = self.fields[("dimensions", "height")]
        tmp = w_edit.text()
        w_edit.setText(h_edit.text())
        h_edit.setText(tmp)

    def swap_lw(self):
        l_edit = self.fields[("dimensions", "length")]
        w_edit = self.fields[("dimensions", "width")]
        tmp = l_edit.text()
        l_edit.setText(w_edit.text())
        w_edit.setText(tmp)

    def swap_lh(self):
        l_edit = self.fields[("dimensions", "length")]
        h_edit = self.fields[("dimensions", "height")]
        tmp = l_edit.text()
        l_edit.setText(h_edit.text())
        h_edit.setText(tmp)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class LabelEditorWindow(QMainWindow):

    DEFAULT_PCD_DIR    = "/home/tumwfml-ubunt6/3D_object_detection/rd_test_dataset/Azure_Kinect_dataset/pcd_colored"
    DEFAULT_LABELS_DIR = "/home/tumwfml-ubunt6/3D_object_detection/rd_test_dataset/Azure_Kinect_dataset/3d_labels"

    def __init__(self):
        super().__init__()
        self.pcd_dir: str | None = None
        self.labels_dir: str | None = None
        self.current_frame_id: str | None = None
        self.current_label_data: dict | None = None
        self.current_objects: list = []
        self._selected_obj_idx: int = -1
        self._active_widget: ObjectFieldWidget | None = None
        self._dirty = False

        self.setWindowTitle("3D Label Editor")
        self.resize(1500, 900)
        self._build_ui()
        self._preload_defaults()

    def _preload_defaults(self):
        if os.path.isdir(self.DEFAULT_PCD_DIR):
            self.pcd_dir = self.DEFAULT_PCD_DIR
            self._pcd_dir_label.setText(f"PCD: {os.path.basename(self.pcd_dir)}")
        if os.path.isdir(self.DEFAULT_LABELS_DIR):
            self.labels_dir = self.DEFAULT_LABELS_DIR
            self._lbl_dir_label.setText(f"Labels: {os.path.basename(self.labels_dir)}")
        if self.pcd_dir and self.labels_dir:
            self._populate_file_list()
            if self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)

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

        btn_pcd = QPushButton("Browse PCD folder")
        btn_pcd.clicked.connect(self._on_browse_pcd)
        layout.addWidget(btn_pcd)

        self._pcd_dir_label = QLabel("PCD folder: not set")
        self._pcd_dir_label.setWordWrap(True)
        layout.addWidget(self._pcd_dir_label)

        btn_lbl = QPushButton("Browse Labels folder")
        btn_lbl.clicked.connect(self._on_browse_labels)
        layout.addWidget(btn_lbl)

        self._lbl_dir_label = QLabel("Labels folder: not set")
        self._lbl_dir_label.setWordWrap(True)
        layout.addWidget(self._lbl_dir_label)

        layout.addWidget(QLabel("PCD files:"))
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        layout.addWidget(self.file_list)

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
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Objects:"))
        self.object_list = QListWidget()
        self.object_list.setFixedHeight(110)
        self.object_list.currentRowChanged.connect(self._on_object_selected)
        layout.addWidget(self.object_list)

        self._fields_container = QWidget()
        self._fields_layout = QVBoxLayout(self._fields_container)
        self._fields_layout.setAlignment(Qt.AlignTop)
        self._fields_layout.setSpacing(8)
        layout.addWidget(self._fields_container)

        bottom_row = QHBoxLayout()
        swap_all_btn = QPushButton("Swap All W↔H")
        swap_all_btn.clicked.connect(self._on_swap_all_pallets)
        bottom_row.addWidget(swap_all_btn)
        save_btn = QPushButton("Save JSON")
        save_btn.clicked.connect(self._on_save)
        bottom_row.addWidget(save_btn)
        layout.addLayout(bottom_row)
        layout.addStretch(1)

        return w

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_browse_pcd(self):
        path = QFileDialog.getExistingDirectory(self, "Select PCD folder")
        if path:
            self.pcd_dir = path
            self._pcd_dir_label.setText(f"PCD: {os.path.basename(path)}")
            self._populate_file_list()

    def _on_browse_labels(self):
        path = QFileDialog.getExistingDirectory(self, "Select Labels folder")
        if path:
            self.labels_dir = path
            self._lbl_dir_label.setText(f"Labels: {os.path.basename(path)}")
            self._populate_file_list()

    def _populate_file_list(self):
        if not self.pcd_dir:
            return
        files = sorted(f for f in os.listdir(self.pcd_dir) if f.endswith(".pcd"))
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(f)
        self.file_list.blockSignals(False)
        self._update_status(f"Found {len(files)} PCD files")

    def _on_file_selected(self, item: QListWidgetItem | None):
        if item is None:
            return
        self._load_frame(item.text().replace(".pcd", ""))

    def _load_frame(self, frame_id: str):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                f"'{self.current_frame_id}' has unsaved changes.\nLoad '{frame_id}' anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        if not self.pcd_dir:
            self._update_status("Please select a PCD folder first")
            return

        pcd_path = os.path.join(self.pcd_dir, f"{frame_id}.pcd")
        if not os.path.exists(pcd_path):
            QMessageBox.warning(self, "Missing PCD", f"PCD file not found:\n{pcd_path}")
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
            self.object_list.setCurrentRow(0)   # triggers _on_object_selected

    def _clear_active_widget(self):
        if self._active_widget is not None:
            self._fields_layout.removeWidget(self._active_widget)
            self._active_widget.setParent(None)
            self._active_widget.deleteLater()
            self._active_widget = None

    def _sync_active(self) -> str | None:
        """Flush active widget values into current_objects. Returns error string or None."""
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
        # Sync old widget before switching
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

        pcd_path = os.path.join(self.pcd_dir, f"{self.current_frame_id}.pcd")
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
            self._update_status(f"Error loading PCD: {e}")
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

    def _on_save(self):
        if self.current_frame_id is None or not self.labels_dir:
            self._update_status("No frame loaded or labels folder not set")
            return
        err = self._sync_active()
        if err:
            QMessageBox.warning(self, "Invalid field values", err)
            return

        save_data = copy.deepcopy(self.current_label_data) if self.current_label_data else {}
        save_data["objects"] = self.current_objects

        out_path = os.path.join(self.labels_dir, f"{self.current_frame_id}.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent="\t")

        self.current_label_data = save_data
        self._dirty = False
        self._update_status(f"Saved {self.current_frame_id}.json successfully")

    def _on_swap_all_pallets(self):
        self._sync_active()
        for obj in self.current_objects:
            obj["dimensions"]["width"], obj["dimensions"]["height"] = \
                obj["dimensions"]["height"], obj["dimensions"]["width"]
        # Refresh active widget to reflect the swap
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
            self._update_status("Please select a PCD folder first")
            return
        available = [f.replace(".pcd", "") for f in os.listdir(self.pcd_dir) if f.endswith(".pcd")]
        if available:
            self._load_frame(random.choice(available))

    def _check_unsaved(self) -> bool:
        """Prompt to save if dirty. Returns True if navigation should proceed, False to cancel."""
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

    def _sync_file_list_selection(self, frame_id: str):
        target = f"{frame_id}.pcd"
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == target:
                self.file_list.blockSignals(True)
                self.file_list.setCurrentRow(i)
                self.file_list.blockSignals(False)
                break

    def _mark_dirty(self):
        self._dirty = True

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
