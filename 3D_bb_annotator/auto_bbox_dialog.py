"""
auto_bbox_dialog.py

3-step validation dialog for the Autonomous 3D Bounding Box Generation feature.
"""

import os
import warnings
import numpy as np

# Suppress noisy deprecation warnings from PyTorch internals / upstream libraries
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*", category=FutureWarning)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QStackedWidget, QWidget, QSizePolicy, QComboBox, QCheckBox,
    QDoubleSpinBox, QRadioButton, QGroupBox, QFileDialog, QMessageBox,
    QLineEdit, QFrame, QApplication,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from pose_estimation_pipeline import (
    apply_hist_depth_filter,
    estimate_3d_pose,
    make_label_object,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_qpixmap(rgb_array: np.ndarray) -> QPixmap:
    """Convert an H×W×3 uint8 numpy array to QPixmap."""
    arr = np.ascontiguousarray(rgb_array, dtype=np.uint8)
    h, w, ch = arr.shape
    if h == 0 or w == 0:
        return QPixmap()
    img = QImage(arr.tobytes(), w, h, w * ch, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def _annotate_detections(rgb_img: np.ndarray, detections, selected_idx: int = -1) -> np.ndarray:
    """Draw detection boxes and masks on a copy of rgb_img."""
    import cv2
    out = rgb_img.copy()
    if detections is None or len(detections) == 0:
        return out

    masks = detections.mask if detections.mask is not None else [None] * len(detections)
    for i, (box, mask) in enumerate(zip(detections.xyxy, masks)):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 200, 255) if i == selected_idx else (100, 100, 100)
        thickness = 3 if i == selected_idx else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if mask is not None:
            overlay = out.copy()
            overlay[mask > 0] = np.clip(
                overlay[mask > 0].astype(int) + np.array(color) * 0.4, 0, 255
            ).astype(np.uint8)
            out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)

    return out


# ---------------------------------------------------------------------------
# Background worker — keeps the Qt event loop alive during slow inference
# ---------------------------------------------------------------------------

class _SegWorker(QThread):
    done  = pyqtSignal(object)  # sv.Detections
    error = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            self.done.emit(self._func())
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# AutoBBoxValidationDialog
# ---------------------------------------------------------------------------

class AutoBBoxValidationDialog(QDialog):
    """
    3-step modal dialog for validating autonomous 3D bounding box generation.

    Pages:
      0 — Method selection + segmentation run
      1 — Depth histogram with HDF bounds
      2 — Top-down X-Z projection + aligned bounding box
    """

    def __init__(self, rgb_img: np.ndarray, depth_img: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float,
                 default_yolo_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autonomous 3D BB — Validation")
        self.setModal(True)
        self.setMinimumSize(700, 600)

        self._rgb   = rgb_img
        self._depth = depth_img
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._default_yolo_path = default_yolo_path
        self._class_name = "object"

        self._selected_idx   = 0
        self._filtered_depth = None
        self._mask_crop      = None
        self._x1 = self._y1 = 0
        self._hdf_params     = {"resolution": 1, "max_height_percent": 5, "ignore_background": False}
        self._depth_crop_cache   = None
        self._depth_masked_cache = None

        # Segmentation state
        self._detections        = None
        self._segmentation_run  = False
        self._yolo_model        = None
        self._seg_worker        = None   # QThread for background inference

        self.conf_threshold: float = 0.5
        self._active_detections = None

        self.result: dict | None = None

        self._build_ui()
        self._init_page0()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._page0 = QWidget()
        self._stack.addWidget(self._page0)
        self._build_page0()

        self._page1 = QWidget()
        self._stack.addWidget(self._page1)
        self._build_page1()

        self._page2 = QWidget()
        self._stack.addWidget(self._page2)
        self._build_page2()

        # Navigation buttons (shared)
        nav = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        self._back_btn   = QPushButton("← Back")
        self._back_btn.clicked.connect(self._on_back)
        self._back_btn.setVisible(False)
        self._next_btn   = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        self._accept_btn = QPushButton("Accept")
        self._accept_btn.clicked.connect(self._on_accept)
        self._accept_btn.setVisible(False)

        nav.addWidget(self._cancel_btn)
        nav.addStretch()
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._accept_btn)
        root.addLayout(nav)

    # ---- page 0: method selection + segmentation run ----
    def _build_page0(self):
        layout = QVBoxLayout(self._page0)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Step 1 / 3 — Choose segmentation method and run:"))

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout.addWidget(self._img_label, stretch=1)

        # ── Method selector ──────────────────────────────────────────
        method_box = QGroupBox("Segmentation method")
        method_layout = QVBoxLayout(method_box)
        method_layout.setSpacing(4)

        self._rb_yolo_default = QRadioButton("YOLO — default model")
        self._rb_yolo_custom  = QRadioButton("YOLO — custom weights")
        self._rb_yolo_default.setChecked(True)

        self._rb_yolo_default.toggled.connect(self._on_method_changed)
        self._rb_yolo_custom.toggled.connect(self._on_method_changed)

        method_layout.addWidget(self._rb_yolo_default)
        method_layout.addWidget(self._rb_yolo_custom)

        # Custom YOLO path row (hidden by default)
        self._custom_path_row = QWidget()
        path_row_layout = QHBoxLayout(self._custom_path_row)
        path_row_layout.setContentsMargins(16, 0, 0, 0)
        path_row_layout.setSpacing(4)
        self._custom_path_edit = QLineEdit()
        self._custom_path_edit.setPlaceholderText("Path to custom .pt weights file…")
        self._custom_path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._on_browse_weights)
        path_row_layout.addWidget(self._custom_path_edit)
        path_row_layout.addWidget(browse_btn)
        self._custom_path_row.setVisible(False)
        method_layout.addWidget(self._custom_path_row)

        layout.addWidget(method_box)

        # ── Run Segmentation button ──────────────────────────────────
        self._run_btn = QPushButton("Run Segmentation")
        self._run_btn.setMinimumHeight(28)
        self._run_btn.clicked.connect(self._on_run_segmentation)
        layout.addWidget(self._run_btn)

        # ── Results section (hidden until segmentation runs) ─────────
        self._seg_results_widget = QWidget()
        res_layout = QVBoxLayout(self._seg_results_widget)
        res_layout.setContentsMargins(0, 4, 0, 0)
        res_layout.setSpacing(4)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        res_layout.addWidget(sep)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Confidence threshold:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.0, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.5)
        self._conf_spin.setFixedWidth(80)
        self._conf_spin.valueChanged.connect(self._apply_conf_filter)
        thr_row.addWidget(self._conf_spin)
        thr_row.addStretch()
        res_layout.addLayout(thr_row)

        res_layout.addWidget(QLabel("Detections above threshold:"))
        self._detection_list = QListWidget()
        self._detection_list.setMaximumHeight(100)
        self._detection_list.currentRowChanged.connect(self._on_detection_selected)
        res_layout.addWidget(self._detection_list)

        self._seg_results_widget.setVisible(False)
        layout.addWidget(self._seg_results_widget)

    def _init_page0(self):
        """Show plain RGB on open; no segmentation."""
        self._refresh_page0_image()

    def _on_method_changed(self):
        self._custom_path_row.setVisible(self._rb_yolo_custom.isChecked())

    def _on_browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO weights file", "", "PyTorch weights (*.pt);;All files (*)"
        )
        if path:
            self._custom_path_edit.setText(path)

    def _get_selected_method(self) -> str:
        if self._rb_yolo_custom.isChecked():
            return "yolo_custom"
        return "yolo_default"

    def _on_run_segmentation(self):
        method = self._get_selected_method()

        if method == "yolo_custom":
            path = self._custom_path_edit.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Custom weights missing",
                                    "Please browse to a valid .pt weights file.")
                return

        # Disable buttons while running
        self._next_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        rgb_copy   = self._rgb.copy()
        method_cap = method

        def _do_inference():
            import supervision as sv
            from PIL import Image as _PIL

            model_path = (self._default_yolo_path
                          if method_cap == "yolo_default"
                          else self._custom_path_edit.text().strip())
            if self._yolo_model is None or getattr(self, "_yolo_model_path", None) != model_path:
                from ultralytics import YOLO
                self._yolo_model = YOLO(model_path)
                self._yolo_model_path = model_path
            result = self._yolo_model.predict(_PIL.fromarray(rgb_copy), conf=0.25)[0]
            return sv.Detections.from_ultralytics(result)

        self._seg_worker = _SegWorker(_do_inference)
        self._seg_worker.done.connect(self._on_segmentation_done)
        self._seg_worker.error.connect(self._on_segmentation_error)
        self._seg_worker.start()

    def _on_segmentation_done(self, detections):
        QApplication.restoreOverrideCursor()
        self._next_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._detections = detections
        self._segmentation_run = True
        self._seg_results_widget.setVisible(True)
        self._apply_conf_filter()

    def _on_segmentation_error(self, msg: str):
        QApplication.restoreOverrideCursor()
        self._next_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        if "import" in msg.lower() or "no module" in msg.lower():
            QMessageBox.critical(self, "Import error", f"Required package not available:\n{msg}")
        else:
            QMessageBox.critical(self, "Segmentation error", msg)

    def _apply_conf_filter(self):
        """Filter detections by current threshold and refresh page 0."""
        thr = self._conf_spin.value()
        self.conf_threshold = thr

        det = self._detections
        if det is None or len(det) == 0:
            self._active_detections = det
            self._detection_list.clear()
            self._refresh_page0_image()
            return

        confs = det.confidence if det.confidence is not None else [1.0] * len(det)
        mask = np.array([float(c) >= thr for c in confs])
        self._active_detections = det[mask]

        names = det.data.get("class_name", [f"det_{i}" for i in range(len(det))])
        self._detection_list.blockSignals(True)
        self._detection_list.clear()
        for i, (nm, cf, keep) in enumerate(zip(names, confs, mask)):
            if keep:
                self._detection_list.addItem(f"[{i}] {nm}  {cf*100:.1f}%")
        self._detection_list.blockSignals(False)
        self._selected_idx = 0
        if self._detection_list.count() > 0:
            self._detection_list.setCurrentRow(0)
        self._refresh_page0_image()

    def _refresh_page0_image(self):
        if self._segmentation_run and self._active_detections is not None:
            annotated = _annotate_detections(self._rgb, self._active_detections, self._selected_idx)
        else:
            annotated = self._rgb
        pix = _ndarray_to_qpixmap(annotated)
        if pix.isNull():
            return
        w = self._img_label.width()
        h = self._img_label.height()
        if w <= 0 or h <= 0:
            self._img_label.setPixmap(pix)
            return
        self._img_label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_detection_selected(self, row: int):
        if row < 0:
            return
        self._selected_idx = row
        det = self._active_detections
        if det is not None and len(det) > row:
            names = det.data.get("class_name", [])
            if row < len(names):
                self._class_name = str(names[row])
        self._refresh_page0_image()

    def run_on_image(self, rgb_img: np.ndarray):
        """Run the same YOLO model on a new image. For batch use."""
        import supervision as sv
        from PIL import Image as _PIL
        result = self._yolo_model.predict(_PIL.fromarray(rgb_img), conf=0.25)[0]
        return sv.Detections.from_ultralytics(result)

    # ---- page 1: depth histogram ----
    def _build_page1(self):
        layout = QVBoxLayout(self._page1)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Step 2 / 3 — Histogram depth filter:"))

        h_split = QHBoxLayout()
        h_split.setSpacing(10)

        self._hist_fig    = Figure(tight_layout=True)
        self._hist_canvas = FigureCanvasQTAgg(self._hist_fig)
        h_split.addWidget(self._hist_canvas, stretch=1)

        overlay_col = QVBoxLayout()
        overlay_col.setSpacing(4)
        overlay_col.addWidget(QLabel("Depth mask on image:"))
        self._overlay_label = QLabel()
        self._overlay_label.setAlignment(Qt.AlignCenter)
        self._overlay_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._overlay_label.setStyleSheet("background-color: #1a1a2e; border: 1px solid #3c3c3c;")
        overlay_col.addWidget(self._overlay_label, stretch=1)
        h_split.addLayout(overlay_col, stretch=1)

        param_panel = QVBoxLayout()
        param_panel.setSpacing(8)

        param_panel.addWidget(QLabel("Resolution (bins/m):"))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["1", "10", "100"])
        self._res_combo.setCurrentIndex(0)
        param_panel.addWidget(self._res_combo)

        param_panel.addWidget(QLabel("Min height %:"))
        self._mhp_combo = QComboBox()
        self._mhp_combo.addItems(["5", "10", "20"])
        self._mhp_combo.setCurrentIndex(0)
        param_panel.addWidget(self._mhp_combo)

        self._ignore_bg_cb = QCheckBox("Ignore background")
        self._ignore_bg_cb.setChecked(False)
        param_panel.addWidget(self._ignore_bg_cb)

        rerun_btn = QPushButton("Re-run")
        rerun_btn.clicked.connect(self._apply_hdf_and_draw)
        param_panel.addWidget(rerun_btn)

        param_panel.addStretch()

        param_widget = QWidget()
        param_widget.setFixedWidth(160)
        param_widget.setLayout(param_panel)
        h_split.addWidget(param_widget)

        layout.addLayout(h_split, stretch=1)

    def _run_hdf_and_show(self):
        det = self._active_detections
        idx = self._selected_idx
        x1, y1, x2, y2 = det.xyxy[idx].astype(int)
        self._x1, self._y1 = x1, y1

        depth_crop = self._depth[y1:y2, x1:x2].astype(float)
        valid_px = depth_crop[depth_crop > 0]
        if valid_px.size > 0 and valid_px.max() <= 100:
            depth_crop = depth_crop * 1000.0

        if det.mask is not None and idx < len(det.mask):
            raw_mask = det.mask[idx].astype(np.uint8)[y1:y2, x1:x2]
        else:
            raw_mask = np.ones_like(depth_crop, dtype=np.uint8)
        self._depth_crop_cache   = depth_crop
        self._depth_masked_cache = np.where(raw_mask, depth_crop, 0)

        self._apply_hdf_and_draw()

    def _apply_hdf_and_draw(self):
        res_bins_per_m = int(self._res_combo.currentText())
        mhp            = int(self._mhp_combo.currentText())
        ignore_bg      = self._ignore_bg_cb.isChecked()

        effective_resolution = res_bins_per_m / 1000.0

        self._hdf_params = {"resolution": effective_resolution,
                            "max_height_percent": mhp,
                            "ignore_background": ignore_bg}

        filtered, hist, bin_edges, lb, ub, threshold, min_h = apply_hist_depth_filter(
            self._depth_masked_cache,
            ignore_background=ignore_bg,
            resolution=effective_resolution,
            max_height_percent=mhp,
        )

        self._filtered_depth = filtered
        self._mask_crop = (filtered > 0).astype(np.uint8)

        self._update_overlay_image()

        ax = (self._hist_fig.clf(), self._hist_fig.add_subplot(1, 1, 1))[1]
        ax.plot(bin_edges[:-1], hist, color="#4fc3f7", label="Histogram")
        ax.axvline(lb,        color="red",    linestyle="--", linewidth=1.5, label=f"LB {lb:.0f}")
        ax.axvline(ub,        color="green",  linestyle="--", linewidth=1.5, label=f"UB {ub:.0f}")
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1,   label="Avg")
        ax.axhline(min_h,     color="blue",   linestyle="--", linewidth=1,   label="Min-h")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_facecolor("#1a1a2e")
        self._hist_fig.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#d4d4d4")
        ax.xaxis.label.set_color("#d4d4d4")
        ax.yaxis.label.set_color("#d4d4d4")
        self._hist_canvas.draw()

    def _update_overlay_image(self):
        """Build the colour crop + cyan HDF mask overlay."""
        det = self._active_detections
        idx = self._selected_idx
        x1, y1, x2, y2 = det.xyxy[idx].astype(int)

        rgb_crop = self._rgb[y1:y2, x1:x2].copy()

        mask = self._mask_crop
        if mask.shape == rgb_crop.shape[:2]:
            overlay = rgb_crop.copy()
            where = mask > 0
            overlay[where] = np.clip(
                rgb_crop[where].astype(np.int32) // 2 + np.array([0, 180, 255]) // 2,
                0, 255,
            ).astype(np.uint8)
            rgb_crop = overlay

        self._overlay_pixmap = _ndarray_to_qpixmap(rgb_crop)
        self._refresh_overlay_pixmap()

    def _refresh_overlay_pixmap(self):
        if not hasattr(self, "_overlay_pixmap") or self._overlay_pixmap is None:
            return
        if self._overlay_pixmap.isNull():
            return
        w = self._overlay_label.width()
        h = self._overlay_label.height()
        if w > 0 and h > 0:
            self._overlay_label.setPixmap(
                self._overlay_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    # ---- page 2: top-down X-Z view ----
    def _build_page2(self):
        layout = QVBoxLayout(self._page2)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Step 3 / 3 — Top-down X-Z view. Verify alignment:"))

        self._topdown_fig    = Figure(figsize=(5, 5), tight_layout=True)
        self._topdown_canvas = FigureCanvasQTAgg(self._topdown_fig)
        layout.addWidget(self._topdown_canvas, stretch=1)

        self._pose_label = QLabel("")
        self._pose_label.setStyleSheet("color: #9cdcfe; font-size: 11px;")
        layout.addWidget(self._pose_label)

    def _run_pose_and_show(self):
        try:
            _, center, dimensions, yaw_deg, bbox_result = estimate_3d_pose(
                self._filtered_depth, self._mask_crop,
                self._x1, self._y1,
                self._fx, self._fy, self._cx, self._cy,
            )
        except Exception as e:
            self._pose_label.setText(f"Pose estimation failed: {e}")
            self._accept_btn.setEnabled(False)
            return

        self._estimated_center     = center
        self._estimated_dimensions = dimensions
        self._estimated_yaw        = yaw_deg

        self._pose_label.setText(
            f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m   "
            f"Dims (L×W×H): {dimensions[0]:.3f}×{dimensions[2]:.3f}×{dimensions[1]:.3f} m   "
            f"Yaw: {yaw_deg:.1f}°"
        )

        ax = (self._topdown_fig.clf(), self._topdown_fig.add_subplot(1, 1, 1))[1]

        if bbox_result is not None:
            pts   = bbox_result["points_xz"]
            hull  = bbox_result["hull_pts"]
            p1, p2 = bbox_result["edge"]
            bbox  = bbox_result["bbox"]
            u     = bbox_result["direction"]
            mean_xz = pts.mean(axis=0)

            ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.25, color="#4fc3f7")
            ax.scatter(hull[:, 0], hull[:, 1], c="white", s=8)

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r", linewidth=3, label="Ref edge")

            bb = np.vstack([bbox, bbox[0]])
            ax.plot(bb[:, 0], bb[:, 1], "g--", linewidth=1.5, label="Aligned BB")

            scale = 0.5 * np.linalg.norm(pts.max(0) - pts.min(0))
            ax.plot([mean_xz[0] - scale * u[0], mean_xz[0] + scale * u[0]],
                    [mean_xz[1] - scale * u[1], mean_xz[1] + scale * u[1]],
                    "r--", linewidth=1, label=f"Yaw {yaw_deg:.1f}°")

        ax.set_xlabel("X (m)", color="#d4d4d4")
        ax.set_ylabel("Z (m)", color="#d4d4d4")
        ax.set_title(f"Top-down view — yaw = {yaw_deg:.1f}°", color="#d4d4d4")
        ax.axis("equal")
        ax.grid(True, color="#333")
        ax.legend(fontsize=8)
        ax.set_facecolor("#1a1a2e")
        self._topdown_fig.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#d4d4d4")
        self._topdown_canvas.draw()

        self._accept_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _on_next(self):
        page = self._stack.currentIndex()
        if page == 0:
            if not self._segmentation_run:
                QMessageBox.warning(self, "Run segmentation first",
                    "Please select a method and click 'Run Segmentation'.")
                return
            if self._active_detections is None or len(self._active_detections) == 0:
                QMessageBox.warning(self, "No detections",
                    "No detections above the confidence threshold.\n"
                    "Adjust the threshold or re-run segmentation.")
                return
            self._run_hdf_and_show()
            self._stack.setCurrentIndex(1)
            self._back_btn.setVisible(True)
        elif page == 1:
            self._run_pose_and_show()
            self._stack.setCurrentIndex(2)
            self._next_btn.setVisible(False)
            self._accept_btn.setVisible(True)

    def _on_back(self):
        page = self._stack.currentIndex()
        if page == 1:
            self._stack.setCurrentIndex(0)
            self._back_btn.setVisible(False)
        elif page == 2:
            self._stack.setCurrentIndex(1)
            self._next_btn.setVisible(True)
            self._accept_btn.setVisible(False)

    def _on_accept(self):
        self.result = make_label_object(
            self._class_name,
            self._estimated_center,
            self._estimated_dimensions,
            self._estimated_yaw,
        )
        self.accept()

    def reject(self):
        self.result = None
        super().reject()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._stack.currentIndex() == 0:
            self._refresh_page0_image()
        elif self._stack.currentIndex() == 1:
            self._refresh_overlay_pixmap()
