"""
pose_estimation_pipeline.py

All pipeline functions for autonomous 3D bounding box generation.
Core algorithm functions are extracted verbatim from
script/AzureKinect_pose_comparison.ipynb.
"""

import os
import re
import json

import numpy as np
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Histogram depth filtering
# ---------------------------------------------------------------------------

def calculate_peak_region(peak_index, hist, h_threshold):
    left_edge = peak_index
    right_edge = peak_index

    while left_edge > 0 and hist[left_edge] > h_threshold:
        left_edge -= 1
    while right_edge < len(hist) - 1 and hist[right_edge] > h_threshold:
        right_edge += 1

    element_count = np.sum(hist[left_edge:right_edge + 1])
    return left_edge, right_edge, element_count


def apply_hist_depth_filter(depth_image, ignore_background=False, resolution=1, max_height_percent=5):
    depth_values = depth_image.flatten()
    depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0)]

    if len(depth_values) == 0:
        return depth_image, np.zeros(1), np.array([0, 1]), 0, 0, 0, 0

    max_val = max(depth_values)
    bins = max(1, round(max_val * resolution))

    hist, bin_edges = np.histogram(depth_values, bins=bins)
    max_peak_index = np.argmax(hist)

    min_height = hist[max_peak_index] * max_height_percent / 100
    threshold = max(np.mean(hist), min_height)

    try:
        left_edge, right_edge, count = calculate_peak_region(max_peak_index, hist, min_height)
    except Exception as e:
        print(f"Peak calculation failed: {e}")
        return depth_image, hist, bin_edges, 0, 0, threshold, min_height

    lower_bound = bin_edges[left_edge]
    upper_bound = bin_edges[right_edge]

    mask = (depth_image >= lower_bound) & (depth_image <= upper_bound)
    filtered_image = np.where(mask, depth_image, 0)

    return filtered_image, hist, bin_edges, lower_bound, upper_bound, threshold, min_height


# ---------------------------------------------------------------------------
# Yaw + aligned bounding box
# ---------------------------------------------------------------------------

def estimate_yaw_and_aligned_bbox_from_top4_front_hull_segments(
    depth_crop,
    mask_crop,
    x1, y1,
    cx, cy, fx, fy
):
    """
    Returns yaw + geometry for visualization. No plotting inside.
    """
    ys, xs = np.where(mask_crop > 0)
    if len(xs) < 30:
        return None

    global_xs = xs + x1
    zs = depth_crop[ys, xs].astype(np.float32) / 1000.0
    valid = zs > 0

    global_xs = global_xs[valid]
    zs = zs[valid]

    X = (global_xs - cx) * zs / fx
    Z = zs
    points_xz = np.stack([X, Z], axis=1)

    if len(points_xz) < 30:
        return None

    hull = ConvexHull(points_xz)
    hull_pts = points_xz[hull.vertices]
    N = len(hull_pts)

    segments = []
    for i in range(N):
        p1 = hull_pts[i]
        p2 = hull_pts[(i + 1) % N]
        d = p2 - p1
        L = np.linalg.norm(d)
        mid_z = 0.5 * (p1[1] + p2[1])
        segments.append({"p1": p1, "p2": p2, "length": L, "mid_z": mid_z})

    top4 = sorted(segments, key=lambda s: s["length"], reverse=True)[:4]
    front2 = sorted(top4, key=lambda s: s["mid_z"])[:2]
    best = max(front2, key=lambda s: s["length"])

    p1, p2 = best["p1"], best["p2"]
    edge_dir = p2 - p1
    edge_dir /= np.linalg.norm(edge_dir)

    # arctan2(-Z, X): angle that rotates local X onto the edge direction in the
    # renderer's _rotation_y convention (R maps local X → [cosθ, 0, -sinθ]).
    yaw = np.arctan2(-edge_dir[1], edge_dir[0])
    yaw_deg = np.rad2deg(yaw)

    if yaw_deg > 90:
        yaw_deg -= 180
    elif yaw_deg < -90:
        yaw_deg += 180

    u = edge_dir
    v = np.array([-u[1], u[0]])

    proj_u = points_xz @ u
    proj_v = points_xz @ v

    u_min, u_max = proj_u.min(), proj_u.max()
    v_min, v_max = proj_v.min(), proj_v.max()

    bbox = np.array([
        u_min * u + v_min * v,
        u_max * u + v_min * v,
        u_max * u + v_max * v,
        u_min * u + v_max * v,
    ])

    return {
        "yaw_deg":   float(yaw_deg),
        "points_xz": points_xz,
        "hull_pts":  hull_pts,
        "edge":      (p1, p2),
        "bbox":      bbox,
        "direction": u,
    }


# ---------------------------------------------------------------------------
# Full 3D pose — accepts intrinsics directly instead of a file path
# ---------------------------------------------------------------------------

def estimate_3d_pose(depth_crop, mask_crop, x1, y1, fx, fy, cx, cy):
    """
    3D pose estimation using aligned X-Z bounding box + axis-aligned Y extent.

    Returns:
        distance (float),
        center   (np.ndarray, shape (3,)),
        dimensions (np.ndarray, shape (3,))  — [width_perp, height_y, depth_along],
        yaw_deg  (float),
        yaw_bbox_result (dict | None)
    """
    ys, xs = np.where(mask_crop > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask")

    global_xs = xs + x1
    global_ys = ys + y1

    zs = depth_crop[ys, xs].astype(np.float32) / 1000.0
    valid = zs > 0

    global_xs = global_xs[valid]
    global_ys = global_ys[valid]
    zs = zs[valid]

    Xs = (global_xs - cx) * zs / fx
    Ys = (global_ys - cy) * zs / fy
    Zs = zs

    yaw_bbox_result = estimate_yaw_and_aligned_bbox_from_top4_front_hull_segments(
        depth_crop, mask_crop, x1, y1, cx, cy, fx, fy
    )

    if yaw_bbox_result is None:
        yaw = 0.0
        bbox_xz = None
    else:
        yaw = yaw_bbox_result["yaw_deg"]
        bbox_xz = yaw_bbox_result["bbox"]

    if bbox_xz is None:
        width = depth = 0.0
        center_x = center_z = 0.0
    else:
        width   = np.linalg.norm(bbox_xz[1] - bbox_xz[0])
        depth   = np.linalg.norm(bbox_xz[2] - bbox_xz[1])
        center_x = bbox_xz[:, 0].mean()
        center_z = bbox_xz[:, 1].mean()

    min_y, max_y = Ys.min(), Ys.max()
    height = max_y - min_y

    dimensions = np.array([width, height, depth])
    center_y   = 0.5 * (min_y + max_y)
    center     = np.array([center_x, center_y, center_z])
    distance   = float(np.linalg.norm(center))

    return distance, center, dimensions, yaw, yaw_bbox_result


# ---------------------------------------------------------------------------
# File discovery helpers — same numeric-ID approach as _find_rgb_image
# ---------------------------------------------------------------------------

def _last_numeric_id(name: str) -> int | None:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def find_depth_file(depth_dir: str, frame_id: str) -> str | None:
    """Find the depth .npy file for frame_id in depth_dir."""
    if not os.path.isdir(depth_dir):
        return None

    # Exact-name fast paths
    for candidate in (f"{frame_id}_depth.npy", f"{frame_id}.npy"):
        path = os.path.join(depth_dir, candidate)
        if os.path.exists(path):
            return path

    # Numeric-ID fallback
    target = _last_numeric_id(frame_id)
    if target is None:
        return None

    for fname in sorted(os.listdir(depth_dir)):
        if not fname.lower().endswith(".npy"):
            continue
        if _last_numeric_id(fname) == target:
            return os.path.join(depth_dir, fname)

    return None


def find_camera_params_file(params_dir: str, frame_id: str) -> str | None:
    """Find camera intrinsics file (.npz or .json) for frame_id in params_dir."""
    if not os.path.isdir(params_dir):
        return None

    # If there is exactly one file in the folder, treat it as a shared intrinsics file
    all_files = [f for f in os.listdir(params_dir)
                 if f.lower().endswith(".npz") or f.lower().endswith(".json")]
    if len(all_files) == 1:
        return os.path.join(params_dir, all_files[0])

    # Exact-name fast paths (prefer .npz)
    for ext in (".npz", ".json"):
        for candidate in (f"{frame_id}_camera_parameters{ext}", f"{frame_id}{ext}"):
            path = os.path.join(params_dir, candidate)
            if os.path.exists(path):
                return path

    # Numeric-ID fallback
    target = _last_numeric_id(frame_id)
    if target is None:
        return None

    for ext_order in (".npz", ".json"):
        for fname in sorted(os.listdir(params_dir)):
            if not fname.lower().endswith(ext_order):
                continue
            if _last_numeric_id(fname) == target:
                return os.path.join(params_dir, fname)

    return None


# ---------------------------------------------------------------------------
# Camera intrinsics loaders
# ---------------------------------------------------------------------------

def load_intrinsics_npz(path: str) -> tuple:
    """Load fx, fy, cx, cy from an Azure Kinect-style .npz file."""
    params = np.load(path)
    intr = params["rgb_intrinsics"]
    return float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])


def load_intrinsics_json(path: str) -> tuple:
    """Load fx, fy, cx, cy from a ZED2-style JSON with a 'left_camera' section."""
    with open(path, "r") as f:
        data = json.load(f)
    cam = data.get("left_camera", data)
    return float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])


def load_intrinsics(path: str) -> tuple:
    """Dispatch to the correct loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_intrinsics_npz(path)
    if ext == ".json":
        return load_intrinsics_json(path)
    raise ValueError(f"Unsupported intrinsics file format: {path}")


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------

def make_label_object(class_name: str, center: np.ndarray,
                      dimensions: np.ndarray, yaw_deg: float) -> dict:
    """
    Convert pipeline output to the GUI's label JSON format.

    dimensions layout: [width_perp_yaw, height_y, depth_along_yaw]
    label JSON layout:
        length = dimensions[0]  (width, perpendicular to yaw)
        width  = dimensions[2]  (depth, along yaw)
        height = dimensions[1]  (Y extent)
    """
    return {
        "name": class_name,
        "centroid": {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        },
        "dimensions": {
            "length": float(dimensions[0]),
            "width":  float(dimensions[2]),
            "height": float(dimensions[1]),
        },
        "rotations": {
            "x": 0.0,
            "y": float(yaw_deg),
            "z": 0.0,
        },
    }
