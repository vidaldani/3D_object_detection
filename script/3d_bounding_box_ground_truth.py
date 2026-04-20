import open3d as o3d
import numpy as np
import random
import os
import glob
import json
import gc

# =============================================================
# CONFIG
# =============================================================
PCD_DIR   = "/home/tumwfml-ubunt6/3D_object_detection/rd_test_dataset/Azure_Kinect_dataset/pcd_colored"
LABEL_DIR = "/home/tumwfml-ubunt6/3D_object_detection/rd_test_dataset/Azure_Kinect_dataset/3d_labels"

FRAME_BASENAME = None  # Set to e.g. "frame_0014" or None for random

CLASS_COLORS = {
    "forklift":     [1.0, 0.0, 0.0],
    "pallet":       [0.0, 1.0, 0.0],
    "pallet truck": [0.0, 0.0, 1.0],
    "klt":          [1.0, 1.0, 0.0],
    "stillage":     [1.0, 0.0, 1.0],
    "person":       [1.0, 0.5, 0.0],
}
DEFAULT_COLOR = [0.0, 1.0, 1.0]

bbox_lines = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
]

def build_bbox_geometry(obj):
    center = np.array([
        obj["centroid"]["x"],
        obj["centroid"]["y"],
        obj["centroid"]["z"]
    ])

    L = obj["dimensions"]["length"]
    W = obj["dimensions"]["width"]
    H = obj["dimensions"]["height"]

    yaw_rad = np.deg2rad(obj["rotations"]["y"])
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, yaw_rad, 0))

    local_corners = np.array([
        [-L/2, -H/2, -W/2],
        [ L/2, -H/2, -W/2],
        [ L/2,  H/2, -W/2],
        [-L/2,  H/2, -W/2],
        [-L/2, -H/2,  W/2],
        [ L/2, -H/2,  W/2],
        [ L/2,  H/2,  W/2],
        [-L/2,  H/2,  W/2],
    ])

    bbox_points = (R @ local_corners.T).T + center

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bbox_points),
        lines=o3d.utility.Vector2iVector(bbox_lines),
    )
    color = CLASS_COLORS.get(obj["name"], DEFAULT_COLOR)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(bbox_lines))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    frame.rotate(R, center=[0, 0, 0])
    frame.translate(center)

    return lineset, frame


def visualize(pcd_path, label_path):
    print(f"\nPoint cloud : {pcd_path}")
    print(f"Label file  : {label_path}")

    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    with open(label_path, "r") as f:
        objects = json.load(f).get("objects", [])
    print(f"Loaded {len(objects)} object(s)")

    geometries = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]

    for i, obj in enumerate(objects):
        print(f"  [{i}] {obj['name']:15s} "
              f"center=({obj['centroid']['x']:.3f}, {obj['centroid']['y']:.3f}, {obj['centroid']['z']:.3f})  "
              f"L={obj['dimensions']['length']:.3f} W={obj['dimensions']['width']:.3f} H={obj['dimensions']['height']:.3f}  "
              f"yaw={obj['rotations']['y']:.1f}°")
        bbox_ls, bbox_frame = build_bbox_geometry(obj)
        geometries.extend([bbox_ls, bbox_frame])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Ground Truth — {os.path.basename(pcd_path)}", width=1600, height=900)

    for g in geometries:
        vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_lookat([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.01)

    vis.run()
    vis.destroy_window()

    del vis, pcd, geometries
    gc.collect()


def main():
    pcd_files = sorted(glob.glob(os.path.join(PCD_DIR, "*.pcd")))
    if not pcd_files:
        raise RuntimeError(f"No .pcd files found in {PCD_DIR}")

    if FRAME_BASENAME is not None:
        pcd_path = os.path.join(PCD_DIR, FRAME_BASENAME + ".pcd")
    else:
        pcd_path = random.choice(pcd_files)
        print(f"Randomly selected: {os.path.basename(pcd_path)}")

    base = os.path.splitext(os.path.basename(pcd_path))[0]
    label_path = os.path.join(LABEL_DIR, base + ".json")

    if not os.path.exists(label_path):
        raise RuntimeError(f"No matching label for {pcd_path}")

    visualize(pcd_path, label_path)


if __name__ == "__main__":
    main()
