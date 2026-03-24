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
PCD_DIR = "/home/tumwfml-ubunt6/3D_object_detection/rd_test_dataset/Azure_Kinect_dataset/pcd_colored"
LABEL_DIR = "/home/tumwfml-ubunt6/3D_object_detection/rd_test_dataset/Azure_Kinect_dataset/3d_labels"

# Set to None to loop through all files
# FRAME_BASENAME = "frame_0023"
FRAME_BASENAME = None

CLASS_COLORS = {
    "forklift": [1.0, 0.0, 0.0],
    "pallet": [0.0, 1.0, 0.0],
    "pallet_truck": [0.0, 0.0, 1.0],
    "klt": [1.0, 1.0, 0.0],
    "stillage": [1.0, 0.0, 1.0],
}
DEFAULT_COLOR = [0.0, 1.0, 1.0]


# =============================================================
# LOAD JSON LABELS
# =============================================================
def load_3d_boxes_from_json(label_path):
    with open(label_path, "r") as f:
        data = json.load(f)

    boxes = []

    for obj in data.get("objects", []):
        name = obj.get("name", "unknown")

        centroid = obj["centroid"]
        dimensions = obj["dimensions"]
        rotations = obj["rotations"]

        center = np.array([
            float(centroid["x"]),
            float(centroid["y"]),
            float(centroid["z"])
        ], dtype=float)

        # Same bbox convention as your working sample:
        # W, H, D = size
        size = np.array([
            float(dimensions["length"]),
            float(dimensions["width"]),
            float(dimensions["height"])
        ], dtype=float)

        # Same yaw convention as your working sample
        yaw_deg = float(rotations["y"])

        boxes.append({
            "name": name,
            "center": center,
            "dimensions": size,
            "yaw": yaw_deg
        })

    return boxes


# =============================================================
# CREATE 3D BOUNDING BOX
# =============================================================
def create_3d_bbox(center, size, yaw_deg, color):
    W, H, D = size

    local_corners = np.array([
        [-W/2, -H/2, -D/2],
        [ W/2, -H/2, -D/2],
        [ W/2,  H/2, -D/2],
        [-W/2,  H/2, -D/2],
        [-W/2, -H/2,  D/2],
        [ W/2, -H/2,  D/2],
        [ W/2,  H/2,  D/2],
        [-W/2,  H/2,  D/2],
    ], dtype=float)

    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    # Exact same rotation convention as your working sample
    R_align = o3d.geometry.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
    yaw_rad = np.deg2rad(yaw_deg)
    R_yaw = o3d.geometry.get_rotation_matrix_from_xyz((0, yaw_rad, 0))
    R = R_yaw @ R_align

    corners = (R @ local_corners.T).T + center

    bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    bbox.colors = o3d.utility.Vector3dVector([color] * len(lines))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
    frame.rotate(R, center=[0, 0, 0])
    frame.translate(center)

    return bbox, frame


# =============================================================
# FILE MATCHING
# =============================================================
def find_matching_label_file(pcd_path, label_dir):
    base = os.path.splitext(os.path.basename(pcd_path))[0]
    json_path = os.path.join(label_dir, base + ".json")
    if os.path.exists(json_path):
        return json_path
    return None


# =============================================================
# VISUALIZATION
# =============================================================
def visualize_pcd_with_boxes(pcd_path, label_path):
    print(f"\nPoint cloud: {pcd_path}")
    print(f"Label file : {label_path}")

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(pcd_path)

    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    boxes_data = load_3d_boxes_from_json(label_path)
    print(f"Loaded {len(boxes_data)} objects")

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=900)

    vis.add_geometry(pcd)
    vis.add_geometry(origin)

    geometries = [pcd, origin]

    for i, box_data in enumerate(boxes_data):
        color = CLASS_COLORS.get(box_data["name"], DEFAULT_COLOR)

        bbox, frame = create_3d_bbox(
            box_data["center"],
            box_data["dimensions"],
            box_data["yaw"],
            color=color
        )

        vis.add_geometry(bbox)
        vis.add_geometry(frame)
        geometries.extend([bbox, frame])

        print(
            f"Object {i}: {box_data['name']} | "
            f"center={box_data['center']} | "
            f"dimensions={box_data['dimensions']} | "
            f"yaw={box_data['yaw']}"
        )

    vis.poll_events()
    vis.update_renderer()

    # ---------------------------------------------------------
    # EXACT same camera convention as your working sample
    # blue forward, green down, red right
    # ---------------------------------------------------------
    ctr = vis.get_view_control()

    eye = np.array([0.0, 0.0, 0.0])
    front = np.array([0.0, 0.0, -1.0])   # look along +Z (blue)
    up = np.array([0.0, -1.0, 0.0])      # Y down

    ctr.set_front(front)
    ctr.set_lookat(eye + front)
    ctr.set_up(up)
    ctr.set_zoom(0.01)

    vis.run()
    vis.destroy_window()

    del vis, pcd, geometries
    gc.collect()


# =============================================================
# MAIN
# =============================================================
def main():
    pcd_files = sorted(glob.glob(os.path.join(PCD_DIR, "*.pcd")))

    if not pcd_files:
        raise RuntimeError(f"No .pcd files found in {PCD_DIR}")

    # ---------------------------------------------------------
    # If a specific frame is set → use it
    # Otherwise → pick a random file
    # ---------------------------------------------------------
    if FRAME_BASENAME is not None:
        pcd_path = os.path.join(PCD_DIR, FRAME_BASENAME + ".pcd")
    else:
        pcd_path = random.choice(pcd_files)
        print(f"Randomly selected: {os.path.basename(pcd_path)}")

    label_path = find_matching_label_file(pcd_path, LABEL_DIR)

    if label_path is None:
        raise RuntimeError(f"No matching label for {pcd_path}")

    visualize_pcd_with_boxes(pcd_path, label_path)


if __name__ == "__main__":
    main()