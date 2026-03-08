#!/usr/bin/env python3
import rclpy
import cv2
import os
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO

# =========================
# Robot config
# =========================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "a0509"

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

# =========================
# YOLO model
# =========================
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "..", ".."))
model = YOLO(
    os.path.join(
        project_root,
        "runs",
        "classify",
        "train",
        "weights",
        "best.pt"
    )
)

# =========================
# RealSense setup
# =========================
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# =========================
# Main
# =========================
def main():
    rclpy.init()
    node = rclpy.create_node("banana_pick_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (
        movel,
        movej,
        posj,
        get_current_posx,
        get_current_posj,
        set_robot_mode,
        set_velx,
        set_accx,
        DR_BASE,
        DR_MV_MOD_ABS,
        ROBOT_MODE_AUTONOMOUS
    )

    # Robot setup
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    
    # =========================
    # MOVE TO NEW INITIAL POSITION
    # Coordinates from your second image:
    # X: 493.58, Y: -0.11, Z: 484.45, RX: 0.02, RY: 89.97, RZ: -90.01
    # =========================
    home_posx = [493.58, -0.11, 484.45, 0.02, 89.97, -90.01]
    
    print(f"Moving to initial position: {home_posx}")
    # Move at a moderate speed to the start position
    movel(home_posx, v=10, a=10, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    
    # Set speed back to VERY SLOW for the detection phase
    set_velx(5, 5)     
    set_accx(10, 10)

    moved_once = False  # prevent repeated motion

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # =========================
            # YOLO classification
            # =========================
            results = model(frame, verbose=False)[0]

            if results.probs is not None:
                cls_id = results.probs.top1
                conf = results.probs.top1conf.item()
                label = model.names[cls_id]

                is_ripe = "ripe" in label.lower() and "unripe" not in label.lower()
                color = (0, 255, 0) if is_ripe else (0, 0, 255)

                cv2.putText(
                    frame,
                    f"{label} {conf:.2%}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                # =========================
                # Depth → XYZ
                # =========================
                h, w, _ = frame.shape
                cx, cy = w // 2, h // 2
                depth = depth_frame.get_distance(cx, cy)

                if depth > 0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        intrinsics,
                        [cx, cy],
                        depth
                    )

                    print(
                        f"{label} | conf={conf:.2f} | "
                        f"X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}"
                    )

                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    # =========================
                    # Robot motion (ONLY IF RIPE)
                    # =========================
                    if is_ripe and not moved_once:
                        moved_once = True
                        print("RIPE banana detected → moving robot")

                        # Get current position to modify Z
                        current_pos = list(get_current_posx()[0])

                        # Move forward relative to the banana depth (Z axis)
                        # Clipping at 80mm to prevent excessive travel
                        dz = np.clip(Z * 1000, 0, 80)   
                        current_pos[2] -= dz

                        print(f"Moving dz: {dz:.2f}mm")
                        print(f"Target posx: {current_pos}")

                        movel(
                            current_pos,
                            v=[5, 5],
                            a=[10, 10],
                            ref=DR_BASE,
                            mod=DR_MV_MOD_ABS
                        )

            cv2.imshow("Banana Classification + 3D Position", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()


def normalize_joints(raw):
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    if isinstance(raw, list):
        return raw
    if hasattr(raw, "j1"):  # Doosan joint object
        return [raw.j1, raw.j2, raw.j3, raw.j4, raw.j5, raw.j6]
    raise TypeError(f"Unknown joint type: {type(raw)}")


if __name__ == "__main__":
    main()
