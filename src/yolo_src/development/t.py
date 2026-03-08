#!/usr/bin/env python3
"""
Reads saved samples and computes T_tool_cam (hand-eye matrix).
Saves result to T_tool_cam.txt
"""

import cv2
import numpy as np
import pickle
import glob
import math

def doosan_tcp_to_matrix(tcp):
    """
    Convert Doosan TCP [x, y, z, rx, ry, rz] to 4x4 homogeneous matrix.
    Doosan uses ZYZ Euler angles in degrees.
    x, y, z in mm → convert to metres.
    """
    x, y, z, rx, ry, rz = tcp
    x /= 1000.0; y /= 1000.0; z /= 1000.0  # mm → m

    # ZYZ Euler: Rz(rz) * Ry(ry) * Rz(rx)
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    Rz1 = np.array([[ math.cos(rz), -math.sin(rz), 0],
                    [ math.sin(rz),  math.cos(rz), 0],
                    [0,              0,             1]])
    Ry  = np.array([[ math.cos(ry), 0, math.sin(ry)],
                    [0,             1, 0            ],
                    [-math.sin(ry), 0, math.cos(ry)]])
    Rz2 = np.array([[ math.cos(rx), -math.sin(rx), 0],
                    [ math.sin(rx),  math.cos(rx), 0],
                    [0,              0,             1]])

    R = Rz1 @ Ry @ Rz2

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T

# ── Load all samples ──
files = sorted(glob.glob("calib_data/sample_*.pkl"))
print(f"Found {len(files)} samples.")
assert len(files) >= 10, "Need at least 10 samples for reliable calibration!"

R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list   = []
t_target2cam_list   = []

for f in files:
    with open(f, "rb") as fp:
        s = pickle.load(fp)

    T = doosan_tcp_to_matrix(s["tcp"])
    R_gripper2base_list.append(T[:3, :3])
    t_gripper2base_list.append(T[:3,  3])
    R_target2cam_list.append(s["R_target2cam"])
    t_target2cam_list.append(s["t_target2cam"])

# ── Run calibration (try multiple methods, pick best) ──
methods = {
    "TSAI":   cv2.CALIB_HAND_EYE_TSAI,
    "PARK":   cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":cv2.CALIB_HAND_EYE_ANDREFF,
}

best_err = 1e9
best_T   = None
best_method = ""

for name, method in methods.items():
    R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list,   t_target2cam_list,
        method=method
    )
    T = np.eye(4)
    T[:3, :3] = R_cam2tool
    T[:3,  3] = t_cam2tool.flatten()

    # Compute reprojection error: for each sample, check consistency
    errors = []
    for i in range(len(files)):
        T_base_tool = np.eye(4)
        T_base_tool[:3,:3] = R_gripper2base_list[i]
        T_base_tool[:3, 3] = t_gripper2base_list[i]

        T_cam_board = np.eye(4)
        T_cam_board[:3,:3] = R_target2cam_list[i]
        T_cam_board[:3, 3] = t_target2cam_list[i]

        # Board origin in base frame
        P_base = T_base_tool @ T @ T_cam_board @ np.array([0,0,0,1])
        errors.append(P_base[:3])

    # Consistency: std dev of board position across poses should be small
    errors = np.array(errors)
    err = np.std(errors, axis=0).mean() * 1000  # mm
    print(f"  {name}: consistency error = {err:.2f} mm")

    if err < best_err:
        best_err    = err
        best_T      = T
        best_method = name

print(f"\n✅ Best method: {best_method}  (error: {best_err:.2f} mm)")
print("\nT_tool_cam (camera relative to TCP):")
print(best_T)

np.savetxt("T_tool_cam.txt", best_T)
print("\n✅ Saved to T_tool_cam.txt")

# Show translation clearly
t_mm = best_T[:3, 3] * 1000
print(f"\nCamera offset from TCP:")
print(f"  X: {t_mm[0]:.1f} mm")
print(f"  Y: {t_mm[1]:.1f} mm")
print(f"  Z: {t_mm[2]:.1f} mm")