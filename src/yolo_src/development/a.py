#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
import pickle
import os
import rclpy

# ── Robot setup ──
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "a0509"

import DR_init
DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node("calib_collector", namespace=ROBOT_ID)
DR_init.__dsr__node = node

from DSR_ROBOT2 import (get_current_posx, set_robot_mode, ROBOT_MODE_MANUAL)
set_robot_mode(ROBOT_MODE_MANUAL)

# ── ChArUco board definition ──
ROWS       = 5
COLS       = 7
CHECKER_MM = 0.025
MARKER_MM  = CHECKER_MM * 0.75
DICTIONARY = cv2.aruco.DICT_4X4_50

aruco_dict   = cv2.aruco.getPredefinedDictionary(DICTIONARY)
board_params = cv2.aruco.DetectorParameters()
board        = cv2.aruco.CharucoBoard((COLS, ROWS), CHECKER_MM, MARKER_MM, aruco_dict)

# ── NEW API detectors ──
aruco_detector   = cv2.aruco.ArucoDetector(aruco_dict, board_params)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ── RealSense ──
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)

profile = pipeline.get_active_profile()
intr    = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
K       = np.array([[intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0,       0,         1]])
dist    = np.array(intr.coeffs)

# ── Storage ──
save_dir = "calib_data"
os.makedirs(save_dir, exist_ok=True)
sample_count = 0

print("\n========================================")
print("INSTRUCTIONS:")
print("  1. Enable direct-teach mode on teach pendant")
print("  2. Move robot so board is visible in window")
print("  3. SPACE = capture,  Q = save & quit")
print("  4. Aim for 15-20 varied poses")
print("========================================\n")

while True:
    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame   = np.asanyarray(color_frame.get_data())
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()

    detected = False
    rvec = tvec = None

    # ── Detection (new API) ──
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 4:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        # CharucoDetector returns: charuco_corners, charuco_ids, marker_corners, marker_ids
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)

        if ch_corners is not None and len(ch_corners) >= 6:
            cv2.aruco.drawDetectedCornersCharuco(display, ch_corners, ch_ids)

            objPoints, imgPoints = board.matchImagePoints(ch_corners, ch_ids)

            ok, rvec, tvec = cv2.solvePnP(
                objPoints,
                imgPoints,
                K,
                dist
            )

            if ok:
                detected = True
                cv2.drawFrameAxes(display, K, dist, rvec, tvec, 0.05)

    # ── Live TCP readout ──
    tcp = None
    try:
        tcp      = get_current_posx()[0]
        tcp_text = f"TCP: X={tcp[0]:.1f}  Y={tcp[1]:.1f}  Z={tcp[2]:.1f}"
        cv2.putText(display, tcp_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    except:
        cv2.putText(display, "TCP: (robot not connected)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

    color  = (0, 255, 0) if detected else (0, 0, 255)
    status = f"BOARD OK | Samples: {sample_count}" if detected else f"Board NOT detected | Samples: {sample_count}"
    cv2.putText(display, status,      (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,  color,        2)
    cv2.putText(display, "SPACE=capture  Q=quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Calibration Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if not detected:
            print("!! Board not visible — reposition and try again."); continue
        if tcp is None:
            print("!! Cannot read robot TCP — check connection.");      continue

        R_cam, _ = cv2.Rodrigues(rvec)
        t_cam    = tvec.flatten()

        sample = {
            "tcp":          list(tcp),
            "R_target2cam": R_cam,
            "t_target2cam": t_cam
        }
        fname = os.path.join(save_dir, f"sample_{sample_count:03d}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(sample, f)

        sample_count += 1
        print(f"✅ Sample {sample_count} captured.")
        print(f"   TCP:     {[round(v, 2) for v in tcp]}")
        print(f"   Board Z: {t_cam[2]*1000:.1f} mm\n")

    elif key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
rclpy.shutdown()
print(f"\nDone. {sample_count} samples saved to '{save_dir}/'")