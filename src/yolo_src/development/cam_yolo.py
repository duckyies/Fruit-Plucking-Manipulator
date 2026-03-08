
import cv2
import os
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# =========================
# Model
# =========================
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "..", ".."))

model = YOLO(os.path.join(project_root, "runs", "classify", "train", "weights", "best.pt"))


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Camera intrinsics
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# =========================
# Main loop
# =========================
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

            # Draw label
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
            # Depth → XYZ (center pixel)
            # =========================
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2

            depth = depth_frame.get_distance(cx, cy)

            if depth > 0:
                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    intrinsics, [cx, cy], depth
                )

                print(
                    f"{label} | conf={conf:.2f} | "
                    f"X={X:.3f} m, Y={Y:.3f} m, Z={Z:.3f} m"
                )

                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        cv2.imshow("Banana Classification + 3D Position", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


