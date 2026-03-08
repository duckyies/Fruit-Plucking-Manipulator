#!/usr/bin/env python3
import rclpy
import tkinter as tk
import numpy as np
import cv2
import os
import threading
import pyrealsense2 as rs
import time
import minimalmodbus
import serial
from PIL import Image, ImageTk
from ultralytics import YOLO
import math
import time

# ==============================
# Robot Configuration
# ==============================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "a0509"

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
T_TOOL_CAM = np.loadtxt("T_tool_cam.txt")
print("✅ Loaded T_tool_cam.txt")
print(f"   Camera offset: X={T_TOOL_CAM[0,3]*1000:.1f}mm  Y={T_TOOL_CAM[1,3]*1000:.1f}mm  Z={T_TOOL_CAM[2,3]*1000:.1f}mm")

def doosan_tcp_to_matrix(tcp):
    """Convert Doosan TCP [x,y,z,rz1,ry,rz2] to 4x4 matrix. mm → metres."""
    x, y, z, rz1_deg, ry_deg, rz2_deg = tcp
    x /= 1000.0; y /= 1000.0; z /= 1000.0

    rz1 = math.radians(rz1_deg)
    ry  = math.radians(ry_deg)
    rz2 = math.radians(rz2_deg)

    Rz1 = np.array([[ math.cos(rz1), -math.sin(rz1), 0],
                    [ math.sin(rz1),  math.cos(rz1), 0],
                    [ 0,              0,              1]])
    Ry  = np.array([[ math.cos(ry),  0, math.sin(ry)],
                    [ 0,             1, 0            ],
                    [-math.sin(ry),  0, math.cos(ry)]])
    Rz2 = np.array([[ math.cos(rz2), -math.sin(rz2), 0],
                    [ math.sin(rz2),  math.cos(rz2), 0],
                    [ 0,              0,              1]])
    R = Rz1 @ Ry @ Rz2
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T

# ==============================
# DH GRIPPER DRIVER (ROBUST VERSION)
# ==============================

class DHGripperUSB:
    def __init__(self, port="/dev/ttyUSB0", slave_id=1):
        try:
            self.gripper = minimalmodbus.Instrument(port, slave_id)
            self.gripper.serial.baudrate = 115200
            self.gripper.serial.bytesize = 8
            self.gripper.serial.parity = serial.PARITY_NONE
            self.gripper.serial.stopbits = 1
            self.gripper.serial.timeout = 0.5
            self.gripper.mode = minimalmodbus.MODE_RTU
            self.gripper.clear_buffers_before_each_transaction = True
            self.connected = True
            print(f"✅ Hardware Link Active on {port}")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            self.connected = False

    def initialize(self):
        if not self.connected: return
        try:
            print("🔧 Initializing Gripper Hardware (Homing)...")
            self.gripper.write_register(0x0100, 1, functioncode=6)
            time.sleep(2.0)
            self.gripper.write_register(0x0101, 50, functioncode=6)
            self.gripper.write_register(0x0102, 80, functioncode=6)
            print("✅ Gripper Ready")
        except Exception as e:
            print(f"⚠️ Gripper Initialization Failed: {e}")

    def move_to(self, position):
        if not self.connected: return
        try:
            val = int(max(0, min(1000, position)))
            self.gripper.write_register(0x0103, val, functioncode=6)
        except minimalmodbus.SlaveReportedException:
            print("❌ Slave Device Failure reported. Attempting Auto-Recovery...")
            self.initialize()
        except Exception as e:
            print(f"⚠️ Comm Error: {e}")

    def open(self): self.move_to(0)
    def close(self): self.move_to(1000)

# ==============================
# Fixed Robot Poses
# ==============================
SEARCH_POSX = [372.65, -110.73, 552.43, 92.97, -90.17, 90.00]
REST_JOINTS = [-0.00, -1.38, 91.33, -0.01, 90.11, 0.00]

# ==============================
# Scanning Configuration
# ==============================
SCAN_MIN, SCAN_MAX = -90.0, 90.0
SCAN_STEP = 5.0
SCAN_SPEED, SCAN_ACC = 20, 10

# ==============================
# YOLO & RealSense Setup
# ==============================
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "..", ".."))
model = YOLO(os.path.join(project_root, "runs", "detect", "train6", "weights", "best.pt"))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

latest_frame = None
latest_depth_frame = None
frame_lock = threading.Lock()
camera_running = True

def camera_thread():
    global latest_frame, latest_depth_frame
    while camera_running:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame and depth_frame:
            with frame_lock:
                latest_frame = np.asanyarray(color_frame.get_data())
                latest_depth_frame = depth_frame

# ==============================
# Outlier Filtering Helper
# ==============================
def filter_and_average(readings, sigma=2.0):
    """
    Given a list of (X, Y, Z) tuples, discard readings whose value on ANY
    axis is more than `sigma` standard deviations from the mean of that axis,
    then return the per-axis mean of the survivors.

    Returns (X_avg, Y_avg, Z_avg) or None if too few readings survive.
    """
    if len(readings) < 3:
        return None                          # not enough data to judge

    arr = np.array(readings)                 # shape (N, 3)
    means = arr.mean(axis=0)
    stds  = arr.std(axis=0)

    # Keep rows where every axis is within sigma * std
    mask = np.all(np.abs(arr - means) <= sigma * stds, axis=1)
    clean = arr[mask]

    n_discarded = len(readings) - int(mask.sum())
    print(f"   Depth validation: {len(readings)} readings, "
          f"{n_discarded} discarded as outliers, "
          f"{int(mask.sum())} used for average.")

    if clean.shape[0] < 3:
        print("   ⚠️  Too few clean readings — aborting move.")
        return None

    avg = clean.mean(axis=0)
    print(f"   Averaged coords → X={avg[0]:.3f}  Y={avg[1]:.3f}  Z={avg[2]:.3f} m")
    return float(avg[0]), float(avg[1]), float(avg[2])

# ==============================
# GUI Class
# ==============================
class BananaVisionGUI:
    def __init__(self, root, api):
        self.root = root
        self.root.title("Integrated Banana Vision Control")

        # Robot API
        self.movel = api["movel"]
        self.movej = api["movej"]
        self.posj = api["posj"]
        self.get_current_posx = api["get_current_posx"]
        self.get_current_posj = api["get_current_posj"]
        self.base = api["DR_BASE"]
        self.abs_mode = api["DR_MV_MOD_ABS"]

        # Initialize DH Gripper
        self.dh_gripper = DHGripperUSB(port="/dev/ttyUSB0")
        threading.Thread(target=self.dh_gripper.initialize, daemon=True).start()

        self.is_moving     = False
        self.scanning      = False
        self.target_locked = False
        self.MIN_UNRIPE_CONF = 0.50

        # ── Depth-validation state ──────────────────────────────────────────
        # When a banana is first spotted we don't move immediately. Instead we
        # gather readings for COLLECT_SECS seconds, reject outliers, average
        # the survivors and only then trigger the move.
        self.collecting       = False   # True while we are in the gather phase
        self.banana_readings  = []      # list of (X, Y, Z) raw readings (metres)
        self.collection_start = None    # time.time() when gathering started
        self.COLLECT_SECS     = 2.0    # how long to gather readings
        self.OUTLIER_SIGMA    = 2.0    # readings > N σ from mean are discarded

        self.cam_label = tk.Label(self.root)
        self.cam_label.pack(padx=10, pady=10)

        self.cart_labels  = []
        self.joint_labels = []

        self.build_gui()
        self.update_camera()
        self.update_display()

    def build_gui(self):
        bf = tk.Frame(self.root)
        bf.pack(pady=5)
        tk.Button(bf, text="🔍 SEARCH", width=14, command=self.go_search).pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="✊ GRIP TEST", width=14, command=self.manual_grip, fg="blue").pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="💤 REST", width=14, command=self.go_rest, fg="red").pack(side=tk.LEFT, padx=5)

        pf = tk.LabelFrame(self.root, text="Cartesian Position")
        pf.pack(padx=10, pady=5)
        for i, name in enumerate(["X", "Y", "Z", "RX", "RY", "RZ"]):
            tk.Label(pf, text=name).grid(row=0, column=i)
            lbl = tk.Label(pf, text="0.00", width=10)
            lbl.grid(row=1, column=i); self.cart_labels.append(lbl)

        jf = tk.LabelFrame(self.root, text="Joint Position")
        jf.pack(padx=10, pady=5)
        for i in range(6):
            tk.Label(jf, text=f"J{i+1}").grid(row=i, column=0)
            lbl = tk.Label(jf, text="0.00", width=8)
            lbl.grid(row=i, column=1); self.joint_labels.append(lbl)

    # ------------------------------------------------------------------
    # Depth-validation helpers
    # ------------------------------------------------------------------
    def _start_collection(self, X, Y, Z):
        """Begin a fresh 2-second collection phase."""
        print("🕐 Banana detected — pausing to validate depth readings...")
        self.collecting       = True
        self.banana_readings  = [(X, Y, Z)]
        self.collection_start = time.time()

    def _add_reading(self, X, Y, Z):
        """Append a reading during the collection phase."""
        self.banana_readings.append((X, Y, Z))

    def _finish_collection(self):
        """
        Called once the collection window has elapsed.
        Filters outliers, averages survivors, and triggers move_to_banana
        (or resets state if the data is too noisy).
        """
        self.collecting = False
        print(f"📊 Collection complete — {len(self.banana_readings)} readings gathered.")

        result = filter_and_average(self.banana_readings, sigma=self.OUTLIER_SIGMA)
        self.banana_readings = []

        if result is None:
            # Bad data — reset so we can try again on the next detection
            print("🔄 Resetting detection state to try again.")
            return

        X_avg, Y_avg, Z_avg = result
        self.move_to_banana(X_avg, Y_avg, Z_avg)

    # ------------------------------------------------------------------
    # Scanning / movement
    # ------------------------------------------------------------------
    def scan_j5(self):
        angle, direction = SCAN_MIN, 1
        while self.scanning:
            joints = self.normalize_joints(self.get_current_posj())
            joints[4] = angle
            self.movej(self.posj(*joints), vel=SCAN_SPEED, acc=SCAN_ACC)
            angle += SCAN_STEP * direction
            if angle >= SCAN_MAX or angle <= SCAN_MIN: direction *= -1

    def camera_to_base(self, X_cam, Y_cam, Z_cam):
        """Transform object point from camera frame → robot base frame (mm)."""
        tcp         = self.get_current_posx()[0]
        T_base_tool = doosan_tcp_to_matrix(tcp)
        P_cam       = np.array([X_cam, Y_cam, Z_cam, 1.0])
        P_base      = T_base_tool @ T_TOOL_CAM @ P_cam
        return P_base[0]*1000, P_base[1]*1000, P_base[2]*1000

    def go_search(self):
        if self.is_moving: return
        self.is_moving = True
        def task():
            self.movel(SEARCH_POSX, v=[15, 15], a=[10, 10], ref=self.base, mod=self.abs_mode)
            self.scanning = True
            threading.Thread(target=self.scan_j5, daemon=True).start()
            self.is_moving = False
        threading.Thread(target=task, daemon=True).start()

    def manual_grip(self):
        def task():
            print("🔧 Manual Grip Cycle...")
            self.dh_gripper.close()
            time.sleep(1.5); self.dh_gripper.open()
        threading.Thread(target=task, daemon=True).start()

    def go_rest(self):
        self.scanning   = False
        self.collecting = False          # abort any pending collection
        self.banana_readings = []
        self.is_moving  = True
        def task():
            try:
                print("🔧 Gripper Testing...")
                self.dh_gripper.close()
                time.sleep(1.5); self.dh_gripper.open(); time.sleep(1.5)
            except: pass
            print("💤 Moving to Rest Pose...")
            self.movej(self.posj(*REST_JOINTS), vel=15, acc=10)
            self.is_moving = False
        threading.Thread(target=task, daemon=True).start()

    def move_to_banana(self, X, Y, Z):
        if self.is_moving:
            return

        dist = np.sqrt(X**2 + Y**2 + Z**2)
        print(f"Banana target | camera coords: X={X:.3f} Y={Y:.3f} Z={Z:.3f} | dist={dist:.3f}m")

        # Close enough — grip it
        if dist < 0.12:
            self.target_locked = True
            self.scanning      = False
            print("✅ Within grip range — locking and gripping.")
            threading.Thread(target=self._grip_and_rest, daemon=True).start()
            return

        self.scanning  = False
        self.is_moving = True

        def task():
            try:
                tx, ty, tz = self.camera_to_base(X, Y, Z)
                print(f"Target in base frame: X={tx:.1f}  Y={ty:.1f}  Z={tz:.1f} mm")

                STEP_MM = 50.0
                STEP_V  = [15, 15]
                STEP_A  = [10, 10]

                current = list(self.get_current_posx()[0])
                cx, cy, cz = current[0], current[1], current[2]
                orient = current[3:]

                while True:
                    
                    dx, dy, dz = tx - cx, ty - cy, tz - cz
                    dist_mm = math.sqrt(dx**2 + dy**2 + dz**2)
                    print(f"  → stepping | remaining: {dist_mm:.1f} mm")

                    if dist_mm <= (STEP_MM + 150):
                        print("Final move")
                        # Back off 150 mm along the approach direction so the
                        # end effector tip (not the flange) reaches the target
                        approach_dist = max(dist_mm - 150.0, 0.0)
                        if approach_dist > 1.0:          # only move if there's meaningful distance left
                            scale = approach_dist / dist_mm
                            fx = cx + dx * scale
                            fy = cy + dy * scale
                            fz = cz + dz * scale
                            self.movel([fx, fy, fz] + orient, v=STEP_V, a=STEP_A,
                                    ref=self.base, mod=self.abs_mode)
                        break

                    scale = STEP_MM / dist_mm
                    nx = cx + dx * scale
                    ny = cy + dy * scale
                    nz = cz + dz * scale
                    self.movel([nx, ny, nz] + orient, v=STEP_V, a=STEP_A,
                               ref=self.base, mod=self.abs_mode)

                    current = list(self.get_current_posx()[0])
                    cx, cy, cz = current[0], current[1], current[2]
                    orient = current[3:]

                print("✊ Opening gripper...")
                self.dh_gripper.open()
                time.sleep(0.8)

                print("✊ Closing gripper (gripping banana)...")
                self.dh_gripper.close()
                time.sleep(1.0)

                # ── Return to rest while holding the banana ─────────────────
                print("💤 Gripped — returning to rest pose...")
                self.target_locked = True
                self.movej(self.posj(*REST_JOINTS), vel=20, acc=10)
                print("✅ At rest. Grip complete.")
            except Exception as e:
                print(f"⚠️ move_to_banana error: {e}")
            finally:
                self.is_moving = False

        threading.Thread(target=task, daemon=True).start()

    def _grip_and_rest(self):
        """Grip the banana then return to rest."""
        print("✊ Closing gripper...")
        self.dh_gripper.close()
        time.sleep(1.5)
        print("💤 Going to rest...")
        self.go_rest()

    # ------------------------------------------------------------------
    # Camera loop — detection + collection gate
    # ------------------------------------------------------------------
    def update_camera(self):
        global latest_frame, latest_depth_frame
        with frame_lock:
            frame   = None if latest_frame is None else latest_frame.copy()
            depth_f = latest_depth_frame

        if frame is not None and depth_f is not None:
            results = model(frame, verbose=False)[0]
            intr    = depth_f.profile.as_video_stream_profile().intrinsics

            banana_this_frame = None   # best unripe detection this frame

# check while going ig ion no
# open before going
# close then move

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth  = depth_f.get_distance(cx, cy)
                if depth <= 0 or depth >= 2.0:
                    print(f"here {depth}")
                    continue
                X, Y, Z   = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth)
                label     = model.names[int(box.cls[0])]
                conf      = float(box.conf[0])
                is_unripe = "ripe" in label.lower() or "unripe" in label.lower()
                color     = (0, 0, 255) if is_unripe else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_unripe and conf >= self.MIN_UNRIPE_CONF and not self.target_locked:
                    banana_this_frame = (X, Y, -Z)   # take first/best qualifying box
                    break

            # ── Collection-phase state machine ──────────────────────────────
            if banana_this_frame is not None and not self.is_moving and not self.target_locked:
                X, Y, Z = banana_this_frame

                if not self.collecting:
                    self.scanning  = False
                    self.is_moving = True

                    # First sighting — start the gather window
                    self._start_collection(X, Y, Z)
                    self.is_moving = False


                else:
                    # Already collecting — add this frame's reading
                    self._add_reading(X, Y, Z)

                    # Check whether the window has elapsed
                    elapsed = time.time() - self.collection_start
                    remaining = self.COLLECT_SECS - elapsed
                    # Overlay a countdown so the operator can see it
                    cv2.putText(frame,
                                f"Validating depth... {remaining:.1f}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 165, 255), 2)

                    if elapsed >= self.COLLECT_SECS:
                        self._finish_collection()

            elif banana_this_frame is None and self.collecting:
                # Banana disappeared mid-collection — reset quietly
                print("⚠️  Banana lost during collection — resetting.")
                self.collecting      = False
                self.banana_readings = []

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.cam_label.imgtk = img
            self.cam_label.config(image=img)

        self.root.after(30, self.update_camera)

    def normalize_joints(self, raw):
        if isinstance(raw, tuple): raw = raw[0]
        if hasattr(raw, "j1"): return [raw.j1, raw.j2, raw.j3, raw.j4, raw.j5, raw.j6]
        return list(raw)

    def update_display(self):
        try:
            posx   = self.get_current_posx()[0]
            for i in range(6): self.cart_labels[i].config(text=f"{posx[i]:.2f}")
            joints = self.normalize_joints(self.get_current_posj())
            for i in range(6): self.joint_labels[i].config(text=f"{joints[i]:.2f}")
        except: pass
        self.root.after(500, self.update_display)

def main():
    global camera_running
    rclpy.init()
    node = rclpy.create_node("banana_gui", namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    from DSR_ROBOT2 import (movel, movej, posj, get_current_posx, get_current_posj,
                             set_velx, set_accx, set_robot_mode, DR_BASE, DR_MV_MOD_ABS, ROBOT_MODE_AUTONOMOUS)
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    set_velx(5, 5); set_accx(10, 10)
    threading.Thread(target=camera_thread, daemon=True).start()
    root = tk.Tk()
    BananaVisionGUI(root, locals())
    try: root.mainloop()
    finally:
        camera_running = False
        pipeline.stop(); rclpy.shutdown()

if __name__ == "__main__":
    main()