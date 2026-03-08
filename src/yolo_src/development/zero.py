#!/usr/bin/env python3
import rclpy
import tkinter as tk
import time
import numpy as np

# ==============================
# Robot Configuration
# ==============================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "a0509"

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class TeachPendantGUI:
    def __init__(self, root, api):
        self.root = root
        self.root.title("Doosan Teach Pendant (SAFE)")

        self.movel = api["movel"]
        self.movej = api["movej"]
        self.posj = api["posj"]
        self.get_current_posx = api["get_current_posx"]
        self.get_current_posj = api["get_current_posj"]
        self.base = api["DR_BASE"]
        self.abs_mode = api["DR_MV_MOD_ABS"]

        self.is_moving = False

        # Saved points: list of dicts
        self.saved_points = []

        # GUI variables
        self.cart_step = tk.DoubleVar(value=10.0)
        self.joint_step = tk.DoubleVar(value=5.0)
        self.loop_count = tk.IntVar(value=1)
        self.play_speed = tk.DoubleVar(value=20.0)

        # Display variables
        self.cart_labels = []
        self.joint_labels = []

        # Home = joint zero
        self.home_joints = [0, 0, 0, 0, 0, 0]

        self.build_gui()
        self.bind_keys()

        # Capture initial position (NO motion)
        self.initial_posx = list(self.get_current_posx()[0])
        self.initial_posj = self.normalize_joints(self.get_current_posj())

        self.update_display()

    # ==============================
    # Helpers
    # ==============================
    def normalize_joints(self, raw):
        if isinstance(raw, tuple):
            raw = raw[0]
        if isinstance(raw, np.ndarray):
            return raw.tolist()
        if hasattr(raw, "j1"):
            return [raw.j1, raw.j2, raw.j3, raw.j4, raw.j5, raw.j6]
        if isinstance(raw, list):
            return raw
        raise TypeError(f"Unknown joint type: {type(raw)}")

    # ==============================
    # GUI
    # ==============================
    def build_gui(self):
        jog = tk.LabelFrame(self.root, text="Cartesian Jog")
        jog.pack(padx=10, pady=5)

        tk.Button(jog, text="↑ Z+", command=lambda: self.jog_xyz(0, 0, +1)).grid(row=0, column=1)
        tk.Button(jog, text="↓ Z-", command=lambda: self.jog_xyz(0, 0, -1)).grid(row=2, column=1)
        tk.Button(jog, text="← Y-", command=lambda: self.jog_xyz(0, -1, 0)).grid(row=1, column=0)
        tk.Button(jog, text="→ Y+", command=lambda: self.jog_xyz(0, +1, 0)).grid(row=1, column=2)
        tk.Button(jog, text="W +X", command=lambda: self.jog_xyz(+1, 0, 0)).grid(row=1, column=3)
        tk.Button(jog, text="S -X", command=lambda: self.jog_xyz(-1, 0, 0)).grid(row=1, column=4)

        stepf = tk.Frame(self.root)
        stepf.pack(pady=3)
        tk.Label(stepf, text="Cartesian step (mm):").pack(side=tk.LEFT)
        tk.Entry(stepf, textvariable=self.cart_step, width=6).pack(side=tk.LEFT)
        tk.Label(stepf, text="   Joint step (deg):").pack(side=tk.LEFT)
        tk.Entry(stepf, textvariable=self.joint_step, width=6).pack(side=tk.LEFT)

        jf = tk.LabelFrame(self.root, text="Joint Jog")
        jf.pack(padx=10, pady=5)

        for i in range(6):
            tk.Label(jf, text=f"J{i+1}").grid(row=i, column=0)
            lbl = tk.Label(jf, text="0.00", width=7)
            lbl.grid(row=i, column=1)
            self.joint_labels.append(lbl)
            tk.Button(jf, text="+", command=lambda i=i: self.jog_joint(i, +1)).grid(row=i, column=2)
            tk.Button(jf, text="-", command=lambda i=i: self.jog_joint(i, -1)).grid(row=i, column=3)

        pf = tk.LabelFrame(self.root, text="Current Cartesian Position")
        pf.pack(padx=10, pady=5)

        for i, name in enumerate(["X", "Y", "Z", "RX", "RY", "RZ"]):
            tk.Label(pf, text=name).grid(row=0, column=i)
            lbl = tk.Label(pf, text="0.00", width=10)
            lbl.grid(row=1, column=i)
            self.cart_labels.append(lbl)

        af = tk.Frame(self.root)
        af.pack(pady=5)

        tk.Button(af, text="HOME (J0)", width=10, command=self.go_home).pack(side=tk.LEFT, padx=5)
        tk.Button(af, text="INITIAL POS", width=12, command=self.go_initial).pack(side=tk.LEFT, padx=5)

        tk.Label(af, text="Loops:").pack(side=tk.LEFT)
        tk.Entry(af, textvariable=self.loop_count, width=4).pack(side=tk.LEFT)

        tk.Label(af, text="Speed (deg/s):").pack(side=tk.LEFT)
        tk.Entry(af, textvariable=self.play_speed, width=5).pack(side=tk.LEFT)

        tk.Button(af, text="Save", command=self.save_point).pack(side=tk.LEFT, padx=5)
        tk.Button(af, text="Play", command=self.play_points).pack(side=tk.LEFT, padx=5)

        self.listbox = tk.Listbox(self.root, width=90)
        self.listbox.pack(pady=5)

    # ==============================
    # Motion
    # ==============================
    def go_home(self):
        if self.is_moving:
            return
        self.is_moving = True
        self.movej(self.posj(*self.home_joints), vel=5, acc=40)
        self.update_display()
        self.is_moving = False

    def go_initial(self):
        if self.is_moving:
            return
        self.is_moving = True
        self.movej(self.posj(*self.initial_posj), vel=5, acc=40)
        self.update_display()
        self.is_moving = False

    def jog_xyz(self, dx, dy, dz):
        if self.is_moving:
            return
        self.is_moving = True

        posx = list(self.get_current_posx()[0])
        step = self.cart_step.get()

        posx[0] += dx * step
        posx[1] += dy * step
        posx[2] += dz * step

        self.movel(posx, v=[15, 15], a=[30, 30],
                   ref=self.base, mod=self.abs_mode)

        self.update_display()
        self.is_moving = False

    def jog_joint(self, idx, direction):
        if self.is_moving:
            return
        self.is_moving = True

        joints = self.normalize_joints(self.get_current_posj())
        joints[idx] += direction * self.joint_step.get()

        self.movej(self.posj(*joints), vel=15, acc=30)
        self.update_display()
        self.is_moving = False

    # ==============================
    # Save / Play  (KEY FIX)
    # ==============================
    def save_point(self):
        posx = list(self.get_current_posx()[0])
        posj = self.normalize_joints(self.get_current_posj())

        self.saved_points.append({
            "posx": posx,
            "posj": posj
        })

        self.listbox.insert(
            tk.END,
            f"P{len(self.saved_points)} → Joints {['%.2f' % j for j in posj]}"
        )

    def play_points(self):
        if self.is_moving or not self.saved_points:
            return

        self.is_moving = True
        loops = max(1, self.loop_count.get())
        speed = max(1.0, self.play_speed.get())

        for _ in range(loops):
            for p in self.saved_points:
                # JOINT replay = exact taught path
                self.movej(
                    self.posj(*p["posj"]),
                    vel=speed,
                    acc=speed * 2
                )
                self.update_display()
                time.sleep(0.05)

        self.is_moving = False

    # ==============================
    # Display
    # ==============================
    def update_display(self):
        posx = list(self.get_current_posx()[0])
        for i in range(6):
            self.cart_labels[i].config(text=f"{posx[i]:.2f}")

        joints = self.normalize_joints(self.get_current_posj())
        for i in range(6):
            self.joint_labels[i].config(text=f"{joints[i]:.2f}")

    # ==============================
    # Keyboard
    # ==============================
    def bind_keys(self):
        self.root.bind("<Up>", lambda e: self.jog_xyz(0, 0, +1))
        self.root.bind("<Down>", lambda e: self.jog_xyz(0, 0, -1))
        self.root.bind("<Left>", lambda e: self.jog_xyz(0, -1, 0))
        self.root.bind("<Right>", lambda e: self.jog_xyz(0, +1, 0))
        self.root.bind("w", lambda e: self.jog_xyz(+1, 0, 0))
        self.root.bind("s", lambda e: self.jog_xyz(-1, 0, 0))


# ==============================
# Main
# ==============================
def main():
    rclpy.init()
    node = rclpy.create_node("dsr_gui", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (
        movel, movej, posj,
        get_current_posx, get_current_posj,
        set_velx, set_accx, set_robot_mode,
        DR_BASE, DR_MV_MOD_ABS, ROBOT_MODE_AUTONOMOUS
    )

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    set_velx(40, 20)
    set_accx(80, 40)

    api = locals()
    root = tk.Tk()
    TeachPendantGUI(root, api)
    root.mainloop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

