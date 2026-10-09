from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import median_filter

def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(np.stack([roll, pitch, yaw], axis=1))


def compute_speed(pos: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(t_sec)
    dt = np.where(dt <= 0, np.nan, dt)
    v = dp / dt
    return np.concatenate([[0.0], v])

    def smooth_positions(pos, window=5):
        out = pos.copy()
        for k in range(3):
            out[:, k] = median_filter(pos[:, k], size=window, mode="nearest")
        return out

    wrist_L_raw = lh[:, 1, :3]
    wrist_R_raw = rh[:, 1, :3]
    wrist_L = smooth_positions(wrist_L_raw, window=5)
    wrist_R = smooth_positions(wrist_R_raw, window=5)

    t_speed = t.copy()
    v_L = compute_speed(wrist_L, t_speed)
    v_R = compute_speed(wrist_R, t_speed)

    v_L = np.where(v_L > 3.0, np.nan, v_L)
    v_R = np.where(v_R > 3.0, np.nan, v_R)

def quat_to_forward(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    fx = -2 * (qx * qz + qw * qy)
    fy = -2 * (qy * qz - qw * qx)
    fz = -1 + 2 * (qx * qx + qy * qy)
    return np.stack([fx, fy, fz], axis=1)

def moving_average(x, w=15):
    return np.convolve(x, np.ones(w) / w, mode="same")

def plot_trajectory(ax, positions, title, color="C0", alpha=0.7):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    ax.plot(x, y, color=color, lw=1, alpha=alpha, label="XY")
    ax.plot(x, z, color="C1", lw=1, alpha=alpha, label="XZ")
    ax.plot(z, y, color="C2", lw=1, alpha=alpha, label="ZY")
    ax.scatter([x[0]], [y[0]], color=color, s=40, marker="o", zorder=5, label="start")
    ax.scatter([x[-1]], [y[-1]], color=color, s=40, marker="x", zorder=5, label="end")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main():
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load("outputs/tracking.npz")

    t_ns = d["timeStampNs"].astype(np.int64)
    t = (t_ns - t_ns[0]) / 1e9

    lh = d["left_hand"]
    rh = d["right_hand"]
    hp = d["head_pose"]
    la = d["left_active"]
    ra = d["right_active"]

    wrist_L = lh[:, 1, :3]
    wrist_R = rh[:, 1, :3]
    head_xyz = hp[:, :3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_trajectory(axes[0], wrist_L, "Left wrist", color="C0")
    plot_trajectory(axes[1], wrist_R, "Right wrist", color="C3")
    plot_trajectory(axes[2], head_xyz, "Head", color="C4")
    fig.suptitle(f"Trajectories {t[-1]:.1f} s (in meters)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "01_trajectories.png", dpi=130)
    plt.close(fig)
    print("  ✓ 01_trajectories.png")

    v_L = compute_speed(wrist_L, t)
    v_R = compute_speed(wrist_R, t)
    v_L_ma = moving_average(np.nan_to_num(v_L, nan=0.0))
    v_R_ma = moving_average(np.nan_to_num(v_R, nan=0.0))

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t[::5], v_L_ma[::5], color="C0", lw=0.9, label="Left wrist")
    ax.plot(t[::5], v_R_ma[::5], color="C3", lw=0.9, alpha=0.8, label="Right wrist")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Speed, m/s")
    ax.set_title("Wrists speed over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_speed.png", dpi=130)
    plt.close(fig)
    print("  ✓ 02_speed.png")

    fig, ax = plt.subplots(figsize=(14, 2.2))
    ax.fill_between(t, 0, la, step="mid", color="C0", alpha=0.6)
    ax.fill_between(t, 0, -ra, step="mid", color="C3", alpha=0.6)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Right active", "off", "Left active"])
    ax.set_xlabel("Time, s")
    ax.set_title("Hand activity")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_activity.png", dpi=130)
    plt.close(fig)
    print("  ✓ 03_activity.png")

    fwd = quat_to_forward(hp[:, 3:7])
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, fwd[:, 0], lw=0.8, label="Forward X")
    ax.plot(t, fwd[:, 1], lw=0.8, label="Forward Y")
    ax.plot(t, fwd[:, 2], lw=0.8, label="Forward Z")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Unit vector component")
    ax.set_title("Gaze direction (head forward vector)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04_head_forward.png", dpi=130)
    plt.close(fig)
    print("  ✓ 04_head_forward.png")

    def finger_angle(hand, mcp, pip, dip):
        v1 = hand[:, pip, :3] - hand[:, mcp, :3]
        v2 = hand[:, dip, :3] - hand[:, pip, :3]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cos = np.einsum("ij,ij->i", v1, v2) / (n1 * n2 + 1e-9)
        cos = np.clip(cos, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos))

    idx_angle = finger_angle(lh, 6, 7, 8)
    mid_angle = finger_angle(lh, 11, 12, 13)
    lit_angle = finger_angle(lh, 21, 22, 23)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, idx_angle, lw=0.9, label="Index")
    ax.plot(t, mid_angle, lw=0.9, label="Middle")
    ax.plot(t, lit_angle, lw=0.9, label="Little")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Angle, °")
    ax.set_title("The curve of the left-hand fingers (MCP → PIP → DIP)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_finger_flexion.png", dpi=130)
    plt.close(fig)
    print("  ✓ 05_finger_flexion.png")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    axes[0, 0].plot(wrist_L[:, 0], wrist_L[:, 1], lw=0.8, color="C0")
    axes[0, 0].set_title("Left wrist: XY")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(wrist_L[:, 0], wrist_L[:, 2], lw=0.8, color="C0")
    axes[0, 1].set_title("Left wrist: XZ")
    axes[0, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(wrist_L[:, 2], wrist_L[:, 1], lw=0.8, color="C0")
    axes[1, 0].set_title("Left wrist: ZY")
    axes[1, 0].set_aspect("equal", adjustable="datalim")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t[::5], v_L_ma[::5], lw=0.7, color="C0", label="Left")
    axes[1, 1].plot(t[::5], v_R_ma[::5], lw=0.7, color="C3", label="Right")
    axes[1, 1].set_title("Wrists speed")
    axes[1, 1].set_xlabel("s")
    axes[1, 1].set_ylabel("m/s")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(t, fwd[:, 0], lw=0.8, label="Forward X")
    axes[2, 0].plot(t, fwd[:, 1], lw=0.8, label="Forward Y")
    axes[2, 0].plot(t, fwd[:, 2], lw=0.8, label="Forward Z")
    axes[2, 0].set_title("Gaze direction")
    axes[2, 0].set_xlabel("s")
    axes[2, 0].set_ylabel("axis")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(t, idx_angle, lw=0.8, label="Index")
    axes[2, 1].plot(t, mid_angle, lw=0.8, label="Middle")
    axes[2, 1].plot(t, lit_angle, lw=0.8, label="Little")
    axes[2, 1].set_title("Fingers curve")
    axes[2, 1].set_xlabel("s")
    axes[2, 1].set_ylabel("°")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    fig.suptitle("XR Tracking — Dashboard", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "00_dashboard.png", dpi=140)
    plt.close(fig)
    print("  ✓ 00_dashboard.png")

    print(f"\nSaving in {out_dir}/")


if __name__ == "__main__":
    main()