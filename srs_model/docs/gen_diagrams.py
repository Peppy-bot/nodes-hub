#!/usr/bin/env python3
"""Generate the srs_model README diagrams.

Reproducible: `python3 docs/gen_diagrams.py` (re)writes `frames.png` and
`arm_angle.png`. The SRS geometry (shoulder/elbow/wrist centers and the joint
axes at q=0) mirrors the constants the Rust crate derives from the URDF, and the
arm poses use the same Product-of-Exponentials forward map, so the figures stay
in sync with the math. Requires matplotlib + numpy.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- SRS geometry in the arm base frame (meters), from the crate's constants ---
S = np.array([0.0, 0.0, 0.1225])    # shoulder center (joints 1,2,3 concurrent)
E = np.array([0.0, 0.220, 0.1225])  # elbow center (on the S-W line)
W = np.array([0.0, 0.436, 0.1225])  # wrist center (joints 5,6,7 concurrent)
L_SU = np.linalg.norm(E - S)        # 0.220
L_UW = np.linalg.norm(W - E)        # 0.216

# Home joint axes (unit) and a point on each axis, in the base frame.
AXES = [np.array(a, float) for a in
        [(0, 0, 1), (-1, 0, 0), (0, 1, 0), (0, 0, -1), (0, 1, 0), (1, 0, 0), (0, 0, 1)]]
PTS = [S, S, S, E, W, W, W]

PURPLE = "#7e3ff2"   # revolute joint axes (frames.png)
BLUE = "#1f77b4"     # upper arm (S->E)
GREEN = "#2ca02c"    # forearm (E->W)
RED = "#e74c3c"      # elbow E
GOLD = "#d4a017"     # psi = 0 reference direction


def rot_about(axis, angle):
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def screw(axis, point, angle, p):
    return rot_about(axis, angle) @ (p - point) + point


def fk_point(p_home, q, upto):
    """Apply the Product-of-Exponentials motion of the proximal joints 0..upto-1."""
    p = p_home.copy()
    for i in reversed(range(upto)):
        p = screw(AXES[i], PTS[i], q[i], p)
    return p


def arm_polyline(q):
    """base -> shoulder -> elbow -> wrist (= EE) at configuration q, base frame."""
    return np.array([
        np.zeros(3),       # arm base (link0 origin)
        S,                 # shoulder is fixed (j1-3 rotate about it)
        fk_point(E, q, 3),  # elbow moved by j1-3
        fk_point(W, q, 4),  # wrist moved by j1-4
    ])


# world <- arm base: the left arm mounts at xyz=(0,0.031,0.698), rpy=(-pi/2,0,0).
R_WB = rot_about(np.array([1.0, 0, 0]), -np.pi / 2)
T_WB = np.array([0.0, 0.031, 0.698])


def to_world(p):
    return (R_WB @ np.asarray(p).T).T + T_WB


# --------------------------------------------------------------------------- #
# Figure 1: coordinate frames + the arm at two poses, in the world frame.
# --------------------------------------------------------------------------- #
def gen_frames():
    poses = [
        (np.zeros(7), "(1) Zero pose (world frame)\nstraight arm hangs down along world -Z"),
        (np.array([0.4, -0.6, 0.3, 1.2, -0.5, 0.4, 0.2]),
         "(2) Operational pose (world frame)\nq=[0.4,-0.6,0.3,1.2,-0.5,0.4,0.2] rad"),
    ]
    fig = plt.figure(figsize=(11, 5))
    for k, (q, title) in enumerate(poses):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        poly = to_world(arm_polyline(q))
        # arm: base->shoulder (gray), shoulder->elbow (upper arm), elbow->wrist (forearm)
        ax.plot(*poly[0:2].T, color="0.5", lw=2)
        ax.plot(*poly[1:3].T, color=BLUE, lw=3)
        ax.plot(*poly[2:4].T, color=GREEN, lw=3)
        ax.scatter(*poly[1], color="k", s=18)
        ax.scatter(*poly[2], color="k", s=18)
        ax.scatter(*poly[3], color="#f1c40f", s=70, ec="k", zorder=5)
        ax.text(*poly[0], "  arm base", fontsize=8)
        ax.text(*poly[3], "  EE", fontsize=8)
        # world RGB triad at the origin
        for vec, col in zip(np.eye(3) * 0.18, ["r", "g", "b"], strict=True):
            ax.quiver(0, 0, 0, *vec, color=col, lw=1.5, arrow_length_ratio=0.18)
        ax.text(0, 0, -0.04, "world origin", fontsize=8, ha="center")
        # purple revolute joint axes at each joint origin
        for i in range(7):
            o = to_world(fk_point(PTS[i], q, i))
            d = R_WB @ rot_about_chain(q, i) @ AXES[i] * 0.06
            ax.quiver(*o, *d, color=PURPLE, lw=1.2, arrow_length_ratio=0.3)
        # bound the view over the arm AND the world origin/triad, so nothing is cut off
        ax.view_init(elev=18, azim=-65)
        _equal_3d(ax, np.vstack([poly, np.zeros((1, 3)), np.eye(3) * 0.18]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("world X (m)", fontsize=7)
        ax.set_ylabel("world Y (m)", fontsize=7)
        ax.set_zlabel("world Z (m)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.text(0.5, 0.03,
             "RGB triad at the origin = world X (red) / Y (green) / Z (blue) axes.  "
             "Purple arrows = revolute joint axes.  Yellow dot = end-effector.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("frames.png", dpi=130)
    plt.close(fig)


def rot_about_chain(q, i):
    """Rotation of joint i's axis under the proximal joints (for drawing arrows)."""
    R = np.eye(3)
    for j in range(i):
        R = rot_about(AXES[j], q[j]) @ R
    return R


def _equal_3d(ax, pts):
    c = pts.mean(0)
    r = max(np.ptp(pts, 0).max(), 0.5) / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


# --------------------------------------------------------------------------- #
# Figure 2: the arm-angle redundancy (the elbow circle about the S-W line).
# --------------------------------------------------------------------------- #
def elbow_circle(Sp, Wp):
    n = (Wp - Sp) / np.linalg.norm(Wp - Sp)
    d = np.linalg.norm(Wp - Sp)
    h = (L_SU ** 2 - L_UW ** 2 + d ** 2) / (2 * d)
    center = Sp + h * n
    radius = np.sqrt(max(L_SU ** 2 - h ** 2, 0))
    # reference axis least aligned with n, projected into the circle plane
    ref = np.eye(3)[np.argmin(np.abs(n))]
    a_hat = ref - n * (ref @ n)
    a_hat /= np.linalg.norm(a_hat)
    b_hat = np.cross(n, a_hat)
    return center, radius, a_hat, b_hat


def gen_arm_angle():
    psi = np.deg2rad(50)  # a chosen arm angle, for illustration
    fig = plt.figure(figsize=(14, 4.3))
    fig.suptitle(
        "Arm-angle redundancy: the target fixes S, W and θ4; the elbow is then "
        "free to swing about the S-W line by ψ", fontsize=11)

    # Panel 1: 3D circle about a (vertical) S-W line. Use an illustrative *bent*
    # pose: at full reach (the home pose) the circle degenerates to a point.
    Sp, Wp = np.array([0, 0, 0.15]), np.array([0, 0, -0.15])
    center, radius, a_hat, b_hat = elbow_circle(Sp, Wp)
    e_psi = center + radius * (np.cos(psi) * a_hat + np.sin(psi) * b_hat)
    ts = np.linspace(0, 2 * np.pi, 200)
    circle = center + radius * (np.cos(ts)[:, None] * a_hat + np.sin(ts)[:, None] * b_hat)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.view_init(elev=12, azim=-70)
    ax.plot(*circle.T, color="#bbbbbb", lw=1)
    ax.plot(*np.array([Sp, Wp]).T, color="k", ls="--", lw=1)
    ax.plot(*np.array([Sp, e_psi]).T, color="#1f77b4", lw=2)
    ax.plot(*np.array([e_psi, Wp]).T, color="#2ca02c", lw=2)
    refp = center + radius * a_hat
    ax.plot(*np.array([center, refp]).T, color=GOLD, lw=1.5)
    ax.scatter(*Sp, color="k", s=20)
    ax.text(*Sp, "  S (shoulder)", fontsize=8)
    ax.scatter(*Wp, color="k", s=20)
    ax.text(*Wp, "  W (wrist center)", fontsize=8)
    ax.scatter(*e_psi, color=RED, s=25)
    ax.text(*e_psi, "  E (elbow)", fontsize=8, color=RED)
    ax.text(*refp, " ψ=0 ref", fontsize=7, color=GOLD)
    ax.set_title("3D: the elbow rides a circle about the (vertical) S-W line", fontsize=9)
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-0.22, 0.22)
    ax.set_zlim(-0.22, 0.22)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Panel 2: 2D S-E-W triangle (the reach fixes theta4).
    ax = fig.add_subplot(1, 3, 2)
    ax.set_aspect("equal")
    Sp2, Wp2, Ep2 = np.array([0, 1.0]), np.array([0, 0.0]), np.array([0.55, 0.52])
    ax.plot(*np.array([Sp2, Ep2]).T, color="#1f77b4", lw=2.5)
    ax.plot(*np.array([Ep2, Wp2]).T, color="#2ca02c", lw=2.5)
    ax.plot(*np.array([Sp2, Wp2]).T, color="k", ls="--", lw=1)
    ax.scatter(*Sp2, color="k", s=25)
    ax.text(Sp2[0] - 0.05, Sp2[1], "S (shoulder) ", ha="right", fontsize=9)
    ax.scatter(*Wp2, color="k", s=25)
    ax.text(Wp2[0] - 0.05, Wp2[1], "W (wrist center) ", ha="right", fontsize=9)
    ax.scatter(*Ep2, color="#e74c3c", s=25)
    ax.text(Ep2[0] + 0.04, Ep2[1], "E (elbow)", fontsize=9, color="#e74c3c")
    ax.text(*(0.5 * Sp2 + 0.5 * Ep2 + np.array([0.03, 0.03])), "upper arm 0.220",
            fontsize=8, color="#1f77b4", rotation=-42)
    ax.text(*(0.38 * Ep2 + 0.62 * Wp2 + np.array([0.05, -0.02])), "forearm 0.216",
            fontsize=8, color="#2ca02c", rotation=44)
    ax.annotate("θ4", Ep2 + np.array([-0.16, -0.03]), fontsize=10)
    ax.set_title("2D side view: the S-E-W triangle,\nθ4 (elbow flex) is fixed by the "
                 "reach |SW|", fontsize=9)
    ax.set_xlim(-0.55, 0.95)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # Panel 3: 2D view along the S-W line (down the circle's normal).
    ax = fig.add_subplot(1, 3, 3)
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#cccccc", lw=1.2)
    ax.scatter(0, 0, color="#1f77b4", s=40, zorder=5)
    ax.text(0, -0.2, "S, W\n(on axis, into page)", fontsize=8, ha="center")
    ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
    ax.text(1.05, 0, "ψ=0 ref", fontsize=9, color=GOLD, va="center")
    pe = np.array([np.cos(psi), np.sin(psi)])
    ax.annotate("", pe, (0, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.scatter(*pe, color=RED, s=40, zorder=5)
    ax.text(pe[0] + 0.05, pe[1] + 0.08, "E (elbow at chosen ψ)", fontsize=9, color=RED)
    arc = np.linspace(0, psi, 30)
    ax.plot(0.33 * np.cos(arc), 0.33 * np.sin(arc), color="k", lw=1.2)
    ax.text(0.46 * np.cos(psi / 2), 0.46 * np.sin(psi / 2), "ψ", fontsize=14)
    ax.set_title("2D view along S-W: the elbow circle,\nψ (arm angle) is the one free "
                 "parameter", fontsize=9)
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("arm_angle.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gen_frames()
    gen_arm_angle()
    print("wrote frames.png and arm_angle.png")
