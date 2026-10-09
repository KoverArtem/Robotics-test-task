from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

HAND_BONES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]
EMPTY: list = []

DEFAULT_CAMERA = dict(
    eye=dict(x=0.0, y=1.1, z=-2.4),
    center=dict(x=0.0, y=-0.5, z=0.3),
    up=dict(x=0, y=1, z=0),
)

GAZE_LENGTH = 0.4

def hand_traces(hand, color: str, name: str, valid: bool):
    if valid:
        pos = hand[:, :3]
        xs, ys, zs = [], [], []
        for a, b in HAND_BONES:
            xs += [float(pos[a, 0]), float(pos[b, 0]), None]
            ys += [float(pos[a, 1]), float(pos[b, 1]), None]
            zs += [float(pos[a, 2]), float(pos[b, 2]), None]
        return [
            dict(type="scatter3d", mode="lines",
                 x=xs, y=ys, z=zs,
                 line=dict(color=color, width=6),
                 name=f"{name} bones", showlegend=True, hoverinfo="skip"),
            dict(type="scatter3d", mode="markers",
                 x=pos[:, 0].tolist(),
                 y=pos[:, 1].tolist(),
                 z=pos[:, 2].tolist(),
                 marker=dict(color=color, size=3),
                 name=f"{name} joints", showlegend=False, hoverinfo="skip"),
        ]
    return [
        dict(type="scatter3d", mode="lines", x=EMPTY, y=EMPTY, z=EMPTY,
             line=dict(color=color, width=6),
             name=f"{name} bones", showlegend=True, hoverinfo="skip"),
        dict(type="scatter3d", mode="markers", x=EMPTY, y=EMPTY, z=EMPTY,
             marker=dict(color=color, size=3),
             name=f"{name} joints", showlegend=False, hoverinfo="skip"),
    ]

def body_trace(body, valid: bool):
    if valid:
        pos = body[:, :3]
        return dict(type="scatter3d", mode="markers",
                    x=pos[:, 0].tolist(),
                    y=pos[:, 1].tolist(),
                    z=pos[:, 2].tolist(),
                    marker=dict(color="#888888", size=2.5, opacity=0.5),
                    name="body", showlegend=True, hoverinfo="skip")
    return dict(type="scatter3d", mode="markers", x=EMPTY, y=EMPTY, z=EMPTY,
                marker=dict(color="#888888", size=2.5, opacity=0.5),
                name="body", showlegend=True, hoverinfo="skip")

def head_traces(head_pose, valid: bool):
    if valid:
        x, y, z = float(head_pose[0]), float(head_pose[1]), float(head_pose[2])
        qx, qy, qz, qw = (float(head_pose[3]), float(head_pose[4]),
                          float(head_pose[5]), float(head_pose[6]))

        fx = -2 * (qx * qz + qw * qy)
        fy = -2 * (qy * qz - qw * qx)
        fz = -1 + 2 * (qx * qx + qy * qy)

        L = GAZE_LENGTH
        ex, ey, ez = x + L * fx, y + L * fy, z + L * fz

        return [
            dict(type="scatter3d", mode="markers",
                 x=[x], y=[y], z=[z],
                 marker=dict(color="#ffb000", size=6),
                 name="head", showlegend=True, hoverinfo="skip"),
            dict(type="scatter3d", mode="lines",
                 x=[x, ex], y=[y, ey], z=[z, ez],
                 line=dict(color="#ffb000", width=6),
                 name="gaze", showlegend=True, hoverinfo="skip"),
        ]
    return [
        dict(type="scatter3d", mode="markers", x=EMPTY, y=EMPTY, z=EMPTY,
             marker=dict(color="#ffb000", size=6),
             name="head", showlegend=True, hoverinfo="skip"),
        dict(type="scatter3d", mode="lines", x=EMPTY, y=EMPTY, z=EMPTY,
             line=dict(color="#ffb000", width=6),
             name="gaze", showlegend=True, hoverinfo="skip"),
    ]

def build_frame_traces(lh, rh, bj, hp):
    out = []
    out += hand_traces(lh, "#2a6fdb", "left",
                       lh is not None and not np.isnan(lh[0, 0]))
    out += hand_traces(rh, "#d62728", "right",
                       rh is not None and not np.isnan(rh[0, 0]))
    out.append(body_trace(bj, bj is not None and not np.isnan(bj[0, 0])))
    out += head_traces(hp, hp is not None and not np.isnan(hp[0]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("outputs/tracking.npz"))
    ap.add_argument("--output", type=Path, default=Path("outputs/hand_animation.html"))
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    t_start = time.perf_counter()

    print(f"[1/4] Load {args.input} ...")
    d = np.load(args.input)
    left_hand = d["left_hand"]
    right_hand = d["right_hand"]
    body_joints = d["body_joints"]
    head_pose = d["head_pose"]
    n = len(d["predictTime"])
    d.close()

    idx = np.arange(0, n, args.step)
    if args.max_frames is not None:
        idx = idx[: args.max_frames]
    print(f"      frames: {n}, after downsample: {len(idx)}")

    print(f"[2/4] Build frames ...")
    t0 = time.perf_counter()
    frames = []
    for k, i in enumerate(tqdm(idx, desc="frames", unit="fr")):
        traces = build_frame_traces(left_hand[i], right_hand[i],
                                    body_joints[i], head_pose[i])
        frames.append(dict(name=str(k), data=traces))
    print(f"      done ({time.perf_counter() - t0:.2f} s)")

    print(f"[3/4] Build object ...")
    t0 = time.perf_counter()

    i0 = idx[0]
    base_traces = build_frame_traces(left_hand[i0], right_hand[i0],
                                     body_joints[i0], head_pose[i0])

    frame_duration_ms = max(1, int(1000.0 / args.fps))

    play_button = dict(
        label="Play",
        method="animate",
        args=[None, dict(
            mode="immediate",
            frame=dict(duration=frame_duration_ms, redraw=True),
            fromcurrent=True,
            transition=dict(duration=0),
        )],
    )
    pause_button = dict(
        label="Pause",
        method="animate",
        args=[[None], dict(
            mode="immediate",
            frame=dict(duration=0, redraw=False),
            transition=dict(duration=0),
        )],
    )

    fig = go.Figure(
        data=base_traces,
        frames=frames,
        layout=go.Layout(
            uirevision="static",
            title=dict(text="XR Tracking — 3D hand + head", x=0.5, y=0.98),
            scene=dict(
                xaxis=dict(title="X, м", range=[-0.9, 0.5], autorange=False),
                yaxis=dict(title="Y, м (top)", range=[-1.8, 0.3], autorange=False),
                zaxis=dict(title="Z, м (forward)", range=[-0.6, 0.8], autorange=False),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1.4, z=1),
                dragmode="pan"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.02, y=1.14, xanchor="left", yanchor="top",
                direction="right",
                buttons=[play_button, pause_button],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="frame "),
                pad=dict(t=30),
                steps=[
                    dict(
                        method="animate",
                        args=[[str(k)], dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        )],
                        label=str(k),
                    )
                    for k in range(len(frames))
                ],
            )],
            margin=dict(l=0, r=0, t=120, b=0),
            height=850,
            legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top",
                        bgcolor="rgba(255,255,255,0.85)"),
            annotations=[
                dict(
                    x=0.97, y=0.55,
                    xref="paper", yref="paper",
                    xanchor="left", yanchor="top",
                    showarrow=False, align="left",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#cccccc", borderwidth=1, borderpad=10,
                    font=dict(family="monospace", size=11, color="#333"),
                    text=(
                        "<b>Scene controls</b><br>"
                        "<br>"
                        "<b>Mouse:</b><br>"
                        "  • LMB + drag — pan<br>"
                        "  • Wheel — zoom<br>"
                        "<br>"
                        "<b>Keyboard:</b><br>"
                        "  • ← / → — rotate<br>"
                        "  • R — reset view<br>"
                        "<br>"
                        "<b>Buttons above:</b><br>"
                        "  • Play / Pause<br>"
                        "<br>"
                        "<b>Slider below:</b><br>"
                        "  • Scrub through time"
                    ),
                ),
            ],
        )
    )
    print(f"      Done ({time.perf_counter() - t0:.2f} s)")

    post_script = """
    (function() {
        const gd = document.querySelector('.js-plotly-plot');
        if (!gd) return;

        const DEF_EYE    = {x: 0.0, y: 1.1, z: -2.4};
        const DEF_CENTER = {x: 0.0, y: -0.5, z: 0.3};
        const DEF_UP     = {x: 0, y: 1, z: 0};

        let camEye    = {...DEF_EYE};
        let camCenter = {...DEF_CENTER};
        let camUp     = {...DEF_UP};

        function applyCamera() {
            Plotly.relayout(gd, {
                'scene.camera.eye':    {...camEye},
                'scene.camera.center': {...camCenter},
                'scene.camera.up':     {...camUp},
            });
        }

        setTimeout(applyCamera, 50);

        gd.on('plotly_relayout', function(e) {
            if (e['scene.camera']) {
                const full = gd._fullLayout.scene.camera;
                if (full && full.eye) {
                    camEye    = {...full.eye};
                    camCenter = {...full.center};
                    camUp     = {...full.up};
                }
            }
        });

        gd.on('plotly_animatingframe', function() {
            Plotly.relayout(gd, {
                'scene.camera.eye':    {...camEye},
                'scene.camera.center': {...camCenter},
                'scene.camera.up':     {...camUp},
            });
        });

        function rotateAzimuth(dRad) {
            const dx = camEye.x - camCenter.x;
            const dz = camEye.z - camCenter.z;
            const r  = Math.sqrt(dx*dx + dz*dz);
            const a  = Math.atan2(dz, dx) + dRad;
            camEye = {
                x: camCenter.x + r * Math.cos(a),
                y: camEye.y,                       
                z: camCenter.z + r * Math.sin(a),
            };
            applyCamera();
        }

        function resetCamera() {
            camEye    = {...DEF_EYE};
            camCenter = {...DEF_CENTER};
            camUp     = {...DEF_UP};
            applyCamera();
        }

        document.addEventListener('keydown', function(e) {
            const t = e.target;
            if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
            if (e.key === 'ArrowLeft')       { rotateAzimuth(+0.08); e.preventDefault(); }
            else if (e.key === 'ArrowRight') { rotateAzimuth(-0.08); e.preventDefault(); }
            else if (e.key === 'r' || e.key === 'R' || e.key === 'к' || e.key === 'К') {
                resetCamera();
            }
        });

        setTimeout(function() {
            const buttons = document.querySelectorAll('.updatemenu-button');
            buttons.forEach(function(btn) {
                const txt = (btn.textContent || '').toLowerCase();
                if (txt.includes('сброс') || txt.includes('⟲')) {
                    btn.onclick = function(ev) {
                        ev.stopPropagation();
                        resetCamera();
                    };
                }
            });
        }, 300);
    })();
    """

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Saving in {args.output} ...")
    t0 = time.perf_counter()
    fig.write_html(
        str(args.output),
        include_plotlyjs="cdn",
        auto_play=False,
        post_script=post_script,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "toImage",
                "sendDataToCloud",
                "orbitRotation",
                "tableRotation",
                "resetCameraLastSave3d",
                "resetCameraDefault3d",
                "hoverClosest3d",
            ],
        },
    )
    dt = time.perf_counter() - t0
    size_mb = args.output.stat().st_size / 1e6
    print(f"      Done: {size_mb:.2f} MB in {dt:.1f} s")
    print(f"Open by path: file://{args.output.resolve()}")

if __name__ == "__main__":
    main()