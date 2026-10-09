from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

RE_PREDICT_TIME = re.compile(r'"predictTime"\s*:\s*([-\d.]+(?:[eE][+-]?\d+)?)')

RE_TIMESTAMP_NS = re.compile(r'"timeStampNs"\s*:\s*(\d+)')

RE_INPUT = re.compile(r'"Input"\s*:\s*(\d+)')

RE_HEAD_POSE = re.compile(r'"Head"\s*:\s*\{\s*"pose"\s*:\s*"([^"]+)"')

RE_LEFT_ACTIVE = re.compile(r'"leftHand"\s*:\s*\{\s*"isActive"\s*:\s*(\d+)')
RE_RIGHT_ACTIVE = re.compile(r'"rightHand"\s*:\s*\{\s*"isActive"\s*:\s*(\d+)')

RE_LEFT_HAND_JOINTS = re.compile(
    r'"leftHand"\s*:\s*\{.*?"HandJointLocations"\s*:\s*(\[.*?\])\s*\}',
    re.DOTALL,
)
RE_RIGHT_HAND_JOINTS = re.compile(
    r'"rightHand"\s*:\s*\{.*?"HandJointLocations"\s*:\s*(\[.*?\])\s*\}',
    re.DOTALL,
)

RE_BODY_JOINTS = re.compile(
    r'"Body"\s*:\s*\{\s*"joints"\s*:\s*(\[.*?\])\s*\}',
    re.DOTALL,
)

RE_ONE_HAND_JOINT = re.compile(
    r'\{\s*"p"\s*:\s*"([^"]+)"\s*,\s*"s"\s*:\s*([-\d.,Ee+]+)\s*,\s*"r"\s*:\s*([-\d.,Ee+]+)\s*\}'
)

RE_ONE_BODY_JOINT = re.compile(
    r'\{\s*"p"\s*:\s*"([^"]+)"\s*,\s*"t"\s*:\s*(\d+)'
)

RE_NUMBER_IN_LIST = re.compile(r'-?\d+(?:,\d+)?(?:[eE][+-]?\d+)?')


def parse_pose_string(s: str) -> list[float]:
    return [float(m.replace(",", ".")) for m in RE_NUMBER_IN_LIST.findall(s)]

def parse_number(s: str) -> float:
    """'−0,08517717' → -0.08517717; '[0.5' → 0.5; '1e+18' → 1e18."""
    s = s.strip().strip("[]").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Parsing error: {s!r}")


def _parse_matrix_string(s: str) -> list[float]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace(" ", "")
    if not s:
        return []
    return [float(x) for x in s.split(",") if x]


def parse_first_line_metadata(line: str) -> dict:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"WARN: first line is not json: {e}")
        data = {}

    meta = {
        "timeStampNs": int(data.get("timeStampNs", 0)),
        "notice": data.get("notice", ""),
    }

    extr = data.get("cameraExtrinsics")
    if isinstance(extr, str) and "|" in extr:
        left, right = extr.split("|", 1)
        meta["cameraExtrinsics_left"] = _parse_matrix_string(left)
        meta["cameraExtrinsics_right"] = _parse_matrix_string(right)

    intr = data.get("cameraIntrinsics")
    if isinstance(intr, str):
        meta["cameraIntrinsics"] = _parse_matrix_string(intr)

    return meta


def extract_hand_joints(block: str, regex) -> np.ndarray | None:
    m = regex.search(block)
    if not m:
        return None
    joints_str = m.group(1)
    matches = RE_ONE_HAND_JOINT.findall(joints_str)
    if not matches:
        return None
    arr = np.zeros((len(matches), 8), dtype=np.float64)
    for i, (p_str, s_str, _r_str) in enumerate(matches):
        vals = parse_pose_string(p_str)
        if len(vals) != 7:
            continue
        arr[i, :7] = vals
        arr[i, 7] = parse_number(s_str)
    return arr


def extract_body_joints(block: str) -> np.ndarray | None:
    m = RE_BODY_JOINTS.search(block)
    if not m:
        return None
    joints_str = m.group(1)
    matches = RE_ONE_BODY_JOINT.findall(joints_str)
    if not matches:
        return None
    arr = np.zeros((len(matches), 8), dtype=np.float64)
    for i, (p_str, t_str) in enumerate(matches):
        vals = parse_pose_string(p_str)
        if len(vals) != 7:
            continue
        arr[i, :7] = vals
        arr[i, 7] = float(t_str)
    return arr


def parse_file(path: Path) -> dict:
    print(f"Counting rows in {path.name} ...")
    n_lines = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n_lines += 1
    print(f"Count of rows: {n_lines}")

    predict_time = []
    timestamp_ns = []
    input_field = []
    head_pose = []
    head_status = []
    left_active = []
    right_active = []
    left_hand = []   
    right_hand = []
    body_joints = []  

    camera_meta = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(tqdm(f, total=n_lines, desc="Parsing")):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue

            if i == 0:
                camera_meta = parse_first_line_metadata(line)
                continue

            m = RE_PREDICT_TIME.search(line)
            predict_time.append(parse_number(m.group(1)) if m else np.nan)

            m = RE_TIMESTAMP_NS.search(line)
            timestamp_ns.append(int(m.group(1)) if m else 0)

            m = RE_INPUT.search(line)
            input_field.append(int(m.group(1)) if m else -1)

            m = RE_HEAD_POSE.search(line)
            if m:
                pose = parse_pose_string(m.group(1))
                head_pose.append(pose if len(pose) == 7 else [np.nan] * 7)
            else:
                head_pose.append([np.nan] * 7)

            m = re.search(r'"Head"\s*:\s*\{[^}]*"status"\s*:\s*(\d+)', line)
            head_status.append(int(m.group(1)) if m else -1)

            m = RE_LEFT_ACTIVE.search(line)
            left_active.append(int(m.group(1)) if m else 0)
            m = RE_RIGHT_ACTIVE.search(line)
            right_active.append(int(m.group(1)) if m else 0)

            lh = extract_hand_joints(line, RE_LEFT_HAND_JOINTS)
            left_hand.append(lh if lh is not None else np.full((26, 8), np.nan))

            rh = extract_hand_joints(line, RE_RIGHT_HAND_JOINTS)
            right_hand.append(rh if rh is not None else np.full((26, 8), np.nan))

            bj = extract_body_joints(line)
            body_joints.append(bj)

    body_lens = [b.shape[0] if b is not None else 0 for b in body_joints]
    body_max = max(body_lens) if body_lens else 0
    body_arr = np.full((len(body_joints), body_max, 8), np.nan, dtype=np.float64)
    for i, b in enumerate(body_joints):
        if b is not None:
            body_arr[i, : b.shape[0], :] = b

    result = {
        "predictTime": np.asarray(predict_time, dtype=np.float64),
        "timeStampNs": np.asarray(timestamp_ns, dtype=np.int64),
        "Input": np.asarray(input_field, dtype=np.int32),
        "head_pose": np.asarray(head_pose, dtype=np.float64),
        "head_status": np.asarray(head_status, dtype=np.int32),
        "left_active": np.asarray(left_active, dtype=np.int32),
        "right_active": np.asarray(right_active, dtype=np.int32),
        "left_hand": np.asarray(left_hand, dtype=np.float64),
        "right_hand": np.asarray(right_hand, dtype=np.float64),
        "body_joints": body_arr,
    }
    if camera_meta:
        for k, v in camera_meta.items():
            if isinstance(v, list):
                result[f"meta_{k}"] = np.asarray(v, dtype=np.float64)
            elif isinstance(v, str):
                result[f"meta_{k}"] = np.array(v)
            else:
                result[f"meta_{k}"] = np.array(v)
    return result


def main():
    ap = argparse.ArgumentParser(description="Data parser")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to tracking data",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tracking.npz"),
        help="Path to output .npz",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"File do not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = parse_file(args.input)

    print(f"\Saving in {args.output} ...")
    np.savez_compressed(args.output, **result)
    print(f"Done. Size: {args.output.stat().st_size / 1e6:.2f} MB")

if __name__ == "__main__":
    main()