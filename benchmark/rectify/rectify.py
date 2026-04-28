"""Stereo rectification.

The remap tables only depend on the calibration JSON, so they're built once
at import time and cached per camera. ``rectify(img, cam_id)`` is then a
single ``cv2.remap`` call. (The original implementation re-parsed the JSON
and recomputed the maps for every frame — a 100x speed-up here.)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
CALIB_JSON = ASSETS / "calib.json"


def _euler_xyz_to_R(deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [          0, 1,          0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [         0,           0, 1]])
    return Rz @ Ry @ Rx


def _intrinsics(cam: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array(cam["dist_coeffs"], dtype=np.float64)
    return K, D, (cam["width"], cam["height"])


def _build_maps() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not CALIB_JSON.exists():
        raise FileNotFoundError(f"missing rectify calibration: {CALIB_JSON}")
    front = json.loads(CALIB_JSON.read_text())["front"]

    R = _euler_xyz_to_R(np.array(front["rotation_in_degree"]))
    T = np.array(front["translation"])
    cams = {c["id"]: c for c in front.get("cameras", [])}
    K1, D1, sz1 = _intrinsics(cams[0])
    K2, D2, sz2 = _intrinsics(cams[1])

    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        K1, D1, K2, D2, sz1, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, sz1, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, sz2, cv2.CV_32FC1)
    # cam_id 2 -> top camera (id 0); cam_id 3 -> bottom camera (id 1)
    return {2: map1, 3: map2}


_MAPS = _build_maps()


def rectify(img: np.ndarray, cam_id: int) -> np.ndarray:
    maps = _MAPS.get(cam_id)
    if maps is None:
        raise ValueError(f"no rectify map for cam_id={cam_id}")
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
