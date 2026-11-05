from __future__ import annotations

import cv2
import numpy as np


def laser_score(image: np.ndarray, cfg: dict) -> np.ndarray:
    b, g, r = cv2.split(image.astype(np.float32))
    mode = cfg.get("score_mode", "blue_minus_max")
    if mode == "blue_gated_intensity":
        gate = np.clip(np.maximum(b - g, 0) / max(float(cfg.get("blue_gate_threshold", 5)), 1e-6), 0, 1)
        score = ((b + g + r) / 3) * gate
    elif mode == "blue_minus_green":
        score = b - g
    elif mode == "blue_minus_mean":
        score = b - (r + g) / 2
    else:
        score = b - np.maximum(r, g)
    return np.clip(score, 0, 255).astype(np.float32)


def _centroid(score: np.ndarray, threshold: float, minimum: float, axis: str) -> np.ndarray:
    score = np.where(score >= threshold, score, 0)
    points = []
    profiles = score if axis == "row" else score.T
    coordinates = np.arange(profiles.shape[1], dtype=np.float32)
    for index, profile in enumerate(profiles):
        total = profile.sum()
        if total < minimum:
            continue
        center = float(profile @ coordinates / total)
        points.append((center, float(index)) if axis == "row" else (float(index), center))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _steger(score: np.ndarray, sigma: float, threshold: float, axis: str) -> np.ndarray:
    smooth = cv2.GaussianBlur(score, (0, 0), sigma)
    # Normalize the 3x3 kernels to derivatives in pixel coordinates.
    # First derivative gain is 8, second/mixed derivative gain is 4.
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    gxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3, scale=1 / 4)
    gyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3, scale=1 / 4)
    gxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3, scale=1 / 4)
    profiles = smooth if axis == "row" else smooth.T
    points = []
    for index, profile in enumerate(profiles):
        peak = int(np.argmax(profile))
        if profile[peak] < threshold:
            continue
        v, u = (index, peak) if axis == "row" else (peak, index)
        H = np.array([[gxx[v, u], gxy[v, u]], [gxy[v, u], gyy[v, u]]])
        values, vectors = np.linalg.eigh(H)
        if values.min() >= -1e-9:
            continue
        normal = vectors[:, np.argmin(values)]
        gradient = np.array([gx[v, u], gy[v, u]])
        denominator = normal @ H @ normal
        if abs(denominator) < 1e-9:
            continue
        offset = -(gradient @ normal) / denominator
        if abs(offset) <= 1.0:
            points.append((u + offset * normal[0], v + offset * normal[1]))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def extract_centers(image: np.ndarray, cfg: dict, mask: np.ndarray | None = None) -> np.ndarray:
    if cfg.get('method') == 'ridge':
        from .laser_ridge import extract_ridge
        return extract_ridge(image, cfg, mask)
    score = laser_score(image, cfg)
    if mask is not None:
        score = np.where(mask, score, 0).astype(np.float32)
    axis = "column" if str(cfg.get("scan_axis", "row")).startswith("col") else "row"
    if cfg.get("method", "steger") == "centroid":
        return _centroid(score, float(cfg.get("score_threshold", 14)),
                         float(cfg.get("min_intensity", 16)), axis)
    return _steger(score, float(cfg.get("steger_sigma", 2.0)),
                   float(cfg.get("score_threshold", 14)), axis)
