"""Thread 1: verify the camera sees the screen and estimate camera->screen homography.

Pipeline (~1-2 FPS):
  mss capture (monitor / active window)
    -> grayscale, downscale, CLAHE
    -> ORB keypoints/descriptors on screen image and camera frame
    -> BFMatcher (Hamming) or FLANN-LSH knn + Lowe ratio test
    -> cv2.findHomography(RANSAC)  [screen -> camera]
    -> validate, invert, and fold capture offset/scale in so that the result maps
       *camera pixels -> absolute screen pixels* directly.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from shared import (DEBUG_WINDOW_TITLE, Config, FrameBuffer, ScreenRegion,
                    SharedState, report_error)

try:  # PyGetWindow raises NotImplementedError on unsupported platforms
    import pygetwindow as gw
except Exception:  # noqa: BLE001
    gw = None


# --------------------------------------------------------------------------- #
# Geometry helpers (also used by main.py and hand_tracker.py)
# --------------------------------------------------------------------------- #
def _signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0].astype(np.float64), q[:, 1].astype(np.float64)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


_UNIT_SIGN = np.sign(_signed_area(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)))


def is_valid_quad(quad: np.ndarray, frame_w: int, frame_h: int, cfg: Config) -> bool:
    """Quad must be finite, convex, same winding as TL-TR-BR-BL (not mirrored), sensibly sized."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(q)):
        return False
    if np.abs(q).max() > 20.0 * max(frame_w, frame_h):
        return False
    contour = q.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return False
    area = _signed_area(q)
    if np.sign(area) != _UNIT_SIGN:
        return False
    frame_area = float(frame_w * frame_h)
    if not (cfg.min_quad_area_frac * frame_area <= abs(area) <= cfg.max_quad_area_frac * frame_area):
        return False
    edges = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
    return bool(edges.min() >= 15.0)


def safe_inverse(H: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Invert a homography, returning None for singular / ill-conditioned input."""
    if H is None or not np.all(np.isfinite(H)):
        return None
    H = np.asarray(H, dtype=np.float64)
    if abs(np.linalg.det(H)) < 1e-12 or np.linalg.cond(H) > 1e10:
        return None
    Hi = np.linalg.inv(H)
    if not np.all(np.isfinite(Hi)) or abs(Hi[2, 2]) < 1e-12:
        return None
    return Hi / Hi[2, 2]


def project_point(M: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    """cv2.perspectiveTransform for one point, guarding the w ~ 0 case.

    OpenCV silently returns (0, 0) when the homogeneous w is ~0, which would teleport
    the cursor to the screen's corner, so w is checked explicitly first.
    """
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    if abs(w) < 1e-6:
        return None
    out = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), np.asarray(M, dtype=np.float64))
    px, py = float(out[0, 0, 0]), float(out[0, 0, 1])
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    return px, py


def primary_monitor_region() -> ScreenRegion:
    import mss
    with mss.mss() as sct:
        m = sct.monitors[1]
        return ScreenRegion(m["left"], m["top"], m["width"], m["height"])


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class ScreenMatcherWorker(threading.Thread):
    def __init__(self, frames: FrameBuffer, state: SharedState, cfg: Config,
                 stop_event: threading.Event) -> None:
        super().__init__(name="ScreenMatcher", daemon=True)
        self.frames, self.state, self.cfg, self.stop_event = frames, state, cfg, stop_event
        self._last_window_region: Optional[ScreenRegion] = None
        self._init_cv()

    def _init_cv(self) -> None:
        cfg = self.cfg
        self._orb = cv2.ORB_create(nfeatures=cfg.orb_features, scaleFactor=1.2,
                                   nlevels=8, fastThreshold=10)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if cfg.matcher == "flann":
            # LSH index is the FLANN option for binary (ORB) descriptors.
            self._matcher = cv2.FlannBasedMatcher(
                dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
                dict(checks=50))
        else:
            self._matcher = self._bf

    # ---- main loop ----
    def run(self) -> None:
        import mss  # mss handles are thread-local: create inside this thread
        period = 1.0 / max(self.cfg.matcher_fps, 0.1)
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                try:
                    self._cycle(sct)
                except Exception:  # noqa: BLE001
                    report_error("ScreenMatcher")
                self.stop_event.wait(max(0.0, period - (time.perf_counter() - t0)))

    def _cycle(self, sct) -> None:
        item = self.frames.get_latest()
        if item is None:
            return
        frame = item[0]
        t0 = time.perf_counter()

        region = self._resolve_region(sct)
        self.state.set_region(region)

        shot = np.asarray(sct.grab({"left": region.left, "top": region.top,
                                    "width": region.width, "height": region.height}))
        screen_gray, _ = self._prep(cv2.cvtColor(shot, cv2.COLOR_BGRA2GRAY), self.cfg.screen_max_width)
        cam_gray, cam_scale = self._prep(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.cfg.cam_max_width)

        result, stats = self._match(screen_gray, cam_gray, cam_scale, frame.shape, region)
        stats["ms"] = (time.perf_counter() - t0) * 1000.0
        if result is None:
            self.state.update_stats(**stats)
        else:
            self.state.set_auto(result["M"], result["quad"], result["region"], stats)

    # ---- helpers ----
    def _prep(self, gray: np.ndarray, max_w: int) -> Tuple[np.ndarray, float]:
        h, w = gray.shape[:2]
        scale = 1.0
        if w > max_w:
            scale = max_w / float(w)
            gray = cv2.resize(gray, (max_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        if self.cfg.use_clahe:
            gray = self._clahe.apply(gray)
        return gray, scale

    def _resolve_region(self, sct) -> ScreenRegion:
        mon = sct.monitors[1]
        fallback = ScreenRegion(mon["left"], mon["top"], mon["width"], mon["height"])
        if self.cfg.target != "window" or gw is None:
            return fallback
        keep = self._last_window_region or fallback
        try:
            win = gw.getActiveWindow()
            if win is None:
                return keep
            title = getattr(win, "title", "") or ""
            # Never target our own debug window (it would show the camera feed -> feedback loop).
            if DEBUG_WINDOW_TITLE in title or getattr(win, "isMinimized", False):
                return keep
            virt = sct.monitors[0]
            l = max(int(win.left), virt["left"])
            t = max(int(win.top), virt["top"])
            r = min(int(win.left + win.width), virt["left"] + virt["width"])
            b = min(int(win.top + win.height), virt["top"] + virt["height"])
            if r - l < 200 or b - t < 150:
                return keep
            self._last_window_region = ScreenRegion(l, t, r - l, b - t)
            return self._last_window_region
        except Exception:  # noqa: BLE001
            return keep

    def _ratio_matches(self, des_s: np.ndarray, des_c: np.ndarray):
        try:
            pairs = self._matcher.knnMatch(des_s, des_c, k=2)
        except cv2.error:  # FLANN-LSH can choke on tiny descriptor sets
            pairs = self._bf.knnMatch(des_s, des_c, k=2)
        # knnMatch may return <2 neighbours per query: skip those
        return [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < self.cfg.ratio_test * p[1].distance]

    def _match(self, screen_gray: np.ndarray, cam_gray: np.ndarray, cam_scale: float,
               frame_shape, region: ScreenRegion) -> Tuple[Optional[Dict], Dict[str, float]]:
        cfg = self.cfg
        stats: Dict[str, float] = {"kp_screen": 0, "kp_cam": 0, "good": 0, "inliers": 0}

        kp_s, des_s = self._orb.detectAndCompute(screen_gray, None)
        kp_c, des_c = self._orb.detectAndCompute(cam_gray, None)
        stats["kp_screen"], stats["kp_cam"] = len(kp_s or []), len(kp_c or [])
        if des_s is None or des_c is None or len(kp_s) < 20 or len(kp_c) < 20:
            return None, stats

        good = self._ratio_matches(des_s, des_c)
        stats["good"] = len(good)
        if len(good) < cfg.min_good_matches:
            return None, stats

        src = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # camera keypoints are rescaled back to full-resolution camera pixels
        dst = (np.float32([kp_c[m.trainIdx].pt for m in good]) / cam_scale).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_thresh)  # screen -> camera
        if H is None or mask is None or not np.all(np.isfinite(H)):
            return None, stats
        inliers = int(mask.sum())
        stats["inliers"] = inliers
        if inliers < cfg.min_inliers or inliers < cfg.min_inlier_ratio * len(good):
            return None, stats

        dh, dw = screen_gray.shape[:2]
        corners = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float64)
        w = H[2, 0] * corners[:, 0] + H[2, 1] * corners[:, 1] + H[2, 2]
        if np.any(np.abs(w) < 1e-6):  # corner at infinity -> degenerate
            return None, stats
        quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H.astype(np.float64)).reshape(4, 2)
        if not is_valid_quad(quad, frame_shape[1], frame_shape[0], cfg):
            return None, stats

        H_c2s = safe_inverse(H)  # camera -> downscaled screen image
        if H_c2s is None:
            return None, stats

        # camera px -> downscaled screen px -> region px -> absolute screen px
        sx, sy = dw / float(max(1, region.width)), dh / float(max(1, region.height))
        S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
        T = np.array([[1, 0, region.left], [0, 1, region.top], [0, 0, 1]], dtype=np.float64)
        M = T @ S_inv @ H_c2s
        if not np.all(np.isfinite(M)) or abs(M[2, 2]) < 1e-12:
            return None, stats
        M = M / M[2, 2]
        return {"M": M, "quad": quad, "region": region}, stats
