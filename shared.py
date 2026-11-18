"""Shared configuration and thread-safe containers used by all workers."""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

DEBUG_WINDOW_TITLE = "Virtual Mouse Debug"


@dataclass
class Config:
    # ---- camera / UI ----
    cam_index: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    display_scale: float = 0.75

    # ---- screen matcher (thread 1) ----
    target: str = "monitor"        # "monitor" | "window"
    matcher: str = "bf"            # "bf" | "flann"
    matcher_fps: float = 1.5
    orb_features: int = 2500
    screen_max_width: int = 960    # screen capture is downscaled before ORB
    cam_max_width: int = 1280
    use_clahe: bool = True         # local contrast equalisation (helps with glare)
    ratio_test: float = 0.75
    min_good_matches: int = 25
    min_inliers: int = 15
    min_inlier_ratio: float = 0.35
    ransac_thresh: float = 5.0     # px, in camera-frame units
    min_quad_area_frac: float = 0.02
    max_quad_area_frac: float = 4.0
    match_ttl: float = 3.0         # seconds an auto-match stays valid without refresh

    # ---- hand tracking (thread 2) ----
    mp_max_width: int = 640
    model_complexity: int = 1
    det_conf: float = 0.6
    track_conf: float = 0.5
    offscreen_margin: float = 0.05  # fraction of region size tolerated outside the screen

    # ---- one euro filter ----
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0

    # ---- flick gesture ----
    flick_window: int = 5           # frames in the rolling buffer
    flick_vel_thresh: float = 2.5   # palm-sizes / second
    flick_min_delta: float = 0.22   # minimum net change in normalised distance
    flick_max_window_s: float = 0.30
    flick_wrist_still: float = 0.35  # max wrist travel (palm-sizes) over the window
    flick_cooldown: float = 0.40
    flick_direction: str = "either"  # "flex" (curl) | "extend" | "either"
    flick_freeze_s: float = 0.30     # keep cursor still after a click
    min_palm_px: float = 10.0


class ScreenRegion(NamedTuple):
    """Capture rectangle in absolute screen coordinates."""
    left: int
    top: int
    width: int
    height: int

    def corners(self) -> np.ndarray:
        l, t = self.left, self.top
        r, b = l + self.width, t + self.height
        return np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)  # TL TR BR BL


@dataclass
class HomographySnapshot:
    H: Optional[np.ndarray]          # camera px -> absolute screen px (3x3) or None
    quad: Optional[np.ndarray]       # screen corners in camera px (4x2) or None
    region: Optional[ScreenRegion]
    source: str                      # "auto" | "manual" | "none"
    stale_quad: Optional[np.ndarray]  # last known quad, for the red "searching" polygon
    stats: Dict[str, float]
    paused: bool
    calibrating: bool

    @property
    def matched(self) -> bool:
        return self.source != "none" and self.H is not None


class SharedState:
    """Everything the two workers and the UI exchange, behind one lock."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._auto = None    # (H, quad, region, t)
        self._manual = None  # (H, quad, region)
        self._last_quad: Optional[np.ndarray] = None
        self._region: Optional[ScreenRegion] = None
        self._stats: Dict[str, float] = {}
        self._paused = False
        self._calibrating = False

    # ---- writers ----
    def set_auto(self, H: np.ndarray, quad: np.ndarray, region: ScreenRegion, stats: Dict[str, float]) -> None:
        with self._lock:
            self._auto = (H.copy(), quad.copy(), region, time.perf_counter())
            self._last_quad = quad.copy()
            self._stats.update(stats)

    def set_manual(self, H: np.ndarray, quad: np.ndarray, region: ScreenRegion) -> None:
        with self._lock:
            self._manual = (H.copy(), quad.copy(), region)
            self._last_quad = quad.copy()

    def clear_manual(self) -> None:
        with self._lock:
            self._manual = None

    def update_stats(self, **kw: float) -> None:
        with self._lock:
            self._stats.update(kw)

    def set_region(self, region: ScreenRegion) -> None:
        with self._lock:
            self._region = region

    def get_region(self) -> Optional[ScreenRegion]:
        with self._lock:
            return self._region

    def set_calibrating(self, v: bool) -> None:
        with self._lock:
            self._calibrating = v

    def toggle_paused(self) -> None:
        with self._lock:
            self._paused = not self._paused

    # ---- reader ----
    def snapshot(self) -> HomographySnapshot:
        now = time.perf_counter()
        with self._lock:
            auto, manual = self._auto, self._manual
            if auto is not None and now - auto[3] <= self._cfg.match_ttl:
                H, quad, region, src = auto[0], auto[1], auto[2], "auto"
            elif manual is not None:
                H, quad, region, src = manual[0], manual[1], manual[2], "manual"
            else:
                H, quad, region, src = None, None, self._region, "none"
            stats = dict(self._stats)
            if auto is not None:
                stats["auto_age"] = now - auto[3]
            return HomographySnapshot(
                H=None if H is None else H.copy(),
                quad=None if quad is None else quad.copy(),
                region=region,
                source=src,
                stale_quad=None if self._last_quad is None else self._last_quad.copy(),
                stats=stats,
                paused=self._paused,
                calibrating=self._calibrating,
            )


class FrameBuffer:
    """Single-slot 'latest frame' mailbox. Frames are never mutated after put()."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ts = 0.0

    def put(self, frame: np.ndarray, ts: float) -> None:
        with self._cond:
            self._frame, self._ts = frame, ts
            self._seq += 1
            self._cond.notify_all()

    def get_latest(self) -> Optional[Tuple[np.ndarray, int, float]]:
        with self._cond:
            if self._frame is None:
                return None
            return self._frame, self._seq, self._ts

    def wait_for_new(self, last_seq: int, timeout: float = 0.25) -> Optional[Tuple[np.ndarray, int, float]]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq or self._frame is None:
                return None
            return self._frame, self._seq, self._ts


@dataclass
class HandResult:
    ts: float = 0.0
    present: bool = False
    landmarks: Optional[np.ndarray] = None  # (21, 3) normalised
    cursor: Optional[Tuple[float, float]] = None
    status: str = ""
    fps: float = 0.0
    flick_time: float = float("-inf")


class HandResultBox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._res = HandResult()

    def set(self, res: HandResult) -> None:
        with self._lock:
            self._res = res

    def get(self) -> HandResult:
        with self._lock:
            return self._res


_err_times: Dict[str, float] = {}


def report_error(tag: str) -> None:
    """Print the current exception at most once every 5 s per tag."""
    now = time.monotonic()
    if now - _err_times.get(tag, 0.0) > 5.0:
        _err_times[tag] = now
        print(f"[{tag}] unexpected error:\n{traceback.format_exc()}")
