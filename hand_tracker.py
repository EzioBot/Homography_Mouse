"""Thread 2: MediaPipe hand tracking, coordinate mapping, smoothing, flick-to-click."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Button, Controller as MouseController

from filters import OneEuroFilter2D
from screen_matcher import project_point
from shared import (Config, FrameBuffer, HandResult, HandResultBox,
                    HomographySnapshot, SharedState, report_error)

mp_hands = mp.solutions.hands

WRIST, INDEX_MCP, MIDDLE_MCP, INDEX_TIP = 0, 5, 9, 8


# --------------------------------------------------------------------------- #
# Flick detection
# --------------------------------------------------------------------------- #
@dataclass
class FlickEvent:
    t: float
    delta: float
    velocity: float
    anchor: Optional[Tuple[float, float]]  # cursor position from the START of the flick


class FlickDetector:
    """Detects a fast change of (|tip - index MCP| / palm size) while the wrist stays put.

    Palm size = |wrist - middle MCP|, so the signal is scale-invariant (distance from camera).
    Velocity is in palm-sizes per second, computed over a rolling window of `flick_window` frames.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._buf: Deque[tuple] = deque(maxlen=cfg.flick_window)
        self._last_click = float("-inf")

    def reset(self) -> None:
        self._buf.clear()

    def update(self, t: float, px: np.ndarray, anchor: Optional[Tuple[float, float]]) -> Optional[FlickEvent]:
        cfg = self.cfg
        palm = float(np.linalg.norm(px[WRIST] - px[MIDDLE_MCP]))
        if palm < cfg.min_palm_px:  # hand too small / degenerate -> no division by ~0
            self._buf.clear()
            return None

        d = float(np.linalg.norm(px[INDEX_TIP] - px[INDEX_MCP])) / palm
        self._buf.append((t, d, px[WRIST].copy(), anchor))
        if len(self._buf) < cfg.flick_window:
            return None
        if t - self._last_click < cfg.flick_cooldown:  # 400 ms cooldown
            return None

        t0, d0, wrist0, anchor0 = self._buf[0]
        dt = t - t0
        if dt <= 1e-3 or dt > cfg.flick_max_window_s:  # stalled stream or too slow to be a flick
            return None

        delta = d - d0
        velocity = delta / dt
        wrist_travel = float(np.linalg.norm(px[WRIST] - wrist0)) / palm

        direction_ok = (cfg.flick_direction == "either"
                        or (cfg.flick_direction == "flex" and delta < 0)
                        or (cfg.flick_direction == "extend" and delta > 0))
        if (direction_ok
                and abs(delta) >= cfg.flick_min_delta
                and abs(velocity) >= cfg.flick_vel_thresh
                and wrist_travel <= cfg.flick_wrist_still):
            self._last_click = t
            self._buf.clear()
            return FlickEvent(t, delta, velocity, anchor0 if anchor0 is not None else anchor)
        return None


# --------------------------------------------------------------------------- #
# Overlay drawing (used by main thread)
# --------------------------------------------------------------------------- #
def draw_hand(vis: np.ndarray, lm: np.ndarray, flicking: bool) -> None:
    h, w = vis.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    color = (0, 0, 255) if flicking else (0, 255, 255)
    for a, b in mp_hands.HAND_CONNECTIONS:
        cv2.line(vis, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(vis, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, pts[INDEX_TIP], 10, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(vis, pts[INDEX_MCP], 5, (0, 165, 255), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class HandWorker(threading.Thread):
    def __init__(self, frames: FrameBuffer, state: SharedState, result_box: HandResultBox,
                 cfg: Config, stop_event: threading.Event, dry_run: bool = False) -> None:
        super().__init__(name="HandWorker", daemon=True)
        self.frames, self.state, self.box = frames, state, result_box
        self.cfg, self.stop_event, self.dry_run = cfg, stop_event, dry_run
        self.filter = OneEuroFilter2D(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
        self.flick = FlickDetector(cfg)
        self.mouse: Optional[MouseController] = None
        self._last_seq = 0
        self._fps = 0.0
        self._last_wall = 0.0
        self._freeze_until = float("-inf")
        self._last_flick_time = float("-inf")
        self._last_cursor: Optional[Tuple[float, float]] = None

    def run(self) -> None:
        cfg = self.cfg
        if not self.dry_run:
            self.mouse = MouseController()
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               model_complexity=cfg.model_complexity,
                               min_detection_confidence=cfg.det_conf,
                               min_tracking_confidence=cfg.track_conf)
        try:
            while not self.stop_event.is_set():
                item = self.frames.wait_for_new(self._last_seq, 0.25)
                if item is None:
                    continue
                frame, self._last_seq, ts = item
                try:
                    self._process(hands, frame, ts)
                except Exception:  # noqa: BLE001
                    report_error("HandWorker")
        finally:
            hands.close()

    # ---- per frame ----
    def _process(self, hands, frame: np.ndarray, t: float) -> None:
        cfg = self.cfg
        wall = time.perf_counter()
        if self._last_wall:
            inst = 1.0 / max(wall - self._last_wall, 1e-6)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_wall = wall

        h, w = frame.shape[:2]
        snap = self.state.snapshot()

        small = frame
        if w > cfg.mp_max_width:
            s = cfg.mp_max_width / float(w)
            small = cv2.resize(frame, (cfg.mp_max_width, int(h * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)

        out = HandResult(ts=t, fps=self._fps, flick_time=self._last_flick_time)
        if not res.multi_hand_landmarks:  # hand left the frame
            self._reset_tracking()
            out.status = "no hand"
            self.box.set(out)
            return

        lm = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark], dtype=np.float32)
        px = lm[:, :2] * np.array([w, h], dtype=np.float32)  # pixel units -> isotropic distances
        out.present, out.landmarks = True, lm

        if not snap.matched:
            self._reset_tracking()
            out.status = "no screen match"
        elif snap.paused:
            self._reset_tracking()
            out.status = "paused (F9 / P)"
        elif snap.calibrating:
            self._reset_tracking()
            out.status = "calibrating"
        else:
            self._control(lm, px, t, snap, out)

        out.flick_time = self._last_flick_time
        self.box.set(out)

    def _reset_tracking(self) -> None:
        self.filter.reset()
        self.flick.reset()
        self._last_cursor = None

    def _control(self, lm: np.ndarray, px: np.ndarray, t: float,
                 snap: HomographySnapshot, out: HandResult) -> None:
        cfg = self.cfg
        region = snap.region
        tip = lm[INDEX_TIP]
        # tip must be (almost) inside the camera frame to be trustworthy
        if region is None or not (-0.02 <= tip[0] <= 1.02 and -0.02 <= tip[1] <= 1.02):
            self._reset_tracking()
            out.status = "fingertip outside camera frame"
            return

        pt = project_point(snap.H, float(px[INDEX_TIP][0]), float(px[INDEX_TIP][1]))
        if pt is None:
            self._reset_tracking()
            out.status = "degenerate projection"
            return

        sx, sy = pt
        mx, my = cfg.offscreen_margin * region.width, cfg.offscreen_margin * region.height
        x_lo, x_hi = region.left, region.left + region.width - 1
        y_lo, y_hi = region.top, region.top + region.height - 1
        if not (x_lo - mx <= sx <= x_hi + mx and y_lo - my <= sy <= y_hi + my):
            self._reset_tracking()
            out.status = "finger off-screen"
            return

        sx = min(max(sx, x_lo), x_hi)  # clamp the small overshoot to the screen edge
        sy = min(max(sy, y_lo), y_hi)
        fx, fy = self.filter(sx, sy, t)
        fx = min(max(fx, x_lo), x_hi)
        fy = min(max(fy, y_lo), y_hi)

        ev = self.flick.update(t, px, anchor=(fx, fy))
        if ev is not None:
            # The flick itself moves the fingertip, so click where the cursor WAS at its start.
            ax, ay = ev.anchor if ev.anchor is not None else (fx, fy)
            self._move(ax, ay)
            self._click()
            self._freeze_until = t + cfg.flick_freeze_s
            self._last_flick_time = t
            self._last_cursor = (ax, ay)
            out.status = f"FLICK d={ev.delta:+.2f} v={ev.velocity:+.1f}"
        elif t >= self._freeze_until:
            self._move(fx, fy)
            self._last_cursor = (fx, fy)
            out.status = "tracking"
        else:
            out.status = "click hold"
        out.cursor = self._last_cursor

    # ---- OS control ----
    def _move(self, x: float, y: float) -> None:
        if self.mouse is None:
            return
        try:
            self.mouse.position = (int(round(x)), int(round(y)))
        except Exception:  # noqa: BLE001
            report_error("mouse.move")

    def _click(self) -> None:
        if self.mouse is None:
            return
        try:
            self.mouse.press(Button.left)
            self.mouse.release(Button.left)
        except Exception:  # noqa: BLE001
            report_error("mouse.click")
