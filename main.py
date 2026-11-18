"""Touchless virtual mouse: entry point, camera loop, debug UI, manual calibration.

Keys (debug window focused):  C calibrate | R reset manual calib | P pause | Q/Esc quit
Global hotkey:                F9 pause/resume cursor control (safety switch)
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hand_tracker import HandWorker, draw_hand
from screen_matcher import (ScreenMatcherWorker, is_valid_quad,
                            primary_monitor_region, safe_inverse)
from shared import (DEBUG_WINDOW_TITLE, Config, FrameBuffer, HandResultBox,
                    HomographySnapshot, SharedState)

GREEN, RED, YELLOW, MAGENTA, WHITE = (0, 220, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 255)
CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def _enable_dpi_awareness() -> None:
    """Make mss and pynput agree on physical pixels on Windows."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def open_camera(cfg: Config) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.cam_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cfg.cam_index}")
    return cap


def start_hotkey(state: SharedState):
    try:
        from pynput import keyboard
        listener = keyboard.GlobalHotKeys({"<f9>": state.toggle_paused})
        listener.daemon = True
        listener.start()
        return listener
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] global F9 hotkey unavailable ({exc}); use 'P' in the debug window.")
        return None


class Calibrator:
    """Manual 4-corner calibration (TL, TR, BR, BL of the screen as seen by the camera)."""

    def __init__(self) -> None:
        self.active = False
        self.points: List[Tuple[float, float]] = []
        self.frame: Optional[np.ndarray] = None

    def start(self, frame: np.ndarray) -> None:
        self.active, self.points, self.frame = True, [], frame.copy()

    def cancel(self) -> None:
        self.active, self.points, self.frame = False, [], None

    def on_mouse(self, event: int, x: int, y: int, scale: float) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.active and len(self.points) < 4:
            self.points.append((x / scale, y / scale))  # back to camera-frame pixels


def finish_calibration(cal: Calibrator, state: SharedState, cfg: Config, frame_shape) -> str:
    h, w = frame_shape[:2]
    region = state.get_region() or primary_monitor_region()
    cam = np.array(cal.points, dtype=np.float32)
    if not is_valid_quad(cam, w, h, cfg):
        return "Calibration rejected: corners must be convex, in order TL,TR,BR,BL"
    M = cv2.getPerspectiveTransform(cam, region.corners())
    M = M.astype(np.float64)
    if not np.all(np.isfinite(M)) or safe_inverse(M) is None or abs(M[2, 2]) < 1e-12:
        return "Calibration rejected: degenerate homography"
    state.set_manual(M / M[2, 2], cam, region)
    return "Manual calibration stored (used as fallback when auto-match is lost)"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def put(vis, text, y, color=WHITE, scale=0.6, thick=2):
    cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(vis: np.ndarray, snap: HomographySnapshot, res, cam_fps: float,
                 cfg: Config, flash: str) -> None:
    # screen boundary polygon: green when matched, red while searching
    quad = snap.quad if snap.matched else snap.stale_quad
    if quad is not None and np.all(np.isfinite(quad)):
        poly = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, GREEN if snap.matched else RED, 3, cv2.LINE_AA)

    now = time.perf_counter()
    flicking = (now - res.flick_time) < 0.4
    if res.present and res.landmarks is not None:
        draw_hand(vis, res.landmarks, flicking)
    if flicking:
        cv2.putText(vis, "FLICK!", (vis.shape[1] // 2 - 110, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, MAGENTA, 5, cv2.LINE_AA)

    st = snap.stats
    if snap.matched:
        extra = f"auto, {int(st.get('inliers', 0))} inliers" if snap.source == "auto" else "MANUAL calibration"
        put(vis, f"SCREEN: MATCHED ({extra})", 28, GREEN)
    else:
        put(vis, f"SCREEN: SEARCHING... good={int(st.get('good', 0))} "
                 f"inl={int(st.get('inliers', 0))}  (press C to calibrate)", 28, RED)
    put(vis, f"Cam FPS {cam_fps:4.1f} | Hand FPS {res.fps:4.1f} | match {st.get('ms', 0):4.0f} ms "
             f"| target={cfg.target}", 54)
    ctrl = "PAUSED" if snap.paused else ("CALIBRATING" if snap.calibrating else "ON")
    put(vis, f"CONTROL: {ctrl} | {res.status}", 80, YELLOW if ctrl != "ON" else WHITE)
    if flash:
        put(vis, flash, vis.shape[0] - 40, YELLOW)
    put(vis, "C calibrate | R reset | P pause (F9 global) | Q quit", vis.shape[0] - 14, WHITE, 0.5, 1)


def draw_calibration(vis: np.ndarray, cal: Calibrator) -> None:
    for i, (x, y) in enumerate(cal.points):
        cv2.circle(vis, (int(x), int(y)), 7, MAGENTA, -1, cv2.LINE_AA)
        cv2.putText(vis, str(i + 1), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, MAGENTA, 2, cv2.LINE_AA)
    if len(cal.points) > 1:
        pts = np.round(np.array(cal.points)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], len(cal.points) == 4, MAGENTA, 2, cv2.LINE_AA)
    nxt = CORNER_NAMES[min(len(cal.points), 3)]
    put(vis, f"CALIBRATION: click the {nxt} corner of your screen ({len(cal.points)}/4)", 28, YELLOW)
    put(vis, "Esc cancels. Cursor control is disabled meanwhile.", 54)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Touchless virtual mouse")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--target", choices=["monitor", "window"], default="monitor",
                   help="match against the primary monitor or the active window (window: Windows only)")
    p.add_argument("--matcher", choices=["bf", "flann"], default="bf")
    p.add_argument("--display-scale", type=float, default=0.75)
    p.add_argument("--min-cutoff", type=float, default=1.0, help="1€ filter: lower = smoother at rest")
    p.add_argument("--beta", type=float, default=0.02, help="1€ filter: higher = less lag when fast")
    p.add_argument("--flick-direction", choices=["flex", "extend", "either"], default="either")
    p.add_argument("--dry-run", action="store_true", help="never touch the real mouse")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _enable_dpi_awareness()
    cfg = Config(cam_index=args.camera, cam_width=args.width, cam_height=args.height,
                 target=args.target, matcher=args.matcher, display_scale=args.display_scale,
                 min_cutoff=args.min_cutoff, beta=args.beta, flick_direction=args.flick_direction)

    stop = threading.Event()
    frames, state, box = FrameBuffer(), SharedState(cfg), HandResultBox()
    cap = open_camera(cfg)
    matcher = ScreenMatcherWorker(frames, state, cfg, stop)
    hand = HandWorker(frames, state, box, cfg, stop, dry_run=args.dry_run)
    hotkey = start_hotkey(state)
    cal = Calibrator()

    cv2.namedWindow(DEBUG_WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(DEBUG_WINDOW_TITLE,
                         lambda ev, x, y, fl, pr: cal.on_mouse(ev, x, y, cfg.display_scale))
    matcher.start()
    hand.start()

    cam_fps, last, fails = 0.0, time.perf_counter(), 0
    flash, flash_until = "", 0.0
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    print("Camera stopped delivering frames; exiting.")
                    break
                time.sleep(0.01)
                continue
            fails = 0
            now = time.perf_counter()
            frames.put(frame, now)  # NOTE: do not mirror/flip: that would break ORB matching
            inst = 1.0 / max(now - last, 1e-6)
            cam_fps = inst if cam_fps == 0 else 0.9 * cam_fps + 0.1 * inst
            last = now

            snap, res = state.snapshot(), box.get()
            vis = (cal.frame if cal.active else frame).copy()
            if cal.active:
                draw_calibration(vis, cal)
            else:
                draw_overlay(vis, snap, res, cam_fps, cfg, flash if now < flash_until else "")
            if cfg.display_scale != 1.0:
                vis = cv2.resize(vis, None, fx=cfg.display_scale, fy=cfg.display_scale,
                                 interpolation=cv2.INTER_AREA)
            cv2.imshow(DEBUG_WINDOW_TITLE, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 and cal.active:
                cal.cancel()
                state.set_calibrating(False)
            elif key in (ord("q"), 27):
                break
            elif key == ord("c") and not cal.active:
                cal.start(frame)
                state.set_calibrating(True)
            elif key == ord("r"):
                state.clear_manual()
                flash, flash_until = "Manual calibration cleared", now + 3
            elif key == ord("p"):
                state.toggle_paused()

            if cal.active and len(cal.points) == 4:
                flash = finish_calibration(cal, state, cfg, frame.shape)
                flash_until = time.perf_counter() + 4
                cal.cancel()
                state.set_calibrating(False)
    finally:
        stop.set()
        matcher.join(timeout=2)
        hand.join(timeout=2)
        if hotkey is not None:
            hotkey.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
