"""One Euro Filter (Casiez et al., CHI 2012), implemented from scratch.

Low speed  -> low cutoff  -> heavy smoothing (no jitter on small buttons).
High speed -> high cutoff -> light smoothing (no lag on fast sweeps).

    cutoff = min_cutoff + beta * |filtered_speed|

Tuning (coordinates are in screen pixels):
  1. beta = 0, raise min_cutoff until jitter at rest is acceptable.
  2. Raise beta until fast movements stop lagging.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple


def _alpha(cutoff: float, dt: float) -> float:
    cutoff = max(cutoff, 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _LowPass:
    def __init__(self) -> None:
        self.y: Optional[float] = None

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t_prev: Optional[float] = None
        self._x_prev: Optional[float] = None
        self._x_lp = _LowPass()
        self._dx_lp = _LowPass()

    def __call__(self, x: float, t: float) -> float:
        """Filter sample `x` taken at time `t` (seconds, monotonic)."""
        if self._t_prev is None:
            self._t_prev, self._x_prev = t, x
            self._x_lp.y, self._dx_lp.y = x, 0.0
            return x

        dt = t - self._t_prev
        if dt <= 1e-6:  # duplicate / non-increasing timestamp: avoid division by zero
            return self._x_lp.y if self._x_lp.y is not None else x

        self._t_prev = t
        dx = (x - self._x_prev) / dt
        self._x_prev = x
        edx = self._dx_lp(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x_lp(x, _alpha(cutoff, dt))


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def __call__(self, x: float, y: float, t: float) -> Tuple[float, float]:
        return self._fx(x, t), self._fy(y, t)
