"""Observation and action conventions of the RecurrentPPO controller.

The same :class:`ObservationBuilder` is used by the training environment
(:mod:`apbench.rl.env`) and by the deployed controller
(:mod:`apbench.controllers.rppo`), both fed with the same
:class:`~apbench.simulator.ControlContext`. Only quantities that a practical
controller can observe are used: the CGM observation, its recent change, and the
insulin rate the pump delivered in the previous control period.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from apbench.simulator import ControlContext


@dataclass(frozen=True)
class ObsActSpec:
    """Scaling constants; stored in the checkpoint metadata and loaded with it."""

    cgm_center: float = 140.0  # mg/dL
    cgm_scale: float = 100.0  # mg/dL
    trend_steps: int = 5  # CGM change over 5 control steps (15 min with Dexcom)
    trend_scale: float = 30.0  # mg/dL per trend window
    max_rate_u_hour: float = 5.0  # action a in [-1, 1] maps linearly to [0, max_rate_u_hour] U/hour

    @property
    def obs_dim(self) -> int:
        return 3

    def to_dict(self) -> dict:
        return asdict(self)

    def action_to_rate_u_min(self, action: float) -> float:
        """Map a policy action in [-1, 1] to a basal request in U/min."""
        a = float(np.clip(action, -1.0, 1.0))
        return (a + 1.0) / 2.0 * self.max_rate_u_hour / 60.0


class ObservationBuilder:
    """Stateful construction of the observation vector ``[cgm, trend, prev_rate]``.

    * ``cgm = (CGM - cgm_center) / cgm_scale``
    * ``trend = (CGM - CGM[trend_steps control steps ago]) / trend_scale``; the history is
      padded with the first observation after ``reset()``.
    * ``prev_rate = delivered rate over the previous control step / max_rate_u_hour``.
    """

    def __init__(self, spec: ObsActSpec):
        self.spec = spec
        self._history: deque[float] = deque(maxlen=spec.trend_steps + 1)

    def reset(self) -> None:
        self._history.clear()

    def build(self, ctx: ControlContext) -> np.ndarray:
        s = self.spec
        if not self._history:
            self._history.extend([ctx.cgm] * (s.trend_steps + 1))
        else:
            self._history.append(ctx.cgm)
        return np.array(
            [
                (ctx.cgm - s.cgm_center) / s.cgm_scale,
                (ctx.cgm - self._history[0]) / s.trend_scale,
                ctx.prev_delivered_u_min * 60.0 / s.max_rate_u_hour,
            ],
            dtype=np.float32,
        )
