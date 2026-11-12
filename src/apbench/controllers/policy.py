"""Deterministic deployment of a Stable-Baselines3 policy as a benchmark controller.

Shared by the RecurrentPPO and PPO controllers. Each checkpoint directory holds
``model.zip`` (SB3's native format) and ``metadata.json``, which records the
observation/action scaling (:class:`~apbench.observation.ObsActSpec`), training
hyperparameters, seeds and software provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apbench.observation import ObservationBuilder, ObsActSpec
from apbench.simulator import ControlContext


def load_spec(metadata_path: Path) -> ObsActSpec:
    with open(metadata_path) as f:
        return ObsActSpec(**json.load(f)["obs_act_spec"])


def load_model(algorithm_cls, model_path: Path | str):
    import torch

    torch.set_num_threads(1)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {model_path}")
    return algorithm_cls.load(model_path, device="cpu"), load_spec(model_path.with_name("metadata.json"))


class PolicyController:
    """Deterministic inference with the shared observation builder.

    For recurrent policies the hidden state persists for the whole query (all control
    steps between two ``reset()`` calls) and is cleared by ``reset()``. Feed-forward
    SB3 policies accept and ignore the state arguments, so it stays ``None``.
    """

    def __init__(self, model, spec: ObsActSpec):
        self.model = model
        self.spec = spec
        self.builder = ObservationBuilder(spec)
        self.reset()

    def reset(self) -> None:
        self.lstm_state = None
        self._episode_start = True
        self.builder.reset()

    def act(self, ctx: ControlContext) -> float:
        obs = self.builder.build(ctx)
        action, self.lstm_state = self.model.predict(
            obs[None, :],
            state=self.lstm_state,
            episode_start=np.array([self._episode_start]),
            deterministic=True,
        )
        self._episode_start = False
        return self.spec.action_to_rate_u_min(float(np.asarray(action).reshape(-1)[0]))
