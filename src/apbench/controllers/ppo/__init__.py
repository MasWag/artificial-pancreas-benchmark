"""Feed-forward PPO (``MlpPolicy``) basal controller trained on the corrected Simglucose.

Trained with the same environment, observation, action scaling, reward, seeds and
model selection as the RecurrentPPO controller; only the policy is memoryless.
"""

from __future__ import annotations

from pathlib import Path

from apbench.controllers.policy import PolicyController, load_model

MODEL_PATH = Path(__file__).resolve().parent / "model.zip"


class PPOController(PolicyController):
    def __init__(self, model_path: Path | str = MODEL_PATH, model=None, spec=None):
        if model is None:
            from stable_baselines3 import PPO

            model, spec = load_model(PPO, model_path)
        if spec is None:
            raise ValueError("an ObsActSpec is required when passing an in-memory model")
        super().__init__(model, spec)
