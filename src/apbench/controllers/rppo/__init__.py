"""RecurrentPPO (``MlpLstmPolicy``) basal controller trained on the corrected Simglucose."""

from __future__ import annotations

from pathlib import Path

from apbench.controllers.policy import PolicyController, load_model

MODEL_PATH = Path(__file__).resolve().parent / "model.zip"


class RecurrentPPOController(PolicyController):
    def __init__(self, model_path: Path | str = MODEL_PATH, model=None, spec=None):
        if model is None:
            from sb3_contrib import RecurrentPPO

            model, spec = load_model(RecurrentPPO, model_path)
        if spec is None:
            raise ValueError("an ObsActSpec is required when passing an in-memory model")
        super().__init__(model, spec)
