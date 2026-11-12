"""Controller interface and registry.

A controller maps a :class:`~apbench.simulator.ControlContext` to a requested
basal insulin rate in U/min (Simglucose's controller convention). The pump
quantizes and clips the request. Controllers must be deterministic given
``reset()``; recurrent controllers reset their hidden state there.
"""

from __future__ import annotations

from typing import Callable, Protocol

from apbench.simulator import ControlContext


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, ctx: ControlContext) -> float:
        """Return the requested basal rate in U/min."""
        ...


def patient_basal_u_min(patient_name: str) -> float:
    """Steady-state basal rate ``u2ss * BW / 6000`` of a Simglucose patient, in U/min."""
    from apbench import _compat  # noqa: F401
    from simglucose.patient.t1dpatient import T1DPatient

    params = T1DPatient.withName(patient_name)._params
    return float(params.u2ss * params.BW / 6000.0)


class NominalBasalController:
    """Constant basal at the patient's steady-state rate (a reference, not a neural controller).

    The rate is a patient parameter, playing the role of a clinician-programmed basal.
    """

    def __init__(self, patient_name: str):
        self.rate_u_min = patient_basal_u_min(patient_name)

    def reset(self) -> None:
        pass

    def act(self, ctx: ControlContext) -> float:
        return self.rate_u_min


def _rppo(patient_name: str):
    from apbench.controllers.rppo import RecurrentPPOController

    return RecurrentPPOController()


def _ppo(patient_name: str):
    from apbench.controllers.ppo import PPOController

    return PPOController()


CONTROLLERS: dict[str, Callable[[str], Controller]] = {
    "rppo": _rppo,
    "ppo": _ppo,
    "nominal-basal": NominalBasalController,
}


def make_controller(spec: str | Callable[[str], Controller], patient_name: str) -> Controller:
    if callable(spec):
        return spec(patient_name)
    try:
        factory = CONTROLLERS[spec]
    except KeyError:
        raise ValueError(f"unknown controller {spec!r}; choose from {sorted(CONTROLLERS)}") from None
    return factory(patient_name)
