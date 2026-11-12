"""Multi-property falsification with Psy-TaLiRo that reuses SUT traces across specifications.

The specifications are targeted one at a time, in the given order, by Psy-TaLiRo's native
DualAnnealing optimizer, whose objective is the robustness of the targeted specification only.
Every trace simulated during a search is also monitored against all the specifications that are
not falsified yet, and such incidental falsifications are recorded immediately. A specification
falsified this way is never targeted. No optimizer state is shared between searches.

A trace falsifies a specification iff its robustness is strictly negative, which coincides with
the stopping condition of DualAnnealing(min_cost=0.0). Zero robustness is inconclusive.

An optional observer is notified of every monitored simulation (for logging and for saving
counterexamples without re-simulating).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from staliro import CostFunc, Inputs, Result, Specification, TestOptions, staliro
from staliro.models import Model
from staliro.optimizers import DualAnnealing

ALL_FALSIFIED = "ALL_FALSIFIED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
TIMEOUT = "TIMEOUT"


@dataclass
class SpecRecord:
    name: str
    formula: str
    specification: Specification[Any, float, Any]
    best_cost: float = float("inf")
    falsified_at_sim: int | None = None
    falsified_at_time: float | None = None
    falsifying_cost: float | None = None
    via: str | None = None  # "target" or "reuse"
    target_sims: int = 0
    target_elapsed: float = 0.0
    target_status: str = "not_reached"  # "falsified", "budget", "timeout", "skipped", "not_reached"

    @property
    def falsified(self) -> bool:
        return self.falsified_at_sim is not None


@dataclass
class Summary:
    records: list[SpecRecord]
    simulations: int
    elapsed: float
    status: str

    @property
    def num_falsified(self) -> int:
        return sum(1 for record in self.records if record.falsified)


class _Timeout(Exception):
    pass


#: ``observer(simulation, target, trace, costs, newly_falsified)``: called once per monitored
#: simulation with its 1-based number, the name of the targeted specification, the simulated trace
#: (the model's ``Result.value``), the robustness of every monitored specification, and the records
#: falsified for the first time by this trace.
Observer = Callable[[int, str, Any, dict[str, float], list[SpecRecord]], None]


@dataclass
class _Campaign:
    model: Model[Any, Any]
    records: list[SpecRecord]
    clock: Callable[[], float]
    start: float
    deadline: float | None
    log: Callable[[str], None]
    observer: Observer | None = None
    simulations: int = 0

    def elapsed(self) -> float:
        return self.clock() - self.start

    def timed_out(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline


@dataclass
class _ReusingCostFunc(CostFunc[float, dict[str, float]]):
    """Cost function for one targeted search: simulate once, monitor all pending specifications."""

    campaign: _Campaign
    target: SpecRecord
    simulations: int = field(default=0)

    def evaluate(self, inputs: Inputs[Any]) -> Result[float, dict[str, float]]:
        campaign = self.campaign
        if campaign.timed_out():
            raise _Timeout()

        trace = campaign.model.simulate(inputs).value
        campaign.simulations += 1
        self.simulations += 1

        # A simulation finishing after the deadline is counted but cannot falsify anything.
        if campaign.timed_out():
            raise _Timeout()

        costs: dict[str, float] = {}
        newly_falsified: list[SpecRecord] = []
        for record in campaign.records:
            if record.falsified and record is not self.target:
                continue
            cost = float(record.specification.evaluate(trace).value)
            costs[record.name] = cost
            record.best_cost = min(record.best_cost, cost)
            if cost < 0.0 and not record.falsified:
                record.falsified_at_sim = campaign.simulations
                record.falsified_at_time = campaign.elapsed()
                record.falsifying_cost = cost
                record.via = "target" if record is self.target else "reuse"
                newly_falsified.append(record)
                campaign.log(
                    f"{record.name} falsified at simulation {record.falsified_at_sim} "
                    f"({record.falsified_at_time:.2f} s, via {record.via})"
                )

        if campaign.observer is not None:
            campaign.observer(campaign.simulations, self.target.name, trace, costs, newly_falsified)

        return Result(costs[self.target.name], costs)


def run_multi_property(
    model: Model[Any, Any],
    specifications: Sequence[tuple[str, str, Specification[Any, float, Any]]],
    make_options: Callable[[], TestOptions],
    *,
    timeout: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    log: Callable[[str], None] = print,
    observer: Observer | None = None,
) -> Summary:
    """Falsify the given (name, formula, specification) triples, reusing every simulated trace.

    Each targeted search is a native Psy-TaLiRo run of DualAnnealing(min_cost=0.0) whose budget is
    given by ``make_options().iterations``. The optional ``timeout`` (in seconds) bounds the whole
    campaign; it is checked before and after each simulation. The optional ``observer`` is called
    after each simulation whose trace is monitored (see :data:`Observer`).
    """

    records = [SpecRecord(name, formula, spec) for name, formula, spec in specifications]
    start = clock()
    campaign = _Campaign(
        model=model,
        records=records,
        clock=clock,
        start=start,
        deadline=None if timeout is None else start + timeout,
        log=log,
        observer=observer,
    )
    status = BUDGET_EXHAUSTED

    for index, record in enumerate(records):
        if record.falsified:
            record.target_status = "skipped"
            continue
        if campaign.timed_out():
            status = TIMEOUT
            for pending in records[index:]:
                if not pending.falsified:
                    pending.target_status = "timeout"
            break

        log(f"Start falsifying {record.name}")
        cost_func = _ReusingCostFunc(campaign, record)
        search_start = clock()
        try:
            staliro(cost_func, DualAnnealing(min_cost=0.0), make_options())
        except _Timeout:
            status = TIMEOUT
            for pending in records[index:]:
                if not pending.falsified:
                    pending.target_status = "timeout"
            break
        finally:
            record.target_sims = cost_func.simulations
            record.target_elapsed = clock() - search_start

        record.target_status = "falsified" if record.via == "target" else "budget"

    if all(record.falsified for record in records):
        status = ALL_FALSIFIED

    return Summary(records, campaign.simulations, campaign.elapsed(), status)


__all__ = [
    "ALL_FALSIFIED",
    "Observer",
    "BUDGET_EXHAUSTED",
    "TIMEOUT",
    "SpecRecord",
    "Summary",
    "run_multi_property",
]
