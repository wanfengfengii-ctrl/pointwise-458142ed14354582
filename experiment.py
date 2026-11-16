"""Public replay and exact-probability random experiment entry points."""
from dataclasses import dataclass
from fractions import Fraction
from allocator import Allocator, observation
from monitor import Monitor

@dataclass(frozen=True)
class Step:
    index: int
    before: tuple
    probability: Fraction
    arm: str
    outcome: int
    after: tuple
    evidence: Fraction
    stopped: bool

@dataclass(frozen=True)
class Result:
    trace: tuple
    counts: tuple
    evidence: Fraction
    stopped: bool

    def as_dict(self):
        return {
            "trace": [
                {"index": x.index, "before": list(x.before),
                 "probability": str(x.probability), "arm": x.arm,
                 "outcome": x.outcome, "after": list(x.after),
                 "evidence": str(x.evidence), "stopped": x.stopped}
                for x in self.trace
            ],
            "counts": list(self.counts), "evidence": str(self.evidence),
            "stopped": self.stopped,
        }

def _horizon(n):
    if type(n) is not int or not 0 <= n <= 16:
        raise ValueError("horizon must be an integer in [0,16]")

def _run(choose, horizon, allocator_factory, monitor_factory):
    _horizon(horizon)
    allocator, monitor = allocator_factory(), monitor_factory()
    trace = []
    for index in range(1, horizon + 1):
        before = allocator.counts
        probability = allocator.probability()
        arm, outcome = choose(probability)
        observation(arm, outcome)
        monitor.observe(probability, arm, outcome)
        allocator.observe(arm, outcome)
        trace.append(Step(index, before, probability, arm, outcome,
                          allocator.counts, monitor.evidence, monitor.stopped))
        if monitor.stopped:
            break
    return Result(tuple(trace), allocator.counts, monitor.evidence, monitor.stopped)

def replay(tape, horizon=16, *, allocator_factory=Allocator, monitor_factory=Monitor):
    """Consume at most horizon pairs; no further input is pulled after stopping."""
    _horizon(horizon)
    iterator = iter(tape)
    def choose(probability):
        try:
            return next(iterator)
        except StopIteration:
            raise ValueError("tape exhausted before horizon or stopping") from None
    return _run(choose, horizon, allocator_factory, monitor_factory)

def simulate(p, rng, horizon=16, *, allocator_factory=Allocator, monitor_factory=Monitor):
    """rng supplies randrange(n); draw allocation first, then the Bernoulli outcome."""
    _horizon(horizon)
    if type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    def choose(q):
        arm = "A" if rng.randrange(q.denominator) < q.numerator else "B"
        outcome = int(rng.randrange(p.denominator) < p.numerator)
        return arm, outcome
    return _run(choose, horizon, allocator_factory, monitor_factory)
