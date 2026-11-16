"""Adaptive allocation; counts are (A successes, A failures, B successes, B failures)."""
from fractions import Fraction

def observation(arm, outcome):
    if arm not in ("A", "B") or type(outcome) is not int or outcome not in (0, 1):
        raise ValueError("expected A/B and integer 0/1")

class Allocator:
    def __init__(self):
        self._counts = (0, 0, 0, 0)

    @property
    def counts(self):
        return self._counts

    def probability(self):
        a, af, b, bf = self._counts
        na, nb = a + af, b + bf
        if not na or not nb:
            return Fraction(1, 2)
        diff = a * nb - b * na
        return Fraction(3 if diff > 0 else 1 if diff < 0 else 2, 4)

    def observe(self, arm, outcome):
        observation(arm, outcome)
        counts = list(self._counts)
        counts[(0 if arm == "A" else 2) + (0 if outcome else 1)] += 1
        self._counts = tuple(counts)
