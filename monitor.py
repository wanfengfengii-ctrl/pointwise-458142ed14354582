"""One-sided evidence for higher A conversion; exact arithmetic."""
from fractions import Fraction
from allocator import observation

THRESHOLD = Fraction(8)
PROBABILITIES = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))

class Monitor:
    def __init__(self):
        self.evidence = Fraction(1)
        self.stopped = False

    def observe(self, probability, arm, outcome):
        observation(arm, outcome)
        if type(probability) is not Fraction or probability not in PROBABILITIES:
            raise ValueError("invalid allocation probability")
        if self.stopped:
            raise RuntimeError("experiment already stopped")
        if outcome:
            denominator = probability if arm == "A" else 1 - probability
            signed = Fraction(1, 4) / denominator
            self.evidence *= 1 + signed if arm == "A" else 1 - signed
        self.stopped = self.evidence >= THRESHOLD
        return self.evidence
