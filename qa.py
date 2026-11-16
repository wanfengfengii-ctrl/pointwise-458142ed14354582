#!/usr/bin/env python3
"""Quality package for the adaptive-allocation continuous-monitoring experiment.

Commands
--------
    python3 qa.py build   regenerate evidence.json from first principles
    python3 qa.py verify  re-run every check, compare saved evidence and report
    python3 qa.py demo    show a real allocation switch, stop, and one mutant

How evidence.json is produced (public statement): ``build`` computes every
deterministic value by calling oracle.py, which re-derives the allocation,
evidence and stopping rules independently of allocator.py / monitor.py /
experiment.py.  It records exact rational strings, not floating point, and
writes them with json.dump(sort_keys=True).  ``verify`` never treats the saved
file as the source of truth: it recomputes the whole dictionary and requires
exact equality with the saved one, then checks that REPORT.md contains the
deterministic conclusions.  Running an old JSON or printing success is not a
verification.
"""
import bisect
import json
import hashlib
import sys
import re
import time
import tracemalloc
import unittest
from fractions import Fraction
from pathlib import Path
from random import Random

import oracle
from allocator import Allocator
from monitor import Monitor
import experiment

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence.json"
REPORT = ROOT / "REPORT.md"
BLOCKERS = ROOT / "deviations.json"
PRODUCTION_FILES = ("allocator.py", "monitor.py", "experiment.py",
                    "checks.py", "samples.json", "demo.py")

THRESHOLD = Fraction(8)
MAX_HORIZON = 16
TOLERANCE = Fraction(1, 100000)
NOMINAL = Fraction(1, 8)
SAMPLE = [("A", 1), ("B", 0), ("A", 1), ("A", 1), ("A", 1), ("A", 1),
          ("A", 1), ("A", 1), ("B", 1)]
FINAL_ONLY_TAPE = SAMPLE[:8] + [("B", 1)] * 8
EQUALITY_TAPE = [("A", 0), ("B", 1), ("A", 1), ("A", 1), ("A", 1), ("A", 1)]
FALLBACK_TAPE = EQUALITY_TAPE + [("A", 0), ("B", 1)]
POINT_PROBES = (Fraction(0), Fraction(1, 1000), Fraction(1, 4),
                Fraction(1, 3), Fraction(1, 2), Fraction(2, 3),
                Fraction(3, 4), Fraction(999, 1000), Fraction(1))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ftext(value):
    return str(Fraction(value))


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in PRODUCTION_FILES}


def summarise_run(run):
    return {
        "steps": len(run["trace"]),
        "consumed": run["consumed"],
        "counts": list(run["counts"]),
        "evidence": ftext(run["evidence"]),
        "stopped": bool(run["stopped"]),
        "first_crossing": run["first_crossing"],
        "probabilities": [ftext(s["probability"]) for s in run["trace"]],
        "counts_before": [list(s["before"]) for s in run["trace"]],
        "trace_digest": digest([{
            "index": s["index"], "before": list(s["before"]),
            "probability": ftext(s["probability"]), "arm": s["arm"],
            "outcome": s["outcome"], "after": list(s["after"]),
            "evidence": ftext(s["evidence"]), "stopped": bool(s["stopped"]),
        } for s in run["trace"]]),
    }


# --------------------------------------------------------------------------- #
# mutation adapters (quality package only; production files untouched)
# --------------------------------------------------------------------------- #
class StaleProbabilityMonitor:
    """Mutation 1: keep feeding the *initial* allocation probability 1/2."""

    def __init__(self):
        self.evidence = Fraction(1)
        self.stopped = False
        self.received = []

    def observe(self, probability, arm, outcome):
        if probability not in oracle.ALLOWED_Q:
            raise ValueError("invalid allocation probability")
        if self.stopped:
            raise RuntimeError("experiment already stopped")
        self.received.append(probability)
        self.evidence *= oracle.evidence_factor(Fraction(1, 2), arm, outcome)
        self.stopped = self.evidence >= THRESHOLD
        return self.evidence


class FinalEvidenceOnlyMonitor:
    """Mutation 2: never stop early; judge only the evidence at the horizon."""

    def __init__(self):
        self.evidence = Fraction(1)
        self.stopped = False
        self.received = []

    def observe(self, probability, arm, outcome):
        if probability not in oracle.ALLOWED_Q:
            raise ValueError("invalid allocation probability")
        self.received.append(probability)
        self.evidence *= oracle.evidence_factor(probability, arm, outcome)
        self.stopped = False
        return self.evidence

    def decision(self):
        return self.evidence >= THRESHOLD


def stale_factory():
    return StaleProbabilityMonitor()


def final_only_factory():
    return FinalEvidenceOnlyMonitor()


# --------------------------------------------------------------------------- #
# deterministic evidence construction
# --------------------------------------------------------------------------- #
def build_evidence():
    facts = {}

    # --- 1. exact F_n polynomials via merged-state DP ---------------------- #
    increments, dp_stats = oracle.stopping_increments(MAX_HORIZON)
    f16 = oracle.cumulative_from_increments(increments, MAX_HORIZON)
    # Path-level sanity from the polynomial itself: F_16(1) must equal the
    # total mass of *some* first crossing by step 16; no crossing is possible
    # before step 6, and F_16(1/2) must lie inside (0, 1/8).
    _expect(oracle.evaluate_polynomial(f16, Fraction(1)) > 0,
            "F16(1) must be positive (some path crosses by step 16)")
    _expect(oracle.evaluate_polynomial(f16, Fraction(1)) <= 1,
            "F16(1) must be a probability")
    _expect(oracle.evaluate_polynomial(f16, Fraction(0)) == 0,
            "F16(0) must be zero (no A success => no evidence)")
    facts["fn"] = {
        "horizon": MAX_HORIZON,
        "increments": [[ftext(c) for c in poly] for poly in increments],
        "increments_digest": digest([[ftext(c) for c in poly] for poly in increments]),
        "F16": [ftext(c) for c in f16],
        "F16_digest": digest([ftext(c) for c in f16]),
        "dp_stats": {k: int(v) for k, v in dp_stats.items()},
    }

    # --- 2. small horizons: full allocation+outcome enumeration ----------- #
    enumeration = []
    for horizon in range(0, 9):
        enumerated, info = oracle.enumerated_stopping_polynomial(horizon)
        merged = oracle.cumulative_from_increments(increments[:horizon], horizon)
        enumeration.append({
            "horizon": horizon,
            "paths_visited": info["paths_visited"],
            "matches_merged_dp": enumerated == merged,
            "enumeration_digest": digest([ftext(c) for c in enumerated]),
            "merged_digest": digest([ftext(c) for c in merged]),
        })
    facts["enumeration"] = enumeration
    facts["enumeration_guard"] = {
        "max_horizon": 8,
        "rejected_horizon": 16,
        "rejected_message": None,
    }
    try:
        oracle.enumerated_stopping_polynomial(16)
    except ValueError as error:
        facts["enumeration_guard"]["rejected_message"] = str(error)

    # --- 3. scalar DP cross-check at rational p --------------------------- #
    scalar = {}
    for p in POINT_PROBES:
        value = oracle.false_positive_probability(p, MAX_HORIZON)
        polynomial = oracle.evaluate_polynomial(f16, p)
        scalar[ftext(p)] = {"scalar": ftext(value), "polynomial": ftext(polynomial),
                            "match": value == polynomial}
    facts["scalar"] = scalar

    # --- 4. whole-interval envelope and certified supremum ---------------- #
    leaves, cover_meta = oracle.bernstein_cover(f16)
    cover_check = oracle.check_cover(leaves)
    sup_l, sup_u = oracle.sup_bounds(leaves)
    widest = max(leaves, key=lambda leaf: leaf["hi"] - leaf["lo"])
    affine = oracle.affine_sup_bounds(f16, pieces=1024)
    combined_l = max(sup_l, affine["lower"])
    combined_u = min(sup_u, affine["upper"])
    facts["envelope"] = {
        "representation": "piecewise-constant dyadic Bernstein hull",
        "leaf_count": len(leaves),
        "max_depth": cover_check["max_depth"],
        "max_width": ftext(oracle.max_leaf_width(leaves)),
        "max_width_ok": oracle.max_leaf_width(leaves) <= TOLERANCE,
        "cover": {
            "covers_unit_interval": cover_check["covers_unit_interval"],
            "measure": ftext(cover_check["measure"]),
        },
        "tree": oracle.cover_tree(leaves),
        "tree_reconstructs": oracle.check_tree(oracle.cover_tree(leaves), leaves),
        "digest": oracle.cover_digest(leaves),
        "widest_leaf": {
            "i": widest["i"], "m": widest["m"],
            "lo": ftext(widest["lo"]), "hi": ftext(widest["hi"]),
            "width": ftext(widest["hi"] - widest["lo"]),
        },
        "bernstein_roundtrip_ok": all(
            oracle.bernstein_roundtrip_ok(leaf, cover_meta["degree"],
                                          cover_meta["basis_lcm"])
            for leaf in leaves),
        "endpoints_contained": all(
            oracle.leaf_value_checks(leaf, f16)[0] for leaf in leaves),
        "basis_denominator": cover_meta["denominator"],
        "degree": cover_meta["degree"],
    }
    facts["sup"] = {
        "bernstein": {"L": ftext(sup_l), "U": ftext(sup_u),
                      "gap": ftext(sup_u - sup_l)},
        "affine_chord": {"L": ftext(affine["lower"]), "U": ftext(affine["upper"]),
                         "gap": ftext(affine["upper"] - affine["lower"]),
                         "pieces": affine["pieces"], "m2": ftext(affine["m2"]),
                         "error": ftext(affine["error"])},
        "combined": {"L": ftext(combined_l), "U": ftext(combined_u),
                     "gap": ftext(combined_u - combined_l)},
        "gap_within_tolerance": (combined_u - combined_l) <= TOLERANCE,
        "upper_below_nominal": combined_u <= NOMINAL,
    }

    # --- 5. pointwise envelope containment on a rational scan ------------- #
    lefts = [oracle.leaf_interval(leaf)[0] for leaf in leaves]
    scan_bad = 0
    for i in range(1001):
        p = Fraction(i, 1000)
        idx = max(0, min(_bisect(lefts, p), len(leaves) - 1))
        leaf = leaves[idx]
        value = oracle.evaluate_polynomial(f16, p)
        if not leaf["lo"] <= value <= leaf["hi"]:
            scan_bad += 1
    facts["envelope_scan"] = {"points": 1001, "outside": scan_bad,
                              "all_inside": scan_bad == 0}
    lookup = {}
    for p in (Fraction(0), Fraction(1, 1000), Fraction(1, 4), Fraction(1, 3),
              Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
              Fraction(999, 1000), Fraction(1)):
        low, high = oracle.envelope_value(leaves, p)
        lookup[ftext(p)] = {"L": ftext(low), "U": ftext(high),
                            "F16": ftext(oracle.evaluate_polynomial(f16, p))}
    facts["envelope_lookup"] = lookup

    # --- 6. executable business scenarios --------------------------------- #
    sample = oracle.reference_run(SAMPLE)
    equality = oracle.reference_run(EQUALITY_TAPE)
    fallback = oracle.reference_run(FALLBACK_TAPE, horizon=8, force_continue=True)
    facts["sample"] = summarise_run(sample)
    facts["threshold_equality"] = {
        **summarise_run(equality),
        "evidence_sequence": [ftext(s["evidence"]) for s in equality["trace"]],
    }
    facts["fallback_after_crossing"] = {
        "evidence_sequence": [ftext(s["evidence"]) for s in fallback["trace"]],
        "first_crossing": fallback["first_crossing"],
        "final_below_threshold": fallback["evidence"] < THRESHOLD,
    }
    facts["same_counts_different_evidence"] = {
        "counts": [1, 1, 1, 0],
        "via_A_A_B": ftext(oracle.reference_run(
            [("A", 0), ("A", 1), ("B", 1)], 3)["evidence"]),
        "via_A_B_A": ftext(oracle.reference_run(
            [("A", 0), ("B", 1), ("A", 1)], 3)["evidence"]),
    }
    zero_tape = [("A", 1), ("B", 0), ("B", 1)]
    facts["zero_evidence"] = {
        "tape": [list(item) for item in zero_tape],
        "B_success_at_q_three_quarters": ftext(
            oracle.evidence_factor(Fraction(3, 4), "B", 1)),
        "tape_evidence": ftext(oracle.reference_run(zero_tape, 3)["evidence"]),
        "stopped": oracle.reference_run(zero_tape, 3)["stopped"],
    }
    facts["empty_horizon"] = summarise_run(oracle.reference_run([], 0))
    facts["endpoints"] = {
        "F16_0": ftext(oracle.evaluate_polynomial(f16, Fraction(0))),
        "F16_1": ftext(oracle.evaluate_polynomial(f16, Fraction(1))),
    }

    # --- 7. production integration through public factories --------------- #
    integrated = {}
    for seed in (0, 1, 7, 19, 31, 101):
        p = Fraction(seed, 100) if seed not in (0,) else Fraction(0)
        if p > 1:
            p = Fraction(1)
        produced = experiment.simulate(p, Random(seed), 16)
        reference = oracle.simulate_reference(p, Random(seed), 16)
        integrated["simulate_seed_%d" % seed] = {
            "p": ftext(p), "differences": len(oracle.diagnose(produced.trace,
                                                             produced.evidence,
                                                             produced.stopped)),
            "matches_reference": not oracle.compare_run(produced, reference),
            "summary": summarise_run(reference),
        }
    produced_sample = experiment.replay(SAMPLE, 16)
    integrated["replay_sample"] = {
        "matches_reference": not oracle.compare_run(
            produced_sample, oracle.reference_run(SAMPLE, 16)),
        "violations": oracle.diagnose(produced_sample.trace,
                                      produced_sample.evidence,
                                      produced_sample.stopped),
    }
    facts["integration"] = integrated

    # --- 8. quantitative impact of each mutant ---------------------------- #
    mutant_rates = {}
    for label, point in (("0", Fraction(0)), ("1/4", Fraction(1, 4)),
                         ("1/2", Fraction(1, 2)), ("3/4", Fraction(3, 4)),
                         ("1", Fraction(1))):
        mutant_rates[label] = {
            "correct": ftext(oracle.false_positive_probability(point, 16)),
            "stale_probability": ftext(
                oracle.stale_probability_false_positive(point, 16)),
            "final_evidence_only": ftext(
                oracle.final_evidence_only_probability(point, 16)),
        }
    facts["mutant_false_positive_rates"] = mutant_rates

    # --- 9. mutants -------------------------------------------------------- #
    facts["mutations"] = _mutation_evidence()
    return facts


def _bisect(lefts, p):
    return bisect.bisect_right(lefts, p) - 1


def _mutation_evidence():
    recorded = {}

    stale_result = experiment.replay(SAMPLE, 16, monitor_factory=stale_factory)
    stale_report = oracle.reference_run(SAMPLE)
    stale_violations = oracle.diagnose(stale_result.trace, stale_result.evidence,
                                       stale_result.stopped)
    stale_diff = oracle.compare_run(stale_result, stale_report)
    recorded["stale_probability"] = {
        "broken_rules": sorted({v["rule"] for v in stale_violations}),
        "violations": _json_safe(stale_violations),
        "first_difference": _json_safe(stale_diff[0]) if stale_diff else None,
        "first_difference_rule": (
            "evidence_update_uses_current_allocation_probability"
            if stale_diff and stale_diff[0]["field"].endswith(".evidence")
            else None),
        "correct": {"steps": len(stale_report["trace"]),
                    "counts": list(stale_report["counts"]),
                    "evidence": ftext(stale_report["evidence"]),
                    "stopped": stale_report["stopped"]},
        "mutant": {"steps": len(stale_result.trace),
                   "counts": list(stale_result.counts),
                   "evidence": ftext(stale_result.evidence),
                   "stopped": stale_result.stopped},
        "differs": bool(stale_diff),
    }

    final_result = experiment.replay(FINAL_ONLY_TAPE, 16,
                                     monitor_factory=final_only_factory)
    final_violations = oracle.diagnose(final_result.trace,
                                       final_result.evidence,
                                       final_result.stopped)
    final_monitor_decision = final_result.evidence >= THRESHOLD
    final_reference = oracle.reference_run(FINAL_ONLY_TAPE)
    recorded["final_evidence_only"] = {
        "broken_rules": sorted({v["rule"] for v in final_violations}),
        "violations": _json_safe(final_violations),
        "first_difference": _json_safe(
            oracle.compare_run(final_result, final_reference)[0]),
        "correct": {"steps": len(final_reference["trace"]),
                    "counts": list(final_reference["counts"]),
                    "evidence": ftext(final_reference["evidence"]),
                    "stopped": True},
        "mutant": {"steps": len(final_result.trace),
                   "counts": list(final_result.counts),
                   "evidence": ftext(final_result.evidence),
                   "stopped": final_result.stopped,
                   "horizon_decision": final_monitor_decision},
        "differs": final_monitor_decision is not True,
    }
    return recorded


def _json_safe(value):
    if isinstance(value, Fraction):
        return ftext(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# full verification
# --------------------------------------------------------------------------- #
class VerifyFailure(Exception):
    pass


def _expect(condition, message, details=None):
    if not condition:
        raise VerifyFailure(message if not details else message + ": " + str(details))


def run_regression():
    suite = unittest.TestLoader().loadTestsFromTestCase(
        __import__("checks").Checks)
    stream = _CollectingStream()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    _expect(result.wasSuccessful(), "ordinary regression failed", stream.text())
    return result.testsRun


class _CollectingStream:
    def __init__(self):
        self.parts = []

    def write(self, text):
        self.parts.append(text)

    def flush(self):
        pass

    def text(self):
        return "".join(self.parts)


def run_dynamic_checks(fresh):
    """The independent, re-run part of verify; returns measured pressure facts."""
    notes = []
    increments = [tuple(Fraction(c) for c in poly)
                  for poly in fresh["fn"]["increments"]]
    f16 = [Fraction(c) for c in fresh["fn"]["F16"]]

    # All horizons share the merged DP; each prefix is F_n exactly.
    for horizon in range(0, MAX_HORIZON + 1):
        merged = oracle.cumulative_from_increments(increments[:horizon], horizon)
        for p in (Fraction(0), Fraction(1, 3), Fraction(1, 2),
                  Fraction(2, 3), Fraction(1)):
            if oracle.evaluate_polynomial(merged, p) != \
                    oracle.false_positive_probability(p, horizon):
                raise VerifyFailure("F_%d mismatch at p=%s" % (horizon, p))
    notes.append("horizons 0..16 agree with scalar DP at five rational p")

    # Small horizons against full allocation+outcome enumeration.
    for horizon in range(0, 9):
        enumerated, info = oracle.enumerated_stopping_polynomial(horizon)
        _expect(enumerated == oracle.cumulative_from_increments(
            increments[:horizon], horizon),
            "enumeration mismatch at horizon %d" % horizon)
    try:
        oracle.enumerated_stopping_polynomial(9)
        raise VerifyFailure("enumeration guard allowed horizon 9")
    except ValueError:
        pass
    # At least one small horizon must have a genuine first stop; horizon 6 is
    # the first horizon at which a crossing can happen at all.
    increments_six = oracle.cumulative_from_increments(increments[:6], 6)
    _expect(oracle.evaluate_polynomial(increments_six, Fraction(1, 2)) > 0,
            "horizon 6 has no genuine first-stop mass")
    _expect(all(Fraction(c) == 0 for c in
                oracle.cumulative_from_increments(increments[:5], 5)),
            "no crossing should be possible before horizon 6")
    notes.append("horizons 0..8 match full allocation+outcome enumeration; "
                 "horizon 6 carries genuine first-stop mass; horizon 16 was "
                 "not enumerated")

    # The judge must not sneak in production arithmetic.
    oracle_source = (ROOT / "oracle.py").read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)",
                         oracle_source, re.MULTILINE)
    forbidden = {"allocator", "monitor", "experiment", "checks", "qa"}
    _expect(not (forbidden & set(imports)),
            "oracle.py must not import production code (%s)"
            % sorted(forbidden & set(imports)))
    notes.append("oracle.py imports no production module (independent rules)")

    # Exact envelopes and supremum certificates.
    # Each per-step first-crossing increment is a probability, so it must be
    # non-negative for every real p in [0,1]; checked with a whole-interval
    # Bernstein range, not a sample.  F_n must then be monotone in n.
    for step, increment in enumerate(increments, 1):
        low, _ = oracle.bernstein_range(list(increment))
        _expect(low >= 0,
                "first-crossing increment at step %d is negative on [0,1]"
                % step, low)
    for horizon in range(1, MAX_HORIZON + 1):
        current = oracle.cumulative_from_increments(increments[:horizon], horizon)
        if horizon == 1:
            previous = [Fraction(0)]
        else:
            previous = oracle.cumulative_from_increments(
                increments[:horizon - 1], horizon - 1)
        width = max(len(current), len(previous))
        difference = [((current[i] if i < len(current) else Fraction(0))
                       - (previous[i] if i < len(previous) else Fraction(0)))
                      for i in range(width)]
        low, _ = oracle.bernstein_range(difference)
        _expect(low >= 0, "F_n is not monotone at horizon %d" % horizon, low)
    notes.append("all per-step crossing increments and F_n itself are "
                 "non-negative and monotone on the whole interval (Bernstein range)")

    leaves, meta = oracle.bernstein_cover(f16)
    _expect(oracle.max_leaf_width(leaves) <= TOLERANCE,
            "envelope width exceeds 1/100000")
    cover = oracle.check_cover(leaves)
    _expect(cover["covers_unit_interval"], "cover does not tile [0,1]", cover)
    _expect(cover["measure"] == 1, "cover measure is not 1", cover["measure"])
    _expect(all(oracle.bernstein_roundtrip_ok(leaf, meta["degree"],
                                              meta["basis_lcm"])
                for leaf in leaves), "Bernstein round-trip failed")
    _expect(all(oracle.leaf_value_checks(leaf, f16)[0] for leaf in leaves),
            "leaf bounds do not contain endpoint values")
    _expect(oracle.check_tree(oracle.cover_tree(leaves), leaves),
            "compact cover tree does not reconstruct the dyadic partition")
    lefts = [oracle.leaf_interval(leaf)[0] for leaf in leaves]
    for i in range(0, 1001):
        p = Fraction(i, 1000)
        leaf = leaves[max(0, min(_bisect(lefts, p), len(leaves) - 1))]
        value = oracle.evaluate_polynomial(f16, p)
        _expect(leaf["lo"] <= value <= leaf["hi"],
                "envelope misses F16 at p=%s" % p)
    for p in (Fraction(0), Fraction(1, 1000), Fraction(1, 4), Fraction(1, 3),
              Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
              Fraction(999, 1000), Fraction(1)):
        low, high = oracle.envelope_value(leaves, p)
        _expect(low <= oracle.evaluate_polynomial(f16, p) <= high,
                "envelope lookup misses F16 at p=%s" % p)
    sup_l, sup_u = oracle.sup_bounds(leaves)
    affine = oracle.affine_sup_bounds(f16, pieces=1024)
    combined_l = max(sup_l, affine["lower"])
    combined_u = min(sup_u, affine["upper"])
    _expect(combined_u - combined_l <= TOLERANCE,
            "supremum bounds exceed tolerance", combined_u - combined_l)
    _expect(combined_u <= NOMINAL, "actual worst-case false positive exceeds 1/8",
            combined_u)
    _expect(combined_l <= combined_u, "supremum bounds inverted")
    # The two certificates must be consistent: brackets overlap.
    _expect(max(sup_l, affine["lower"]) <= min(sup_u, affine["upper"]),
            "the two supremum certificates are inconsistent")
    # Affine certificate's Lipschitz error must be genuinely rigorous.
    _expect(affine["upper"] - affine["lower"] <= TOLERANCE,
            "affine supremum bracket exceeds tolerance")
    notes.append("whole-interval envelope + two independent supremum "
                 "certificates (Bernstein hull and chord/|F''| bound) agree")

    # Adaptive allocation / evidence handoff through production factories.
    seen = []

    class RecordingAllocator(Allocator):
        def probability(self):
            q = super().probability()
            seen.append(("allocator", self.counts, q))
            return q

    class RecordingMonitor(Monitor):
        def observe(self, probability, arm, outcome):
            seen.append(("monitor", probability, arm, outcome))
            return super().observe(probability, arm, outcome)

    seen.clear()
    produced = experiment.replay(
        SAMPLE, 16, allocator_factory=RecordingAllocator,
        monitor_factory=RecordingMonitor)
    reference = oracle.reference_run(SAMPLE, 16)
    _expect(not oracle.compare_run(produced, reference),
            "production replay differs from reference",
            oracle.compare_run(produced, reference))
    allocator_events = [e for e in seen if e[0] == "allocator"]
    monitor_events = [e for e in seen if e[0] == "monitor"]
    _expect(len(allocator_events) == len(reference["trace"]),
            "allocator probability() call count differs")
    for index, (event, step) in enumerate(zip(allocator_events,
                                              reference["trace"]), 1):
        _expect(event[1] == step["before"],
                "allocator counted state before step %d differs" % index)
        _expect(event[2] == step["probability"],
                "allocator q at step %d differs from reference" % index)
    _expect(len(monitor_events) == len(reference["trace"]),
            "monitor observe count differs (consumption after stop?)")
    for index, (event, step) in enumerate(zip(monitor_events,
                                              reference["trace"]), 1):
        _expect(event[1] == step["probability"],
                "monitor received q=%s at step %d; pre-step q was %s"
                % (event[1], index, step["probability"]))
    notes.append("recorded production handoff: monitor q equals pre-step "
                 "allocator q on every step, no post-stop observation")

    # No production deviation may exist.
    violations = oracle.diagnose(produced.trace, produced.evidence,
                                 produced.stopped)
    _expect(not violations, "production trace violates a business rule",
            _json_safe(violations))

    # Random regression is reproducibility glue, not probability evidence.
    for seed in (0, 1, 19, 31):
        p = Fraction(seed, 100)
        left = experiment.simulate(p, Random(seed), 16)
        right = experiment.simulate(p, Random(seed), 16)
        _expect(left == right, "simulate not reproducible at seed %d" % seed)
        _expect(not oracle.compare_run(left, oracle.simulate_reference(
            p, Random(seed), 16)), "simulate differs from reference at seed %d" % seed)
    notes.append("fixed-seed random runs reproducible and identical to reference")

    # Production entry points must also honour equality-at-threshold and the
    # fall-back tape; otherwise a one-sided comparison could hide a deviation.
    for name, tape, horizon in (("equality", EQUALITY_TAPE, 16),
                                ("fallback", FALLBACK_TAPE, 16)):
        production = experiment.replay(tape, horizon)
        independent = oracle.reference_run(tape, horizon)
        _expect(not oracle.compare_run(production, independent),
                "production replay differs from reference on %s tape" % name,
                oracle.compare_run(production, independent))
        _expect(not oracle.diagnose(production.trace, production.evidence,
                                    production.stopped),
                "production %s tape violates a business rule" % name,
                _json_safe(oracle.diagnose(production.trace,
                                           production.evidence,
                                           production.stopped)))
    notes.append("production replay rechecked on equality and fall-back tapes")

    # Threshold equality and fallback.
    equality = oracle.reference_run(EQUALITY_TAPE)
    _expect(equality["stopped"] and equality["evidence"] == THRESHOLD
            and len(equality["trace"]) == 6,
            "equality case must stop exactly at E=8 on step 6", equality)
    fallback = oracle.reference_run(FALLBACK_TAPE, horizon=8, force_continue=True)
    _expect(fallback["first_crossing"] == 6
            and fallback["evidence"] < THRESHOLD,
            "continued observation must reveal a fall back below 8", fallback)
    _expect(oracle.reference_run(FALLBACK_TAPE)["consumed"] == 6,
            "significance event must remain the first crossing")
    notes.append("threshold equality stops; hypothetical continuation can fall "
                 "below threshold but cannot replace the first-crossing event")

    # Same counts, different evidence; zero evidence; endpoints; empty horizon.
    a = oracle.reference_run([("A", 0), ("A", 1), ("B", 1)], 3)
    b = oracle.reference_run([("A", 0), ("B", 1), ("A", 1)], 3)
    _expect(a["counts"] == b["counts"] == (1, 1, 1, 0),
            "collision scenario counts wrong")
    _expect(a["evidence"] != b["evidence"],
            "same counts must be able to carry different evidence")
    zero = oracle.reference_run([("A", 1), ("B", 0), ("B", 1)], 3)
    _expect(zero["evidence"] == 0 and not zero["stopped"],
            "zero evidence branch not exercised", zero["evidence"])
    _expect(zero["counts"] == (1, 0, 1, 1), "zero evidence counts wrong")
    empty = oracle.reference_run([], 0)
    _expect(empty["trace"] == [] and empty["evidence"] == 1
            and empty["stopped"] is False, "empty horizon wrong")
    _expect(oracle.evaluate_polynomial(f16, Fraction(0)) == 0,
            "F16(0) must be zero")
    _expect(oracle.evaluate_polynomial(f16, Fraction(1)) > 0,
            "F16(1) must be positive")
    notes.append("adaptive switch, count/evidence collision, zero evidence, "
                 "p endpoints and empty horizon all rechecked")

    # Mutants must be detected *by rule*, with a real trace difference.
    stale = experiment.replay(SAMPLE, 16, monitor_factory=stale_factory)
    stale_reference = oracle.reference_run(SAMPLE)
    stale_violations = oracle.diagnose(stale.trace, stale.evidence, stale.stopped)
    stale_rules = {v["rule"] for v in stale_violations}
    _expect("evidence_update_uses_current_allocation_probability" in stale_rules,
            "stale-q mutant not detected by its broken rule", stale_rules)
    _expect(any(v.get("stale_probability_explains") == Fraction(1, 2)
                for v in stale_violations),
            "stale-q mutant not explained by an ignored probability of 1/2")
    _expect("allocation_probability_from_prior_counts" not in stale_rules,
            "stale-q mutant must not be misreported as an allocation fault")
    _expect(oracle.compare_run(stale, stale_reference),
            "stale-q mutant produced no trace difference")
    first_difference = oracle.compare_run(stale, stale_reference)[0]
    _expect(first_difference["field"] == "step.3.evidence"
            and first_difference["expected"] == Fraction(2)
            and first_difference["actual"] == Fraction(9, 4),
            "stale-q first divergence is not the expected step-3 evidence",
            _json_safe(first_difference))
    _expect(stale.evidence != stale_reference["evidence"]
            or len(stale.trace) != len(stale_reference["trace"]),
            "stale-q mutant has identical behaviour")

    final = experiment.replay(FINAL_ONLY_TAPE, 16,
                              monitor_factory=final_only_factory)
    final_violations = oracle.diagnose(final.trace, final.evidence, final.stopped)
    final_rules = {v["rule"] for v in final_violations}
    _expect("first_crossing_must_stop" in final_rules,
            "final-only mutant not detected as violating first-crossing stop",
            final_rules)
    _expect("stop_requires_threshold" not in final_rules,
            "final-only mutant must not be misreported as a stop-threshold fault",
            final_rules)
    crossing_fault = next(v for v in final_violations
                          if v["rule"] == "first_crossing_must_stop")
    _expect(crossing_fault["index"] == 8
            and crossing_fault["evidence"] == Fraction(2048, 243),
            "final-only mutant must be pinned to the step-8 crossing",
            _json_safe(crossing_fault))
    _expect(len(final.trace) == 16 and final.evidence < THRESHOLD,
            "final-only mutant should reach the horizon with E < 8 on this tape",
            {"steps": len(final.trace), "evidence": final.evidence})
    _expect(oracle.reference_run(FINAL_ONLY_TAPE)["stopped"]
            and oracle.reference_run(FINAL_ONLY_TAPE)["consumed"] == 8,
            "reference must stop at step 8 on the final-only tape")
    notes.append("both executable mutants detected by the violated rule, with "
                 "recorded trace/probability differences")

    # Finite stress: measured fresh every run, deterministic sizes recorded.
    stress = {}
    stress["dh16"] = _stressed(
        lambda: oracle.stopping_increments(MAX_HORIZON), "polynomial DP horizon 16")
    _, dp_stats = oracle.stopping_increments(MAX_HORIZON)
    stress["dh16"]["deterministic"] = {k: int(v) for k, v in dp_stats.items()}
    stress["cover"] = _stressed(lambda: oracle.bernstein_cover(f16),
                                "Bernstein envelope for F16")
    stress["cover"]["deterministic"] = {"leaf_count": len(leaves),
                                        "max_depth": cover["max_depth"]}
    tape = FINAL_ONLY_TAPE

    def replay_batch():
        total = 0
        for _ in range(300):
            run = experiment.replay(tape, 16)
            total += len(run.trace)
        return total

    stress["replay300"] = _stressed(replay_batch, "300 production replays")
    stress["replay300"]["deterministic"] = {"steps_processed": 300 * 8}
    _expect(stress["replay300"]["deterministic"]["steps_processed"]
            == stress["replay300"]["value"], "replay stress counted wrong")
    notes.append("finite stress measured with tracemalloc; sizes deterministic")

    return {
        "notes": notes,
        "sup_l": ftext(combined_l), "sup_u": ftext(combined_u),
        "sup_gap": ftext(combined_u - combined_l),
        "max_width": ftext(oracle.max_leaf_width(leaves)),
        "leaves": len(leaves), "max_depth": cover["max_depth"],
        "f16_1": ftext(oracle.evaluate_polynomial(f16, Fraction(1))),
        "f16_half": ftext(oracle.evaluate_polynomial(f16, Fraction(1, 2))),
        "stress": {k: _json_safe(v) for k, v in stress.items()},
    }


def _stressed(function, label):
    tracemalloc.start()
    start = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"label": label, "seconds": round(elapsed, 6),
            "peak_bytes": int(peak), "value": value}


def report_conclusions(fresh):
    f16 = [Fraction(c) for c in fresh["fn"]["F16"]]
    f = lambda p: ftext(oracle.evaluate_polynomial(f16, p))  # noqa: E731
    sup = fresh["sup"]["combined"]
    widest = fresh["envelope"]["widest_leaf"]
    sample = fresh["sample"]
    stale = fresh["mutations"]["stale_probability"]
    final = fresh["mutations"]["final_evidence_only"]
    return [
        "F_16(1/2) = %s" % f(Fraction(1, 2)),
        "F_16(1) = %s" % f(Fraction(1)),
        "sup_{p in [0,1]} F_16(p) in [%s, %s]" % (sup["L"], sup["U"]),
        "supremum bracket width = %s" % sup["gap"],
        "max envelope width = %s" % widest["width"],
        "sample stops at step %d with evidence %s" % (sample["steps"],
                                                      sample["evidence"]),
        "final-only mutant reaches %d observations with final evidence %s"
        % (final["mutant"]["steps"], final["mutant"]["evidence"]),
        "stale-q mutant stops at step %d with evidence %s"
        % (stale["mutant"]["steps"], stale["mutant"]["evidence"]),
    ]


def verify():
    # Cheap integrity gate first: a tampered production file or a missing
    # evidence file must block before the expensive recomputation runs.
    print("== integrity gate ==")
    _expect(EVIDENCE.exists(), "evidence.json is missing; run qa.py build")
    saved = json.loads(EVIDENCE.read_text())
    current_hashes = source_hashes()
    recorded_hashes = frozen_source_hashes(saved)
    _expect(recorded_hashes == current_hashes,
            "production source changed after evidence was built",
            {name: {"recorded": recorded_hashes.get(name),
                    "current": current_hashes.get(name)}
             for name in PRODUCTION_FILES
             if recorded_hashes.get(name) != current_hashes.get(name)})
    _expect(not BLOCKERS.exists(),
            "deviations.json present: an unresolved deviation blocks release; "
            "review it and remove it only after a human resolution (%s)"
            % (json.loads(BLOCKERS.read_text()).get("reason", "no reason")
               if BLOCKERS.exists() else "no reason"))
    print("   production source hashes match the frozen build")
    print("   no unresolved deviations.json")

    print("== normal regression (checks.py, unchanged) ==")
    test_count = run_regression()
    print("   %d ordinary tests passed" % test_count)

    print("== independent judge and whole-interval certificate ==")
    fresh = build_evidence()
    measured = run_dynamic_checks(fresh)
    for note in measured["notes"]:
        print("   ok:", note)
    print("   certified sup bracket: [%s, %s], width %s"
          % (measured["sup_l"], measured["sup_u"], measured["sup_gap"]))
    print("   envelope leaves %d, max depth %d, max width %s"
          % (measured["leaves"], measured["max_depth"], measured["max_width"]))
    print("   finite stress (fresh measurements):")
    for key, record in measured["stress"].items():
        print("      %-10s %-28s %8.3f s  peak %10d B  %s"
              % (key, record["label"], record["seconds"],
                 record["peak_bytes"], record["deterministic"]))

    print("== saved evidence comparison ==")
    expected_deterministic = evidence_deterministic(fresh)
    _expect(saved.get("deterministic") == expected_deterministic,
            "saved evidence.json does not match a fresh derivation")
    _expect(saved.get("report_conclusions") == report_conclusions(fresh),
            "saved report conclusions do not match recomputation")
    print("   evidence.json matches fresh derivation exactly")

    print("== report conclusions ==")
    _expect(REPORT.exists(), "REPORT.md is missing")
    report_text = REPORT.read_text()
    for line in report_conclusions(fresh):
        _expect(line in report_text, "REPORT.md missing conclusion: %s" % line)
    for keyword in ("F_16(1)", "sup_{p", "1/100000", "变异"):
        _expect(keyword in report_text,
                "REPORT.md lacks discussion of %s" % keyword)
    for forbidden_claim in ("网格最大值可作为", "蒙特卡洛最大值作为"):
        _expect(forbidden_claim not in report_text,
                "REPORT.md relies on a grid/Monte-Carlo maximum: %s"
                % forbidden_claim)
    print("   REPORT.md contains every recomputed business conclusion")

    # Release opinion is conditional but may only be issued when all pass.
    _expect(Fraction(measured["sup_u"]) <= NOMINAL,
            "actual false-positive upper bound exceeds 1/8")
    _expect(not BLOCKERS.exists(),
            "deviations.json present: unresolved deviation blocks release")
    print("\nRESULT: PASS")
    print("release opinion: PASS - global envelope and supremum certificate "
          "valid, actual worst-case false positive <= 1/8, no production "
          "deviation, both mutants detected by rule.")
    return 0


def evidence_deterministic(fresh):
    keys = ("fn", "enumeration", "enumeration_guard", "scalar", "envelope",
            "sup", "envelope_scan", "envelope_lookup", "sample", "threshold_equality",
            "fallback_after_crossing", "same_counts_different_evidence",
            "zero_evidence", "empty_horizon", "endpoints", "integration",
            "mutant_false_positive_rates", "mutations")
    return {key: _json_safe(fresh[key]) for key in keys}


# Production hashes are frozen by the build and must not drift: a changed
# allocator/monitor/experiment invalidates the recorded evidence even if the
# numbers happen to still agree.
def frozen_source_hashes(saved):
    return saved.get("source_hashes", {})


def build():
    fresh = build_evidence()
    document = {
        "schema": "adaptive-monitor-quality-evidence/1",
        "generator": "python3 qa.py build (oracle.py only; production imports "
                     "used exclusively in the integration section)",
        "source_hashes": source_hashes(),
        "deterministic": evidence_deterministic(fresh),
        "report_conclusions": report_conclusions(fresh),
    }
    EVIDENCE.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    print("wrote %s (%d bytes)" % (EVIDENCE.name, EVIDENCE.stat().st_size))
    print("F_16 digest:", fresh["fn"]["F16_digest"])
    print("envelope digest:", fresh["envelope"]["digest"])
    print("sup bracket:", fresh["sup"]["combined"])
    for line in document["report_conclusions"]:
        print(" -", line)
    return 0


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def demo():
    print("== real allocation switch and stop (production replay) ==")
    result = experiment.replay(SAMPLE, 16)
    for step in result.trace:
        print("  step %2d  before=%s  q=%s  %s/%d  after=%s  E=%s%s"
              % (step.index, list(step.before), step.probability, step.arm,
                 step.outcome, list(step.after), step.evidence,
                 "  STOP" if step.stopped else ""))
    print("  result: steps=%d counts=%s E=%s stopped=%s"
          % (len(result.trace), list(result.counts), result.evidence,
             result.stopped))
    print("  q switched from 1/2 to 3/4 only after counts made A look better;")
    print("  the 9th tape item (B,1) was never consumed.")

    print("\n== mutant: analysis keeps the stale initial probability 1/2 ==")
    stale = experiment.replay(SAMPLE, 16, monitor_factory=stale_factory)
    for left, right in zip(stale.trace, result.trace):
        marker = "" if left.evidence == right.evidence else "  <-- diverges"
        print("  step %2d  q=%s  %s/%d  correct E=%s  mutant E=%s%s"
              % (left.index, left.probability, left.arm, left.outcome,
                 right.evidence, left.evidence, marker))
    print("  correct: stop at step %d E=%s" % (len(result.trace),
                                               result.evidence))
    print("  mutant : stop at step %d E=%s" % (len(stale.trace),
                                               stale.evidence))
    found = oracle.diagnose(stale.trace, stale.evidence, stale.stopped)
    rules = sorted({v["rule"] for v in found})
    print("  independent judge identifies broken rule(s): %s" % ", ".join(rules))
    detail = next(v for v in found
                  if v["rule"] == "evidence_update_uses_current_allocation_probability")
    print("  first evidence fault at step %d: expected %s, actual %s "
          "(stale q = %s explains it)"
          % (detail["index"], detail["expected"], detail["actual"],
             detail["stale_probability_explains"]))
    print("  business difference: the mutant stops one observation early "
          "(%d vs %d observations), so the reported experiment is premature."
          % (len(stale.trace), len(result.trace)))

    print("\n== mutant: continue to horizon, judge only final evidence ==")
    final = experiment.replay(FINAL_ONLY_TAPE, 16,
                              monitor_factory=final_only_factory)
    reference = oracle.reference_run(FINAL_ONLY_TAPE)
    print("  correct: stop at step %d, E=%s, stopped=%s"
          % (len(reference["trace"]), reference["evidence"], reference["stopped"]))
    print("  mutant : %d observations, final E=%s, horizon verdict=%s"
          % (len(final.trace), final.evidence, final.evidence >= THRESHOLD))
    violations = oracle.diagnose(final.trace, final.evidence, final.stopped)
    print("  independent judge identifies broken rule(s): %s"
          % ", ".join(sorted({v["rule"] for v in violations})))
    print("  business difference: the significant early crossing is missed, "
          "so a real effect is reported as not significant.")
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv):
    if len(argv) != 2 or argv[1] not in ("build", "verify", "demo"):
        print("usage: qa.py {build|verify|demo}", file=sys.stderr)
        return 64
    command = argv[1]
    if command == "build":
        return build()
    if command == "demo":
        return demo()
    try:
        return verify()
    except VerifyFailure as failure:
        payload = {"blocked": True, "reason": str(failure),
                   "replay": {"sample_tape": SAMPLE,
                              "final_only_tape": FINAL_ONLY_TAPE,
                              "equality_tape": EQUALITY_TAPE,
                              "fallback_tape": FALLBACK_TAPE}}
        BLOCKERS.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
        print("\nRESULT: BLOCKED - %s" % failure)
        print("wrote deviations.json; release opinion withheld.")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
