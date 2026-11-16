"""Independent exact judge for the adaptive allocation / continuous-monitoring contract.

This module deliberately does not import allocator.py, monitor.py or experiment.py.
Every rule below is re-derived from CONTRACT.md:

* allocation probability q is computed from the counts *before* the current step;
* the allocation draw uses q / (1-q);
* the outcome then updates evidence with the exact factor
      1 + 1/(4q)      (arm A, y = 1)
      1 - 1/(4(1-q))  (arm B, y = 1)
      1               (y = 0)
  and is multiplied by the outcome mass p or 1-p under H0;
* the first step whose evidence is >= 8 stops the experiment (equality included),
  and no later tape item may be consumed.

F_n(p) is the exact polynomial probability that a first crossing happens at or
before n under a single common success probability p.  It is obtained by an
exact dynamic program over (counts, evidence) states; histories with the same
counts and same evidence are merged, which is why horizon 16 is not enumerated
as 4**16 histories.  The same quantity is cross-checked for small horizons by
full enumeration over allocation *and* outcome histories and by an independent
scalar DP at rational p.

The whole-interval enclosure of F_16 is a dyadic Bernstein hull computed in
integer arithmetic.  For every dyadic leaf the polynomial is re-expressed in
the Bernstein basis whose coefficients bound the polynomial on that leaf; the
hull is the piecewise lower/upper step function.  The union of the leaves is
checked to be exactly [0, 1], every leaf width is checked against the contract
tolerance, and the Bernstein representation is checked by exact coefficient
round-trip.  This yields rational bounds on sup_p F_16(p) over the *whole*
interval, not a grid or Monte-Carlo maximum.
"""
from fractions import Fraction
from hashlib import sha256
from math import comb, lcm
import bisect

THRESHOLD = Fraction(8)
MAX_HORIZON = 16
ENVELOPE_TOLERANCE = Fraction(1, 100000)
NOMINAL_LIMIT = Fraction(1, 8)
ALLOWED_Q = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))


# --------------------------------------------------------------------------- #
# basic exact rules
# --------------------------------------------------------------------------- #
def validate_observation(arm, outcome):
    if arm not in ("A", "B") or type(outcome) is not int or outcome not in (0, 1):
        raise ValueError("expected arm A/B and non-boolean integer outcome 0/1")


def validate_counts(counts):
    if (not isinstance(counts, tuple) or len(counts) != 4
            or any(type(x) is not int or x < 0 for x in counts)):
        raise ValueError("counts must be a 4-tuple of non-negative integers")


def allocation_probability(counts):
    """q from the counts before this step (contract section '实验规则')."""
    validate_counts(counts)
    a, af, b, bf = counts
    na, nb = a + af, b + bf
    if na == 0 or nb == 0:
        return Fraction(1, 2)
    left, right = a * nb, b * na
    if left > right:
        return Fraction(3, 4)
    if left < right:
        return Fraction(1, 4)
    return Fraction(1, 2)


def evidence_factor(q, arm, outcome):
    """Multiplicative evidence update for one observed outcome."""
    validate_observation(arm, outcome)
    if q not in ALLOWED_Q:
        raise ValueError("allocation probability must be 1/4, 1/2 or 3/4")
    if outcome == 0:
        return Fraction(1)
    if arm == "A":
        return 1 + Fraction(1, 4) / q
    return 1 - Fraction(1, 4) / (1 - q)


def _apply(counts, arm, outcome):
    a, af, b, bf = counts
    if arm == "A":
        return (a + 1, af, b, bf) if outcome else (a, af + 1, b, bf)
    return (a, af, b + 1, bf) if outcome else (a, af, b, bf + 1)


def validate_horizon(horizon):
    if type(horizon) is not int or not 0 <= horizon <= MAX_HORIZON:
        raise ValueError("horizon must be a non-boolean integer in [0,16]")


# --------------------------------------------------------------------------- #
# independent replay
# --------------------------------------------------------------------------- #
def reference_run(tape, horizon=MAX_HORIZON, force_continue=False):
    """Replay a tape with the rules above.

    Returns a dict with 'trace' (list of dicts), 'counts', 'evidence',
    'stopped', 'consumed' (number of pulled tape items) and 'first_crossing'
    (1-based index or None).  With force_continue=True the run never stops;
    this is used only to demonstrate that evidence can fall back after a
    crossing, never to define the significance event.
    """
    validate_horizon(horizon)
    iterator = iter(tape)
    counts = (0, 0, 0, 0)
    evidence = Fraction(1)
    trace = []
    consumed = 0
    stopped = False
    first_crossing = None
    for index in range(1, horizon + 1):
        probability = allocation_probability(counts)
        try:
            item = next(iterator)
        except StopIteration:
            raise ValueError("tape exhausted before horizon or stopping") from None
        consumed += 1
        if (not isinstance(item, (tuple, list)) or len(item) != 2):
            raise ValueError("tape items must be (arm, outcome) pairs")
        arm, outcome = item
        validate_observation(arm, outcome)
        before = counts
        evidence = evidence * evidence_factor(probability, arm, outcome)
        counts = _apply(before, arm, outcome)
        crossed = evidence >= THRESHOLD
        if crossed and first_crossing is None:
            first_crossing = index
        trace.append({
            "index": index, "before": before, "probability": probability,
            "arm": arm, "outcome": outcome, "after": counts,
            "evidence": evidence, "stopped": (crossed or stopped) and not force_continue,
        })
        if crossed and not force_continue:
            stopped = True
            break
    return {
        "trace": trace, "counts": counts, "evidence": evidence,
        "stopped": stopped, "consumed": consumed,
        "first_crossing": first_crossing,
    }


# --------------------------------------------------------------------------- #
# exact polynomial dynamic program for F_n(p)
# --------------------------------------------------------------------------- #
def _poly_mul_p(poly):
    return [Fraction(0)] + list(poly)


def _poly_mul_one_minus_p(poly):
    out = [Fraction(0)] * (len(poly) + 1)
    for i, c in enumerate(poly):
        out[i] += c
        out[i + 1] -= c
    return out


def _poly_add_into(dst, src):
    if len(dst) < len(src):
        dst.extend([Fraction(0)] * (len(src) - len(dst)))
    for i, c in enumerate(src):
        dst[i] += c


def _scaled(poly, weight):
    return [c * weight for c in poly]


def stopping_increments(horizon=MAX_HORIZON):
    """Exact probabilities that the *first* crossing happens at step k+1.

    Returns (increments, stats). increments[k] is a polynomial in p of degree
    at most k+1 for k = 0..horizon-1.  States with identical counts and
    identical evidence are merged (evidence is a function of the path), which
    keeps the state space finite and far below 4**horizon.
    """
    validate_horizon(horizon)
    states = {(0, 0, 0, 0, Fraction(1)): [Fraction(1)]}
    increments = []
    stats = {"max_live_states": 1, "states_seen": 1, "transitions": 0}
    for step in range(horizon):
        nxt = {}
        stop = [Fraction(0)] * (step + 2)
        transitions = 0
        for (a, af, b, bf, evidence), poly in states.items():
            q = allocation_probability((a, af, b, bf))
            for arm in ("A", "B"):
                alloc_weight = q if arm == "A" else 1 - q
                for outcome in (0, 1):
                    transitions += 1
                    weights = (_poly_mul_p(poly) if outcome
                               else _poly_mul_one_minus_p(poly))
                    weighted = _scaled(weights, alloc_weight)
                    new_evidence = evidence * evidence_factor(q, arm, outcome)
                    if new_evidence >= THRESHOLD:
                        _poly_add_into(stop, weighted)
                    else:
                        key = _apply((a, af, b, bf), arm, outcome) + (new_evidence,)
                        if key in nxt:
                            _poly_add_into(nxt[key], weighted)
                        else:
                            nxt[key] = weighted
        increments.append(tuple(stop))
        states = nxt
        stats["transitions"] += transitions
        stats["max_live_states"] = max(stats["max_live_states"], len(states))
        stats["states_seen"] += len(states)
    return increments, stats


def cumulative_from_increments(increments, horizon):
    cumulative = [Fraction(0)] * (horizon + 1)
    for poly in increments:
        for i, c in enumerate(poly):
            cumulative[i] += c
    return cumulative


def stopping_polynomial(horizon=MAX_HORIZON):
    """F_horizon(p) as an exact coefficient list (degree <= horizon)."""
    increments, stats = stopping_increments(horizon)
    return cumulative_from_increments(increments, horizon), increments, stats


def evaluate_polynomial(poly, p):
    if not isinstance(p, Fraction) or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    total = Fraction(0)
    power = Fraction(1)
    for coefficient in poly:
        total += coefficient * power
        power *= p
    return total


def _scalar_stopping_probability(p, horizon, factor_fn):
    states = {(0, 0, 0, 0, Fraction(1)): Fraction(1)}
    total = Fraction(0)
    for _ in range(horizon):
        nxt = {}
        for (a, af, b, bf, evidence), weight in states.items():
            q = allocation_probability((a, af, b, bf))
            for arm in ("A", "B"):
                alloc = q if arm == "A" else 1 - q
                for outcome in (0, 1):
                    path_weight = weight * alloc * (p if outcome else 1 - p)
                    if path_weight == 0:
                        continue
                    new_evidence = evidence * factor_fn(q, arm, outcome)
                    if new_evidence >= THRESHOLD:
                        total += path_weight
                    else:
                        key = _apply((a, af, b, bf), arm, outcome) + (new_evidence,)
                        nxt[key] = nxt.get(key, Fraction(0)) + path_weight
        states = nxt
    return total


def false_positive_probability(p, horizon=MAX_HORIZON):
    """Independent scalar DP at a single rational p (no polynomial algebra)."""
    if type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    validate_horizon(horizon)
    return _scalar_stopping_probability(p, horizon, evidence_factor)


def stale_probability_false_positive(p, horizon=MAX_HORIZON):
    """False-positive risk of a monitor that always updates evidence with q=1/2.

    This is the executable mutation "analysis keeps the initial allocation
    probability".  Used to quantify the business damage of the deviation.
    """
    if type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    validate_horizon(horizon)
    return _scalar_stopping_probability(
        p, horizon, lambda q, arm, y: evidence_factor(Fraction(1, 2), arm, y))


def final_evidence_only_probability(p, horizon=MAX_HORIZON):
    """Probability that a no-early-stop monitor ends at the horizon with E >= 8.

    This is the executable mutation "look only at the final evidence"; it is
    not the contract's first-crossing event and can both miss and misreport.
    """
    if type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    validate_horizon(horizon)
    states = {(0, 0, 0, 0, Fraction(1)): Fraction(1)}
    for _ in range(horizon):
        nxt = {}
        for (a, af, b, bf, evidence), weight in states.items():
            q = allocation_probability((a, af, b, bf))
            for arm in ("A", "B"):
                alloc = q if arm == "A" else 1 - q
                for outcome in (0, 1):
                    path_weight = weight * alloc * (p if outcome else 1 - p)
                    if path_weight == 0:
                        continue
                    new_evidence = evidence * evidence_factor(q, arm, outcome)
                    key = _apply((a, af, b, bf), arm, outcome) + (new_evidence,)
                    nxt[key] = nxt.get(key, Fraction(0)) + path_weight
        states = nxt
    return sum((weight for (_, _, _, _, evidence), weight in states.items()
                if evidence >= THRESHOLD), Fraction(0))


def enumerated_stopping_polynomial(horizon):
    """Full enumeration of allocation *and* outcome histories (small n only).

    Deliberately refuses horizons above 8: the contract forbids handling the
    16-step configuration by enumerating every allocation/outcome history
    (4**16 = 4294967296 paths).  The merged-state DP above is used for 16.
    """
    validate_horizon(horizon)
    if horizon > 8:
        raise ValueError("full history enumeration is only allowed for horizon <= 8")
    total = [Fraction(0)] * (horizon + 1)
    paths = [0]

    def walk(step, counts, evidence, poly):
        if step == horizon:
            return
        q = allocation_probability(counts)
        for arm in ("A", "B"):
            alloc = q if arm == "A" else 1 - q
            for outcome in (0, 1):
                paths[0] += 1
                weights = (_poly_mul_p(poly) if outcome
                           else _poly_mul_one_minus_p(poly))
                weighted = _scaled(weights, alloc)
                new_evidence = evidence * evidence_factor(q, arm, outcome)
                if new_evidence >= THRESHOLD:
                    for i, c in enumerate(weighted):
                        total[i] += c
                else:
                    walk(step + 1, _apply(counts, arm, outcome),
                         new_evidence, weighted)

    walk(0, (0, 0, 0, 0), Fraction(1), [Fraction(1)])
    return total, {"paths_visited": paths[0]}


def simulate_reference(p, rng, horizon=MAX_HORIZON):
    """Independent simulator using the contract's rng protocol.

    rng.randrange(n) is called first for the allocation draw and then for the
    Bernoulli outcome, exactly as the contract specifies.  No production code
    is imported or reused.
    """
    if type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    validate_horizon(horizon)
    counts = (0, 0, 0, 0)
    evidence = Fraction(1)
    trace = []
    first_crossing = None
    for index in range(1, horizon + 1):
        q = allocation_probability(counts)
        arm = "A" if rng.randrange(q.denominator) < q.numerator else "B"
        outcome = int(rng.randrange(p.denominator) < p.numerator)
        before = counts
        evidence = evidence * evidence_factor(q, arm, outcome)
        counts = _apply(before, arm, outcome)
        crossed = evidence >= THRESHOLD
        if crossed and first_crossing is None:
            first_crossing = index
        trace.append({"index": index, "before": before, "probability": q,
                      "arm": arm, "outcome": outcome, "after": counts,
                      "evidence": evidence, "stopped": crossed})
        if crossed:
            break
    return {"trace": trace, "counts": counts, "evidence": evidence,
            "stopped": bool(trace and trace[-1]["stopped"]),
            "consumed": len(trace), "first_crossing": first_crossing}


# --------------------------------------------------------------------------- #
# whole-interval Bernstein enclosure and certified sup bounds
# --------------------------------------------------------------------------- #
def _common_denominator(poly):
    denominator = 1
    for coefficient in poly:
        denominator = lcm(denominator, coefficient.denominator)
    return denominator


def _bernstein_bounds(ints, n, leaf_index, m, basis):
    """Integer-scaled hull of P((i+t)/2^m) on t in [0,1].

    H(t) = 2**(n*m) * P((i+t)/2**m) has integer coefficients h[r].
    B[j] = L * b_j where b_j are the Bernstein coefficients of H; both are
    exact integers, and P((i+t)/2^m) = sum_j b_j * C(n,j) t^j (1-t)^(n-j).
    Everything below is integer arithmetic.  Returns h and b.
    """
    h = [0] * (n + 1)
    power = [1]
    for k in range(n + 1):
        coefficient = ints[k] << ((n - k) * m) if ints[k] else 0
        if coefficient:
            for r, pc in enumerate(power):
                h[r] += coefficient * pc
        if k < n:
            nxt = [0] * (len(power) + 1)
            for r, pc in enumerate(power):
                nxt[r] += pc * leaf_index
                nxt[r + 1] += pc
            power = nxt
    b = [0] * (n + 1)
    for j in range(n + 1):
        s = 0
        row = basis[j]
        for k in range(j + 1):
            s += h[k] * row[k]
        b[j] = s
    return h, b


def bernstein_cover(poly, tolerance=ENVELOPE_TOLERANCE):
    """Dyadic Bernstein hull of a polynomial on [0,1].

    The returned leaves tile [0,1]; each leaf carries exact rational lower and
    upper bounds whose difference is <= tolerance.  Bounds are valid on the
    whole leaf, so the piecewise step functions enclose the polynomial for
    every real p in [0,1].  On leaf [i/2^m, (i+1)/2^m] the value of P is
    sum_j b_j * C(n,j) t^j (1-t)^(n-j) with t in [0,1], hence it lies between
    min b_j and max b_j; b_j are computed exactly as integers scaled by
    2**(n*m) and the basis LCM.
    """
    n = len(poly) - 1
    if n < 0:
        raise ValueError("empty polynomial")
    den = _common_denominator(poly)
    ints = [c.numerator * (den // c.denominator) for c in poly]
    binom_row = [comb(n, k) for k in range(n + 1)]
    lcm_basis = lcm(*binom_row) if n else 1
    basis = [[comb(j, k) * (lcm_basis // binom_row[k]) for k in range(j + 1)]
             for j in range(n + 1)]
    tol_num, tol_den = tolerance.numerator, tolerance.denominator
    leaves = []
    depth_limit = 48

    def subdivide(i, m):
        h, b = _bernstein_bounds(ints, n, i, m, basis)
        lo, hi = min(b), max(b)
        # width = (hi - lo) / (den * lcm_basis * 2**(n*m)) <= tol_num/tol_den
        if (hi - lo) * tol_den <= tol_num * den * lcm_basis * (1 << (n * m)) \
                or m >= depth_limit:
            divisor = den * lcm_basis * (1 << (n * m))
            leaves.append({"i": i, "m": m, "lo": Fraction(lo, divisor),
                           "hi": Fraction(hi, divisor),
                           "lo_scaled": lo, "hi_scaled": hi,
                           "divisor": divisor, "h": h, "b": b})
        else:
            subdivide(2 * i, m + 1)
            subdivide(2 * i + 1, m + 1)

    subdivide(0, 0)
    return leaves, {"degree": n, "denominator": den, "basis_lcm": lcm_basis,
                    "tolerance": tolerance}


def leaf_interval(leaf):
    return Fraction(leaf["i"], 1 << leaf["m"]), Fraction(leaf["i"] + 1, 1 << leaf["m"])


def check_cover(leaves):
    """Verify the leaves tile exactly [0,1] (no gap, no overlap, full measure)."""
    spans = sorted((Fraction(leaf["i"], 1 << leaf["m"]),
                    Fraction(leaf["i"] + 1, 1 << leaf["m"])) for leaf in leaves)
    gap_free = spans[0][0] == 0 and spans[-1][1] == 1
    if gap_free:
        for (_, left_end), (right_start, _) in zip(spans, spans[1:]):
            if left_end != right_start:
                gap_free = False
                break
    measure = sum((right - left for left, right in spans), Fraction(0))
    return {"covers_unit_interval": gap_free, "measure": measure,
            "leaf_count": len(leaves),
            "max_depth": max(leaf["m"] for leaf in leaves)}


def bernstein_roundtrip_ok(leaf, n, lcm_basis):
    """Exact check that leaf['b'] really are L times the Bernstein coeffs of H."""
    h, b = leaf["h"], leaf["b"]
    reconstructed = [0] * (n + 1)
    for j in range(n + 1):
        base = b[j] * comb(n, j)
        for r in range(n - j + 1):
            reconstructed[j + r] += base * comb(n - j, r) * (-1) ** r
    return reconstructed == [c * lcm_basis for c in h]


def leaf_value_checks(leaf, poly):
    """Exact endpoint values: leaf bounds must straddle P at both endpoints."""
    n = len(poly) - 1
    i, m = leaf["i"], leaf["m"]
    left = evaluate_polynomial(poly, Fraction(i, 1 << m))
    right = evaluate_polynomial(poly, Fraction(i + 1, 1 << m))
    return (leaf["lo"] <= left <= leaf["hi"] and leaf["lo"] <= right <= leaf["hi"],
            left, right)


def bernstein_range(poly):
    """Exact hull of a polynomial over the whole interval [0,1].

    The Bernstein coefficients on [0,1] bracket every value of the polynomial,
    so min/max of those coefficients is a certified range for all real p in
    [0,1].  Used, for example, to bound |P''| globally.
    """
    n = len(poly) - 1
    if n < 0:
        raise ValueError("empty polynomial")
    den = _common_denominator(poly)
    ints = [c.numerator * (den // c.denominator) for c in poly]
    binom_row = [comb(n, k) for k in range(n + 1)]
    lcm_basis = lcm(*binom_row) if n else 1
    basis = [[comb(j, k) * (lcm_basis // binom_row[k]) for k in range(j + 1)]
             for j in range(n + 1)]
    _, b = _bernstein_bounds(ints, n, 0, 0, basis)
    divisor = den * lcm_basis
    return Fraction(min(b), divisor), Fraction(max(b), divisor)


def sup_bounds(leaves):
    lower = max(leaf["lo"] for leaf in leaves)
    upper = max(leaf["hi"] for leaf in leaves)
    return lower, upper


def affine_sup_bounds(poly, pieces=1024):
    """Second, independent certified interval for sup_p P(p).

    On each piece of a uniform grid with h = 1/pieces the chord between the
    endpoint values stays within M2*h^2/8 of the curve, where M2 is a certified
    global bound on |P''| (Bernstein range on [0,1], not a grid sample).
    Hence max(chords) + M2*h^2/8 is an upper bound and max(values) is a lower
    bound for the supremum over the whole interval.
    """
    n = len(poly) - 1
    second = [Fraction(k * (k - 1)) * poly[k] for k in range(2, n + 1)]
    if second:
        second_lo, second_hi = bernstein_range(second)
        m2 = max(abs(second_lo), abs(second_hi))
    else:
        m2 = Fraction(0)
    h = Fraction(1, pieces)
    values = [evaluate_polynomial(poly, Fraction(k, pieces))
              for k in range(pieces + 1)]
    error = m2 * h * h / 8
    lower = max(values)
    upper = max(max(values[k], values[k + 1]) for k in range(pieces)) + error
    return {"lower": lower, "upper": upper, "m2": m2, "pieces": pieces,
            "error": error}


def max_leaf_width(leaves):
    return max(leaf["hi"] - leaf["lo"] for leaf in leaves)


def cover_digest(leaves):
    digest = sha256()
    for leaf in sorted(leaves, key=lambda x: (x["m"], x["i"])):
        digest.update(("%d/%d:%s:%s\n" % (
            leaf["i"], leaf["m"], leaf["lo"], leaf["hi"])).encode())
    return digest.hexdigest()


def cover_tree(leaves):
    """Compact exact encoding of the cover: per depth, contiguous index runs.

    Leaves never overlap, so at each depth the generated indices form disjoint
    consecutive runs; storing (depth, start, end) is enough to reconstruct the
    dyadic partition exactly.  Bounds are regenerated from the polynomial.
    """
    runs = {}
    for leaf in sorted(leaves, key=lambda x: (x["m"], x["i"])):
        entry = runs.setdefault(leaf["m"], [])
        if entry and entry[-1][1] + 1 == leaf["i"]:
            entry[-1][1] = leaf["i"]
        else:
            entry.append([leaf["i"], leaf["i"]])
    return {str(depth): [list(run) for run in entry]
            for depth, entry in sorted(runs.items())}


def check_tree(tree, leaves):
    """Rebuild the dyadic partition from the compact tree and compare."""
    rebuilt = []
    for depth_text, run_list in tree.items():
        depth = int(depth_text)
        for start, end in run_list:
            for i in range(start, end + 1):
                rebuilt.append((i, depth))
    original = sorted((leaf["i"], leaf["m"]) for leaf in leaves)
    return sorted(rebuilt) == original


def envelope_value(leaves, p):
    """Lower/upper hull of F at an exact rational p in [0,1]."""
    if type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("p must be a Fraction in [0,1]")
    lefts = [leaf_interval(leaf)[0] for leaf in leaves]  # leaves generated in order
    idx = bisect.bisect_right(lefts, p) - 1
    idx = max(0, min(idx, len(leaves) - 1))
    leaf = leaves[idx]
    return leaf["lo"], leaf["hi"]


def fraction_text(value):
    return str(value)


# --------------------------------------------------------------------------- #
# judge for a production trace (duck-typed fields, no production imports)
# --------------------------------------------------------------------------- #
def diagnose(trace, final_evidence=None, final_stopped=None):
    """Check a produced step trace against the independent rules.

    Returns a list of violations; each names the *business rule* that the
    numbers break (not a mutation label), with expected and actual values.
    """
    violations = []
    counts = (0, 0, 0, 0)
    previous_evidence = Fraction(1)
    last = len(trace)
    for position, step in enumerate(trace, 1):
        required = ("index", "before", "probability", "arm", "outcome",
                    "after", "evidence", "stopped")
        if any(not hasattr(step, name) for name in required):
            violations.append({"rule": "trace_schema", "index": position,
                               "detail": "step lacks required fields"})
            return violations
        if step.index != position:
            violations.append({"rule": "trace_index", "index": position,
                               "expected": position, "actual": step.index})
        if tuple(step.before) != counts:
            violations.append({"rule": "count_before_matches_prior_history",
                               "index": position, "expected": counts,
                               "actual": tuple(step.before)})
        expected_q = allocation_probability(tuple(step.before))
        if step.probability != expected_q:
            violations.append({"rule": "allocation_probability_from_prior_counts",
                               "index": position, "expected": expected_q,
                               "actual": step.probability})
        try:
            validate_observation(step.arm, step.outcome)
            factor = evidence_factor(step.probability, step.arm, step.outcome)
        except ValueError as error:
            violations.append({"rule": "observation_validity", "index": position,
                               "detail": str(error)})
            return violations
        expected_evidence = previous_evidence * factor
        if step.evidence != expected_evidence:
            stale = None
            if step.probability != Fraction(1, 2):
                stale_factor = evidence_factor(Fraction(1, 2), step.arm, step.outcome)
                if step.evidence == previous_evidence * stale_factor:
                    stale = Fraction(1, 2)
            violations.append({
                "rule": "evidence_update_uses_current_allocation_probability",
                "index": position, "probability": step.probability,
                "expected": expected_evidence, "actual": step.evidence,
                "stale_probability_explains": stale,
                "arm": step.arm, "outcome": step.outcome})
        expected_after = _apply(tuple(step.before), step.arm, step.outcome)
        if tuple(step.after) != expected_after:
            violations.append({"rule": "count_update", "index": position,
                               "expected": expected_after,
                               "actual": tuple(step.after)})
        crossing = step.evidence >= THRESHOLD
        if crossing and position < last:
            violations.append({"rule": "first_crossing_must_stop",
                               "index": position, "expected": "stop",
                               "actual": "continued", "evidence": step.evidence})
        if crossing and position == last and not step.stopped:
            violations.append({"rule": "first_crossing_must_stop",
                               "index": position, "expected": "stop",
                               "actual": "not stopped", "evidence": step.evidence})
        if step.stopped and not crossing:
            violations.append({"rule": "stop_requires_threshold",
                               "index": position, "expected": THRESHOLD,
                               "actual": step.evidence})
        if step.stopped and position < last:
            violations.append({"rule": "no_consumption_after_stop",
                               "index": position,
                               "detail": "trace has steps after a stop"})
        counts = tuple(step.after)
        previous_evidence = step.evidence
    if final_evidence is not None and trace:
        if final_evidence != trace[-1].evidence:
            violations.append({"rule": "final_evidence_matches_trace",
                               "expected": trace[-1].evidence,
                               "actual": final_evidence})
    if final_stopped is not None and trace:
        expected_stopped = trace[-1].evidence >= THRESHOLD
        if bool(final_stopped) != expected_stopped:
            violations.append({"rule": "final_stop_matches_first_crossing",
                               "expected": expected_stopped,
                               "actual": bool(final_stopped)})
    return violations


def compare_run(produced, reference):
    """Field-by-field comparison of a production Result against reference_run.

    Per-step differences are reported before any length difference, so the
    first entry is always the earliest business divergence.
    """
    differences = []
    produced_trace = list(produced.trace)
    reference_trace = reference["trace"]
    for index, (left, right) in enumerate(zip(produced_trace, reference_trace), 1):
        pairs = (("index", right["index"], left.index),
                 ("before", right["before"], tuple(left.before)),
                 ("probability", right["probability"], left.probability),
                 ("arm", right["arm"], left.arm),
                 ("outcome", right["outcome"], left.outcome),
                 ("after", right["after"], tuple(left.after)),
                 ("evidence", right["evidence"], left.evidence),
                 ("stopped", right["stopped"], left.stopped))
        for field, expected, actual in pairs:
            if expected != actual:
                differences.append({"field": "step.%d.%s" % (index, field),
                                    "expected": expected, "actual": actual})
    if len(produced_trace) != len(reference_trace):
        differences.append({"field": "trace_length",
                            "expected": len(reference_trace),
                            "actual": len(produced_trace)})
    for field, expected, actual in (
            ("counts", reference["counts"], tuple(produced.counts)),
            ("evidence", reference["evidence"], produced.evidence),
            ("stopped", reference["stopped"], produced.stopped)):
        if expected != actual:
            differences.append({"field": field, "expected": expected,
                                "actual": actual})
    return differences
