"""
How one year of change is allocated to cells.

Every demand is placed on cells that can make that transition. The cells are chosen in one of two ways,
selected with the randomness argument:

  "gumbel", "jitter" or "none": ranking. Every cell gets a score, which is its transition potential plus
      some noise, and each demand goes to the cells with the highest scores. tau sets the size of the noise.
  "acceptance": the original rule of the model, described below. It uses no noise, so tau is not used.

The demand can be split between the inside and the outside of the centralities, and cells in a water zone
never change.
"""
from typing import Literal

import numpy as np

from . import core_functions

RANDOMNESS = ("gumbel", "jitter", "none", "acceptance")


def update_built_up_map(
    built_up_map,
    prob_maps_dict,
    annual_rates,
    rng=None,
    randomness: Literal["gumbel", "jitter", "none", "acceptance"] = "gumbel",
    tau=0.1,
    water_zone=None,
    central_zones=None,
    annual_rates_outside=None
):
    """
    Simulate one year of change and return the updated map.

    Args:
        built_up_map: 2D float array of density classes, NaN outside the study area
        prob_maps_dict: Dict[(from, to)] -> 2D suitability array, normalised to [0, 1]
        annual_rates: Dict[(from, to)] -> demand in cells for this year
        rng: numpy random generator; created with seed 42 if not given
        randomness: how the cells are chosen
          - "gumbel" (default): each demand goes to the cells with the highest score, where
            score = pot + tau * Gumbel(0,1)
          - "jitter": the same, with score = pot + Normal(0, tau * max(pot))
          - "none": the same, with score = pot, so no noise
          - "acceptance": the original rule of the model, see the end of this file. It uses
            the potential without noise, and each demand is rounded down instead of rounded
        tau: noise scale for "gumbel" and "jitter"; try 0.05–0.2 for your potential range.
            It is not used by "none" and "acceptance"
        water_zone: optional 2D array; cells equal to 1 are never changed
        central_zones: optional 2D array, 1 inside the centralities and 0 outside. If given,
            annual_rates is the demand inside the centralities and annual_rates_outside the demand outside
        annual_rates_outside: Dict[(from, to)] -> demand in cells outside the centralities;
            needed when central_zones is given

    Returns:
        A copy of built_up_map with the allocated cells changed to their new class
    """

    if rng is None:
        rng = np.random.default_rng(42)

    if randomness not in RANDOMNESS:
        raise ValueError(f"randomness must be one of {RANDOMNESS}, not {randomness!r}")

    for name, zone in (("water_zone", water_zone), ("central_zones", central_zones)):
        if zone is not None and zone.shape != built_up_map.shape:
            raise ValueError(f"{name} has shape {zone.shape} but the built-up map has shape {built_up_map.shape}")

    new_map = built_up_map.copy()
    # 1) base potentials
    potentials = core_functions.calc_transition_potentials(new_map, prob_maps_dict)

    # 2) where each demand is allocated: everywhere, or inside and outside the centralities
    allowed = np.ones(new_map.shape, dtype=bool) if water_zone is None else (water_zone != 1)
    if central_zones is None:
        zones = [(annual_rates, allowed, "")]
    else:
        if annual_rates_outside is None:
            raise ValueError("annual_rates_outside is needed when central_zones is given")
        zones = [(annual_rates, allowed & (central_zones == 1), " inside the centralities"),
                 (annual_rates_outside, allowed & (central_zones == 0), " outside the centralities")]

    if randomness == "acceptance":
        # 3) the original rule: the cells are accepted one by one, using the potential without noise
        for rates, mask, where in zones:
            _allocate_by_acceptance(new_map, potentials, rates, mask, rng, where)
        return new_map

    # 3) add noise to create "scores" used for ranking
    noisy = {}
    for key, pot in potentials.items():
        if randomness == "gumbel":
            g = rng.gumbel(0.0, 1.0, size=pot.shape)
            noisy[key] = pot + tau * g
        elif randomness == "jitter":
            m = np.nanmax(pot)
            sigma = (tau * m) if np.isfinite(m) and m > 0 else 0.0
            noisy[key] = pot + rng.normal(0.0, sigma, size=pot.shape)
        else:
            noisy[key] = pot  # "none": deterministic

    # 4) eligible donors sorted by descending *noisy* score
    elig_per_zone = [
        core_functions.calc_eligible_indices_generic(new_map, noisy, rates, sort=True, zone_mask=mask)
        for rates, mask, _ in zones
    ]

    # 5) allocate
    H, W = new_map.shape
    assigned = np.zeros((H, W), dtype=bool)

    for (rates, _, where), elig in zip(zones, elig_per_zone):
        items = [((int(i), int(j)), max(0, int(round(v)))) for (i, j), v in rates.items() if v > 0]
        items.sort(key=lambda kv: kv[1], reverse=True)

        for (i, j), need in items:
            cand = elig.get((i, j))
            if cand is not None:
                cand = cand[~assigned.ravel()[cand]]
            take = min(need, 0 if cand is None else cand.size)
            if take < need:
                print(f"    Only {take} of {need} cells could be allocated for {i} -> {j}{where}")
            if take <= 0:
                continue
            chosen = cand[:take]  # already ranked by noisy score
            new_map.flat[chosen] = float(j)
            assigned.flat[chosen] = True

    return new_map


# ---------------------------------------------------------------------------------------------------------
# The original acceptance rule of the model (randomness="acceptance")
#
# The cells that can change are visited in order of their transition potential, highest first.
# A cell is accepted for a transition if its potential is above a random number times a temperature.
# The temperature cools by a fixed amount after every pass over the cells, so the rule gets easier
# to satisfy, until the demand of every transition is met.
#
# This follows update_built_up_map of the original scripts. Two things differ:
#   - the random numbers come from the generator that is passed in, so a run can be repeated with a seed
#   - a pass is only repeated when it can still change something, so a demand that cannot be met
#     is reported instead of looping forever
# ---------------------------------------------------------------------------------------------------------

INITIAL_TEMPERATURE = 1.0
COOLING_RATE = 0.05
MIN_TEMPERATURE = 0.0
MAX_PASSES = 100

EXPANSION = ((0, 1), (0, 2), (0, 3))
DENSIFICATION = ((1, 2), (1, 3), (2, 3))


def _flat_potential(potentials, key, shape):
    pot = potentials.get(key)
    return np.zeros(shape[0] * shape[1], dtype=np.float32) if pot is None else np.asarray(pot, dtype=np.float32).ravel()


def _quota(rates, key):
    # the number of cells to change is the demand rounded down
    return max(0, int(np.floor(rates.get(key, 0.0))))


def _report(counts, quotas, where):
    for (i, j), quota in quotas.items():
        if counts[(i, j)] < quota:
            print(f"    Only {counts[(i, j)]} of {quota} cells could be allocated for {i} -> {j}{where}")


def _expand(new_map, start_map, potentials, quotas, donors, rng, where):
    """Expansion: class 0 cells become class 1, 2 or 3."""
    shape = new_map.shape
    pots = [_flat_potential(potentials, key, shape) for key in EXPANSION]
    score = pots[0] + pots[1] + pots[2]

    # Cells that are class 0, inside the zone and not water. A cell with no potential for any
    # class can never be accepted, so it is left out.
    cand = np.flatnonzero(donors.ravel() & (start_map.ravel() == 0) & (score > 0))
    cand = cand[np.argsort(-score[cand], kind="stable")]  # highest potential first

    counts = {key: 0 for key in EXPANSION}
    alive = np.ones(cand.size, dtype=bool)  # cells that have not changed yet

    for pass_number in range(MAX_PASSES):
        if all(counts[k] >= quotas[k] for k in EXPANSION):
            break
        temperature = max(MIN_TEMPERATURE, INITIAL_TEMPERATURE - pass_number * COOLING_RATE)

        pos = np.flatnonzero(alive)
        idx = cand[pos]
        r = rng.random((3, idx.size))
        accepted = np.stack([pots[n][idx] > r[n] * temperature for n in range(3)])

        progressed = False
        for p in np.flatnonzero(accepted.any(axis=0)):
            # a cell tries class 1 first, then class 2, then class 3
            for n, key in enumerate(EXPANSION):
                if counts[key] < quotas[key] and accepted[n, p]:
                    new_map.flat[idx[p]] = float(key[1])
                    counts[key] += 1
                    alive[pos[p]] = False
                    progressed = True
                    break
            if all(counts[k] >= quotas[k] for k in EXPANSION):
                break

        # at the lowest temperature every cell with a potential is accepted, so nothing is left to try
        if temperature == MIN_TEMPERATURE and not progressed:
            break

    _report(counts, quotas, where)


def _densify(new_map, start_map, potentials, quotas, donors, rng, where):
    """Densification: class 1 cells become class 2 or 3, and class 2 cells become class 3."""
    shape = new_map.shape
    p12, p13, p23 = [_flat_potential(potentials, key, shape) for key in DENSIFICATION]
    score = p12 + p13 + p23

    # Cells that are class 1 or 2, inside the zone and not water, as they were before the expansion
    start_flat = start_map.ravel()
    cand = np.flatnonzero(donors.ravel() & ((start_flat == 1) | (start_flat == 2)))
    cand = cand[np.argsort(-score[cand], kind="stable")]  # highest potential first
    start_class = start_flat[cand]

    counts = {key: 0 for key in DENSIFICATION}
    alive = np.ones(cand.size, dtype=bool)
    cooling_iters = (INITIAL_TEMPERATURE - MIN_TEMPERATURE) / COOLING_RATE

    for iteration in range(MAX_PASSES + 1):
        if all(counts[k] >= quotas[k] for k in DENSIFICATION):
            break
        temperature = max(MIN_TEMPERATURE, INITIAL_TEMPERATURE - iteration * COOLING_RATE)
        # the second temperature only starts to cool once the first has reached its minimum
        temperature_2 = max(MIN_TEMPERATURE, INITIAL_TEMPERATURE - max(0, iteration - cooling_iters) * COOLING_RATE)

        pos = np.flatnonzero(alive)
        idx = cand[pos]
        cls = start_class[pos]
        r = rng.random((3, idx.size))
        pot12, pot13, pot23 = p12[idx], p13[idx], p23[idx]

        # 1 -> 2 changes its rule in three phases
        if iteration <= cooling_iters:
            accept_12 = pot12 > r[0] * temperature
        elif iteration < 2 * cooling_iters:
            accept_12 = pot13 > r[0] * temperature_2
        elif iteration > 2 * cooling_iters:
            accept_12 = pot12 >= 0  # any cell
        else:
            accept_12 = np.zeros(idx.size, dtype=bool)

        # 1 -> 3 follows the cooling once and then accepts any cell
        if iteration <= cooling_iters:
            accept_13 = pot13 > r[1] * temperature
        else:
            accept_13 = pot13 >= 0

        accept_23 = pot23 > r[2] * temperature

        accept_12 &= cls == 1
        accept_13 &= cls == 1
        accept_23 &= cls == 2

        progressed = False
        for p in np.flatnonzero(accept_12 | accept_13 | accept_23):
            if cls[p] == 1:
                if counts[(1, 2)] < quotas[(1, 2)] and accept_12[p]:
                    new_map.flat[idx[p]] = 2.0
                    counts[(1, 2)] += 1
                elif counts[(1, 3)] < quotas[(1, 3)] and accept_13[p]:
                    new_map.flat[idx[p]] = 3.0
                    counts[(1, 3)] += 1
                else:
                    continue
            else:
                if counts[(2, 3)] < quotas[(2, 3)] and accept_23[p]:
                    new_map.flat[idx[p]] = 3.0
                    counts[(2, 3)] += 1
                else:
                    continue
            alive[pos[p]] = False
            progressed = True
            if all(counts[k] >= quotas[k] for k in DENSIFICATION):
                break

        # after the last phase every remaining cell is accepted, so nothing is left to try
        if iteration > 2 * cooling_iters and not progressed:
            break

    _report(counts, quotas, where)


def _allocate_by_acceptance(new_map, potentials, rates, zone_mask, rng, where=""):
    """
    Place the demand with the original acceptance rule, changing new_map in place.

    Args:
        new_map: 2D float array of density classes; the cells that change are written into it
        potentials: Dict[(from, to)] -> 2D transition potential, from core_functions.calc_transition_potentials
        rates: Dict[(from, to)] -> demand in cells; it is rounded down, as in the original
        zone_mask: 2D bool array; only cells where it is True can change
        rng: numpy random generator
        where: text added to the message when a demand cannot be met in full
    """
    unsupported = [k for k, v in rates.items() if v > 0 and tuple(k) not in EXPANSION + DENSIFICATION]
    if unsupported:
        raise ValueError(f"allocation='acceptance' only supports the transitions 0->1, 0->2, 0->3, 1->2, 1->3 and 2->3, not {unsupported}")

    quotas = {key: _quota(rates, key) for key in EXPANSION + DENSIFICATION}
    start_map = new_map.copy()  # both parts look at the cells as they were before this year's changes

    _expand(new_map, start_map, potentials, {k: quotas[k] for k in EXPANSION}, zone_mask, rng, where)
    _densify(new_map, start_map, potentials, {k: quotas[k] for k in DENSIFICATION}, zone_mask, rng, where)
