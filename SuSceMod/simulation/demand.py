"""
How much demand there is for each transition in each year, and how much of it goes inside the centralities.

A demand rule returns one entry for every simulated year: (rates, freed).
  rates: Dict[(from, to)] -> demand in cells for that year
  freed: Dict[(from, to)] -> the part of a densification demand that comes from expansion that was phased out.
         It is only filled by densification_only_demand and is empty for the other rules.
"""
import numpy as np


def constant_demand(annual_rates, n_years):
    """The same demand every year (Growth-As-Usual)."""
    return [(dict(annual_rates), {}) for _ in range(n_years)]


def trend_demand(annual_rates, differential_change_rates, n_years):
    """
    The demand changes every year by a fixed amount per transition (Business-As-Usual), and never goes below zero.

    Args:
        annual_rates: Dict[(from, to)] -> demand in the first year
        differential_change_rates: Dict[(from, to)] -> change in demand per year, can be negative
        n_years: number of years to simulate
    """
    delta = {k: float(v) for k, v in (differential_change_rates or {}).items()}
    current = {k: float(v) for k, v in annual_rates.items()}

    schedule = []
    for _ in range(n_years):
        schedule.append((dict(current), {}))
        # advance the rates by one year, clamped to >= 0
        following = {}
        for k in set(current) | set(delta):
            value = round(current.get(k, 0.0) + delta.get(k, 0.0), 2)
            following[k] = 0.0 if value < 0 else value
        current = following
    return schedule


def densification_only_demand(annual_rates, n_years):
    """
    The demand for expansion (from class 0) falls linearly to zero in the last year. The cells this frees up
    are added to the densification demand, paired in sorted order: 0->1 with 1->2, 0->2 with 1->3 and 0->3 with 2->3.
    """
    exp_keys = sorted(k for k in annual_rates if k[0] == 0)
    dens_keys = sorted(k for k in annual_rates if k[0] != 0)
    decrease_rates = {k: round(annual_rates[k] / n_years, 2) for k in exp_keys}
    dens_pairs = dict(zip(dens_keys, exp_keys))

    schedule = []
    for year in range(n_years):
        rates, freed = {}, {}
        # expansion decreases at a constant rate every year, clamped to >= 0
        for k in exp_keys:
            rates[k] = max(0.0, round(annual_rates[k] - (year + 1) * decrease_rates[k], 2))
        # the decrease in expansion is compensated by an increase in densification
        for k in dens_keys:
            extra = decrease_rates[dens_pairs[k]] if k in dens_pairs else 0.0
            rates[k] = round(annual_rates[k] + extra, 2)
            freed[k] = extra
        schedule.append((rates, freed))
    return schedule


def constant_central_shares(initial_year, final_year, shares):
    """The same share of the demand inside the centralities every year, as {year: [expansion, densification]}."""
    return {year: list(shares) for year in range(initial_year, final_year + 1)}


def central_shares_by_year(initial_year, final_year, initial_shares, final_shares):
    """
    The share of the demand inside the centralities for every year, moving in equal yearly steps from
    initial_shares to final_shares, as {year: [expansion, densification]}.
    """
    steps = [(final_shares[i] - initial_shares[i]) / (final_year - initial_year) for i in range(len(initial_shares))]
    shares = {}
    for i in range(final_year - initial_year + 1):
        shares[initial_year + i] = [
            round(initial_shares[0] + steps[0] * i, 2),
            round(initial_shares[1] + steps[1] * i, 2),
        ]
    return shares


def split_demand_by_zone(annual_rates, shares, freed=None):
    """
    Split the demand of every transition into a part inside and a part outside the centralities.

    Args:
        annual_rates: Dict[(from, to)] -> demand in cells for the year
        shares: [share of expansion inside, share of densification inside]; expansion is any
                transition from class 0, densification is any other transition
        freed: optional Dict[(from, to)] -> the part of a densification demand that comes from expansion that
               was phased out. That part goes inside by the expansion share, so each zone's own cancelled
               expansion turns into its own densification.

    Returns:
        inside, outside: two Dict[(from, to)] -> whole cells. For every transition they add up
        to the rounded demand; if rounding loses a cell, it goes to the inside part.
    """
    freed = freed or {}
    inside, outside = {}, {}
    for (i, j), rate in annual_rates.items():
        if int(i) == 0:
            share = shares[0]
        else:
            freed_part = freed.get((i, j), 0.0)
            if freed_part > 0 and rate > 0:
                share = (shares[1] * (rate - freed_part) + shares[0] * freed_part) / rate
            else:
                share = shares[1]
        total = max(0, int(round(rate)))
        out = int(np.floor(round(total * (1 - share), 2)))
        inside[(i, j)] = float(total - out)
        outside[(i, j)] = float(out)
    return inside, outside
