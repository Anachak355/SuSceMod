"""
The scenarios.

Every scenario decides how much demand there is each year (see demand.py) and then runs the same yearly loop.
That loop can also send part of the demand inside the centralities: give central_zones and central_shares to
force the same share every year, which keeps the historical share as it is.

    Growth-As-Usual (GAU)  simulate_growth_based_scenario
    Business-As-Usual (BAU) simulate_density_based_scenario
    Densification Only (DO) simulate_densification_only_scenario
    Centralities            simulate_centralities_scenario (GAU demand, with a share that moves over the years)
"""
import os
from typing import Literal

import numpy as np

from . import demand as demand_rules
from .allocation import update_built_up_map


def _print_demands(annual_rates):
    print("\nDemands (cells/year):")
    for (i, j), rate in sorted(annual_rates.items()):
        print(f"    {i} -> {j}: {rate}")


def _forced_shares(central_zones, central_shares, initial_year, final_year):
    """The same share inside the centralities every year, or None when centralities are not used."""
    if (central_zones is None) != (central_shares is None):
        raise ValueError("central_zones and central_shares must be given together")
    if central_zones is None:
        return None
    return demand_rules.constant_central_shares(initial_year, final_year, central_shares)


def _run(
    initial_year,
    final_year,
    initial_built_up_map,
    prob_maps_dict,
    demand,
    output_folder,
    randomness: Literal['gumbel', 'jitter', 'none', 'acceptance'],
    seed,
    tau,
    water_zone,
    central_zones=None,
    central_shares=None
):
    """
    The yearly loop that every scenario uses.

    demand: one (rates, freed) entry for each year, from demand.py
    central_shares: {year: [expansion, densification]} share inside the centralities, needed with central_zones
    """
    if central_zones is not None and not (np.any(central_zones == 1) and np.any(central_zones == 0)):
        raise ValueError("central_zones must have cells equal to 1 (inside) and cells equal to 0 (outside)")

    simulated_maps, output_paths, class_counts = {}, {}, {}

    current_built_up_map = initial_built_up_map.copy()

    if central_zones is not None:
        print(f"\nShare inside the centralities (expansion, densification): "
              f"{list(central_shares[initial_year])} in {initial_year}, {list(central_shares[final_year])} in {final_year}")

    rng = np.random.default_rng(seed)

    for year in range(final_year - initial_year):
        # Remember previous simulation values to compare with new
        unique_old, counts_old = np.unique(current_built_up_map, return_counts=True)

        # Calculate the current year
        current_year = initial_year + year

        print("\n###################################################################\n")
        print(f"Current year: {current_year}")
        print(f"Simulating for year: {current_year + 1}")

        rates, freed = demand[year]
        if central_zones is None:
            rates_inside, rates_outside = rates, None
            print("\nCurrent demands (cells/year):")
            for (i, j), rate in sorted(rates.items()):
                print(f"    {i} -> {j}: {rate}")
        else:
            # Split the demand into the part inside and the part outside the centralities
            shares = central_shares[current_year + 1]
            rates_inside, rates_outside = demand_rules.split_demand_by_zone(rates, shares, freed)
            print(f"\nShare inside the centralities: expansion {shares[0]}, densification {shares[1]}")
            print("Demands inside / outside the centralities (cells/year):")
            for (i, j) in sorted(rates_inside):
                print(f"    {i} -> {j}: {rates_inside[(i, j)]} / {rates_outside[(i, j)]}")

        # Run simulation, save map, and overwrite previous year map
        updated_map = update_built_up_map(current_built_up_map, prob_maps_dict, rates_inside,
                                          rng=rng, randomness=randomness, tau=tau, water_zone=water_zone,
                                          central_zones=central_zones, annual_rates_outside=rates_outside)

        simulated_maps[current_year + 1] = updated_map
        current_built_up_map = updated_map

        # Set output paths
        output_paths[current_year + 1] = os.path.join(output_folder, f"sim_{current_year + 1}.tif")

        # Debugging: Print summary of the updated map
        unique, counts = np.unique(current_built_up_map, return_counts=True)
        unique = [float(x) if not np.isnan(x) else 'nan' for x in unique]
        counts = [int(x) for x in counts]
        print('\n------------------------------------------------------')
        print(f"\nYear {current_year + 1}: Updated built-up map summary (value: count):")
        print(dict(zip(unique, counts)))
        if year > 0:
            unique_old = [float(x) if not np.isnan(x) else 'nan' for x in unique_old]
            curr = {int(k): int(v) for k, v in zip(unique, counts) if k != 'nan'}
            prev = {int(k): int(v) for k, v in zip(unique_old, counts_old) if k != 'nan'}
            all_classes = sorted(set(curr) | set(prev))
            counts_diff = {c: curr.get(c, 0) - prev.get(c, 0) for c in all_classes}
            print("\nChanges from previous year summary (value: count):")
            print(counts_diff)

        # Saving values to a dictionary to save to file
        class_counts[current_year + 1] = [unique, counts]

    return simulated_maps, output_paths, class_counts


def simulate_growth_based_scenario(
    initial_year,
    final_year,
    initial_built_up_map,
    prob_maps_dict,
    annual_rates,
    output_folder,
    randomness: Literal['gumbel', 'jitter', 'none', 'acceptance'] = 'gumbel',
    seed=2118,
    tau=0.1,
    water_zone=None,
    central_zones=None,
    central_shares=None
):
    """
    Growth-As-Usual (GAU): the same annual demand for every transition, every year.

    Args:
        initial_year, final_year: simulates the years initial_year + 1 to final_year
        initial_built_up_map: 2D float array of density classes, NaN outside the study area
        prob_maps_dict: Dict[(from, to)] -> 2D suitability array, normalised to [0, 1]
        annual_rates: Dict[(from, to)] -> demand in cells per year
        output_folder: folder the simulated maps are meant to be saved in; used to build output_paths
        randomness: how the cells are chosen: "gumbel" (default), "jitter", "none" or "acceptance",
            the original rule of the model. See allocation.update_built_up_map
        seed: seed of the random generator, which is created once for the whole run
        tau: noise scale of "gumbel" and "jitter", see allocation.update_built_up_map
        water_zone: optional 2D array; cells equal to 1 are never changed
        central_zones: optional 2D array, 1 inside the centralities and 0 outside
        central_shares: [expansion, densification] share of each year's demand that is placed inside the
            centralities, the same every year. Needed together with central_zones.
            Without these two arguments nothing decides where the demand goes: the model places it where
            the transition potential is highest, which tends to be near existing built-up areas, so the
            share inside the centralities is whatever comes out. With them, that share is forced.
            The demand itself does not change, only where it goes. For example [0.13, 0.26] keeps the
            historical split, with 13% of the expansion and 26% of the densification inside the
            centralities. Only the share inside is given, the share outside is one minus it.
            For a share that moves over the years, use simulate_centralities_scenario

    Returns:
        simulated_maps: Dict[year] -> 2D array
        output_paths: Dict[year] -> path to save that year's map to
        class_counts: Dict[year] -> [classes, cell counts]
    """
    shares = _forced_shares(central_zones, central_shares, initial_year, final_year)
    _print_demands(annual_rates)
    schedule = demand_rules.constant_demand(annual_rates, final_year - initial_year)
    return _run(initial_year, final_year, initial_built_up_map, prob_maps_dict, schedule, output_folder,
                randomness, seed, tau, water_zone, central_zones, shares)


def simulate_density_based_scenario(
    initial_year,
    final_year,
    initial_built_up_map,
    prob_maps_dict,
    annual_rates,
    differential_change_rates,
    output_folder,
    randomness: Literal['gumbel', 'jitter', 'none', 'acceptance'] = 'gumbel',
    seed=2118,
    tau=0.1,
    water_zone=None,
    central_zones=None,
    central_shares=None
):
    """
    Business-As-Usual (BAU): the demand changes every year by a fixed amount, which follows the trend between
    two historical periods. Demand never goes below zero.

    Args:
        initial_year, final_year: simulates the years initial_year + 1 to final_year
        initial_built_up_map: 2D float array of density classes, NaN outside the study area
        prob_maps_dict: Dict[(from, to)] -> 2D suitability array, normalised to [0, 1]
        annual_rates: Dict[(from, to)] -> demand in cells per year in the first year
        differential_change_rates: Dict[(from, to)] -> change in demand per year, can be negative
        output_folder: folder the simulated maps are meant to be saved in; used to build output_paths
        randomness: how the cells are chosen: "gumbel" (default), "jitter", "none" or "acceptance",
            the original rule of the model. See allocation.update_built_up_map
        seed: seed of the random generator, which is created once for the whole run
        tau: noise scale of "gumbel" and "jitter", see allocation.update_built_up_map
        water_zone: optional 2D array; cells equal to 1 are never changed
        central_zones: optional 2D array, 1 inside the centralities and 0 outside
        central_shares: [expansion, densification] share of each year's demand that is placed inside the
            centralities, the same every year. Needed together with central_zones.
            Without these two arguments nothing decides where the demand goes: the model places it where
            the transition potential is highest, which tends to be near existing built-up areas, so the
            share inside the centralities is whatever comes out. With them, that share is forced.
            The demand itself does not change, only where it goes. For example [0.13, 0.26] keeps the
            historical split, with 13% of the expansion and 26% of the densification inside the
            centralities. Only the share inside is given, the share outside is one minus it.
            For a share that moves over the years, use simulate_centralities_scenario

    Returns:
        simulated_maps: Dict[year] -> 2D array
        output_paths: Dict[year] -> path to save that year's map to
        class_counts: Dict[year] -> [classes, cell counts]
    """
    shares = _forced_shares(central_zones, central_shares, initial_year, final_year)
    _print_demands(annual_rates)
    schedule = demand_rules.trend_demand(annual_rates, differential_change_rates, final_year - initial_year)
    return _run(initial_year, final_year, initial_built_up_map, prob_maps_dict, schedule, output_folder,
                randomness, seed, tau, water_zone, central_zones, shares)


def simulate_densification_only_scenario(
    initial_year,
    final_year,
    initial_built_up_map,
    prob_maps_dict,
    annual_rates,
    output_folder,
    randomness: Literal['gumbel', 'jitter', 'none', 'acceptance'] = 'gumbel',
    seed=2118,
    tau=0.1,
    water_zone=None,
    central_zones=None,
    central_shares=None
):
    """
    Densification Only (DO): the demand for expansion (from class 0) falls linearly to zero by final_year.
    The cells this frees up are added to the densification demand, paired in sorted order:
    0->1 with 1->2, 0->2 with 1->3 and 0->3 with 2->3.

    With centralities, each zone's own cancelled expansion turns into that zone's densification: the freed
    demand goes inside by the expansion share.

    Args:
        initial_year, final_year: simulates the years initial_year + 1 to final_year
        initial_built_up_map: 2D float array of density classes, NaN outside the study area
        prob_maps_dict: Dict[(from, to)] -> 2D suitability array, normalised to [0, 1]
        annual_rates: Dict[(from, to)] -> demand in cells per year
        output_folder: folder the simulated maps are meant to be saved in; used to build output_paths
        randomness: how the cells are chosen: "gumbel" (default), "jitter", "none" or "acceptance",
            the original rule of the model. See allocation.update_built_up_map
        seed: seed of the random generator, which is created once for the whole run
        tau: noise scale of "gumbel" and "jitter", see allocation.update_built_up_map
        water_zone: optional 2D array; cells equal to 1 are never changed
        central_zones: optional 2D array, 1 inside the centralities and 0 outside
        central_shares: [expansion, densification] share of each year's demand that is placed inside the
            centralities, the same every year. Needed together with central_zones.
            Without these two arguments nothing decides where the demand goes: the model places it where
            the transition potential is highest, which tends to be near existing built-up areas, so the
            share inside the centralities is whatever comes out. With them, that share is forced.
            The demand itself does not change, only where it goes. For example [0.13, 0.26] keeps the
            historical split, with 13% of the expansion and 26% of the densification inside the
            centralities. Only the share inside is given, the share outside is one minus it.
            For a share that moves over the years, use simulate_centralities_scenario

    Returns:
        simulated_maps: Dict[year] -> 2D array
        output_paths: Dict[year] -> path to save that year's map to
        class_counts: Dict[year] -> [classes, cell counts]
    """
    shares = _forced_shares(central_zones, central_shares, initial_year, final_year)
    _print_demands(annual_rates)
    schedule = demand_rules.densification_only_demand(annual_rates, final_year - initial_year)
    return _run(initial_year, final_year, initial_built_up_map, prob_maps_dict, schedule, output_folder,
                randomness, seed, tau, water_zone, central_zones, shares)


def simulate_centralities_scenario(
    initial_year,
    final_year,
    initial_built_up_map,
    prob_maps_dict,
    annual_rates,
    central_zones,
    initial_central_shares,
    output_folder,
    final_central_shares=(0.75, 0.75),
    water_zone=None,
    randomness: Literal['gumbel', 'jitter', 'none', 'acceptance'] = 'gumbel',
    seed=2118,
    tau=0.1
):
    """
    Centralities scenario: the total demand is the same every year, as in GAU, but a growing share of it is
    allocated inside the centralities. The share moves in equal yearly steps from its initial value
    to its final value.

    Args:
        initial_year, final_year: simulates the years initial_year + 1 to final_year
        initial_built_up_map: 2D float array of density classes, NaN outside the study area
        prob_maps_dict: Dict[(from, to)] -> 2D suitability array, normalised to [0, 1]
        annual_rates: Dict[(from, to)] -> total demand in cells per year, the same every year
        central_zones: 2D array, 1 inside the centralities and 0 outside
        initial_central_shares: [expansion, densification] share of the demand inside the centralities
            in initial_year, for example from input_handling.calculate_central_shares
        output_folder: folder the simulated maps are meant to be saved in; used to build output_paths
        final_central_shares: [expansion, densification] share inside the centralities in final_year.
            Only the share inside is given, the share outside is always one minus it
        water_zone: optional 2D array; cells equal to 1 are never changed
        randomness: how the cells are chosen: "gumbel" (default), "jitter", "none" or "acceptance",
            the original rule of the model. See allocation.update_built_up_map
        seed: seed of the random generator, which is created once for the whole run
        tau: noise scale of "gumbel" and "jitter", see allocation.update_built_up_map

    Expansion is any transition from class 0, and densification any other transition. Each year the
    demand of a transition is split into whole cells inside and outside the centralities, and the two
    parts always add up to the total.

    Returns:
        simulated_maps: Dict[year] -> 2D array
        output_paths: Dict[year] -> path to save that year's map to
        class_counts: Dict[year] -> [classes, cell counts]
    """
    shares = demand_rules.central_shares_by_year(initial_year, final_year, initial_central_shares, final_central_shares)
    _print_demands(annual_rates)
    schedule = demand_rules.constant_demand(annual_rates, final_year - initial_year)
    return _run(initial_year, final_year, initial_built_up_map, prob_maps_dict, schedule, output_folder,
                randomness, seed, tau, water_zone, central_zones, shares)
