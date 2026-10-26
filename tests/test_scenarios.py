"""Fast tests on small synthetic maps. They need no example data."""
import contextlib
import io

import numpy as np
import pytest

from typing import get_args, get_type_hints

from SuSceMod.simulation import allocation, core_functions, demand, input_handling, scenarios

TRANSITIONS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


@pytest.fixture
def world():
    """An 80 x 80 map with a built-up block, random suitability maps, centralities on the left half and water on the right."""
    rng = np.random.default_rng(1)
    size = 80
    built_up = np.zeros((size, size), dtype=np.float32)
    built_up[10:70, 10:70] = 1
    built_up[30:50, 30:50] = 2
    probs = {k: rng.random((size, size)).astype(np.float32) for k in TRANSITIONS}
    zones = np.zeros((size, size), dtype=np.float32)
    zones[:, :40] = 1
    water = np.zeros((size, size), dtype=np.float32)
    water[:, 60:] = 1
    return built_up, probs, zones, water


def test_split_demand_adds_up_to_the_rounded_total():
    rng = np.random.default_rng(0)
    for _ in range(500):
        rates = {k: rng.uniform(0, 600) for k in [(0, 1), (0, 2), (1, 2), (2, 3)]}
        shares = [round(rng.uniform(0, 1), 2), round(rng.uniform(0, 1), 2)]
        inside, outside = demand.split_demand_by_zone(rates, shares)
        for k, rate in rates.items():
            assert inside[k] + outside[k] == round(rate)
            assert inside[k] >= 0 and outside[k] >= 0


def test_split_demand_gives_the_rounding_remainder_to_the_inside():
    inside, outside = demand.split_demand_by_zone({(0, 1): 10.0, (1, 2): 10.0}, [0.75, 0.5])
    assert (inside[(0, 1)], outside[(0, 1)]) == (8.0, 2.0)
    assert (inside[(1, 2)], outside[(1, 2)]) == (5.0, 5.0)


def test_central_shares_are_measured_from_two_maps():
    start = np.zeros((40, 40), dtype=np.float32)
    start[0:10, 0:10] = 1
    start[0:3, 25:28] = 1
    end = start.copy()
    zones = np.zeros((40, 40), dtype=np.float32)
    zones[:, :20] = 1
    end[20, 5] = 1
    end[21, 6] = 1
    end[22, 30] = 2
    end[23, 35] = 3
    end[0, 0] = 2
    end[5, 5] = 3
    end[8, 8] = 2
    end[0, 25] = 2
    assert input_handling.calculate_central_shares(start, end, zones) == [0.5, 0.75]


def test_each_zone_gets_its_own_demand(world):
    built_up, probs, zones, water = world
    inside = {(0, 1): 30.0, (0, 2): 10.0, (1, 2): 25.0, (1, 3): 5.0, (2, 3): 4.0}
    outside = {(0, 1): 10.0, (0, 2): 6.0, (1, 2): 15.0, (1, 3): 3.0, (2, 3): 2.0}
    new = scenarios.update_built_up_map(built_up, probs, inside, rng=np.random.default_rng(2),
                                        central_zones=zones, annual_rates_outside=outside, water_zone=water)
    for (i, j), asked in inside.items():
        moved = (built_up == i) & (new == j)
        assert int((moved & (zones == 1)).sum()) == asked
        assert int((moved & (zones == 0)).sum()) == outside[(i, j)]


def test_water_is_never_changed(world):
    built_up, probs, zones, water = world
    rates = {(0, 1): 40.0, (0, 2): 20.0, (1, 2): 30.0, (1, 3): 10.0, (2, 3): 3.0}
    new = scenarios.update_built_up_map(built_up, probs, rates, rng=np.random.default_rng(1), water_zone=water)
    assert int(((new != built_up) & (water == 1)).sum()) == 0
    assert int((new != built_up).sum()) > 0


def test_water_zone_with_nan_allows_every_cell(world):
    built_up, probs, zones, water = world
    rates = {(0, 1): 40.0, (1, 2): 30.0}
    free = scenarios.update_built_up_map(built_up, probs, rates, rng=np.random.default_rng(1))
    nan_water = np.full(built_up.shape, np.nan, dtype=np.float32)
    masked = scenarios.update_built_up_map(built_up, probs, rates, rng=np.random.default_rng(1), water_zone=nan_water)
    assert np.array_equal(free, masked, equal_nan=True)


def test_outside_demand_is_needed_with_centralities(world):
    built_up, probs, zones, _ = world
    with pytest.raises(ValueError, match="annual_rates_outside"):
        scenarios.update_built_up_map(built_up, probs, {(0, 1): 5.0}, central_zones=zones)


def test_zone_rasters_must_match_the_map_shape(world):
    built_up, probs, zones, water = world
    with pytest.raises(ValueError, match="water_zone"):
        scenarios.update_built_up_map(built_up, probs, {(0, 1): 5.0}, water_zone=water[:-1])
    with pytest.raises(ValueError, match="central_zones"):
        scenarios.update_built_up_map(built_up, probs, {(0, 1): 5.0}, central_zones=zones[:, :-1], annual_rates_outside={})


def test_centralities_raster_needs_both_values(world):
    built_up, probs, zones, _ = world
    with pytest.raises(ValueError, match="central_zones"):
        scenarios.simulate_centralities_scenario(2020, 2022, built_up, probs, {(0, 1): 5.0}, np.ones_like(zones), [0.2, 0.2], "out")
    with pytest.raises(ValueError, match="central_zones"):
        scenarios.simulate_centralities_scenario(2020, 2022, built_up, probs, {(0, 1): 5.0}, np.zeros_like(zones), [0.2, 0.2], "out")


def test_demand_that_cannot_be_met_is_reported(world, capsys):
    built_up, probs, _, _ = world
    zones = np.zeros(built_up.shape, dtype=np.float32)
    zones[0:3, 0:3] = 1  # far from any built-up cell, so no eligible cell inside
    scenarios.update_built_up_map(built_up, probs, {(0, 1): 5.0}, rng=np.random.default_rng(0), randomness="none",
                                  central_zones=zones, annual_rates_outside={(0, 1): 2.0})
    assert "Only 0 of 5 cells could be allocated for 0 -> 1 inside the centralities" in capsys.readouterr().out


def test_centralities_scenario_keeps_the_demand_and_moves_the_share(world):
    built_up, probs, zones, water = world
    rates = {(0, 1): 60.0, (0, 2): 20.0, (1, 2): 40.0, (1, 3): 8.0, (2, 3): 6.0}
    sims, paths, counts = quiet(scenarios.simulate_centralities_scenario, 2020, 2030, built_up, probs, rates, zones,
                                [0.2, 0.3], "out", final_central_shares=(0.75, 0.75), water_zone=water, seed=5)
    assert sorted(sims) == list(range(2021, 2031))
    assert paths[2021].endswith("sim_2021.tif")

    previous = built_up
    expansion_share, densification_share = [], []
    for year in range(2021, 2031):
        current = sims[year]
        expanded = (previous == 0) & (current > 0)
        densified = (previous >= 1) & (current > previous)
        assert int(expanded.sum()) == 80 and int(densified.sum()) == 54  # the total demand never changes
        expansion_share.append(int((expanded & (zones == 1)).sum()) / 80)
        densification_share.append(int((densified & (zones == 1)).sum()) / 54)
        assert int(((current != built_up) & (water == 1)).sum()) == 0
        previous = current

    assert expansion_share[0] < expansion_share[-1] and abs(expansion_share[-1] - 0.75) < 0.06
    assert densification_share[0] < densification_share[-1] and abs(densification_share[-1] - 0.75) < 0.06
    assert all(np.diff(expansion_share) > -0.06)


def test_same_seed_gives_the_same_maps(world):
    built_up, probs, zones, water = world
    rates = {(0, 1): 60.0, (1, 2): 40.0}
    a = quiet(scenarios.simulate_centralities_scenario, 2020, 2024, built_up, probs, rates, zones, [0.2, 0.3], "out", water_zone=water, seed=5)[0]
    b = quiet(scenarios.simulate_centralities_scenario, 2020, 2024, built_up, probs, rates, zones, [0.2, 0.3], "out", water_zone=water, seed=5)[0]
    assert all(np.array_equal(a[y], b[y], equal_nan=True) for y in a)


# ---- the original acceptance rule (randomness="acceptance")

@pytest.fixture
def acceptance_world():
    """A map where every transition has cells to work with: class 2 on the west edge of the built-up block,
    class 3 on the east edge and below the class 2 strip."""
    rng = np.random.default_rng(4)
    size = 60
    built_up = np.zeros((size, size), dtype=np.float32)
    built_up[20:40, 20:40] = 1
    built_up[20:40, 20:24] = 2
    built_up[20:40, 36:40] = 3
    built_up[40:44, 20:24] = 3
    probs = {k: rng.random((size, size)).astype(np.float32) for k in TRANSITIONS}
    return built_up, probs


ACCEPTANCE_RATES = {(0, 1): 10.6, (0, 2): 6.2, (0, 3): 4.9, (1, 2): 12.0, (1, 3): 5.0, (2, 3): 3.0}


def test_acceptance_moves_the_demand_rounded_down(acceptance_world):
    built_up, probs = acceptance_world
    new = scenarios.update_built_up_map(built_up, probs, ACCEPTANCE_RATES, rng=np.random.default_rng(0), randomness="acceptance")
    for (i, j), rate in ACCEPTANCE_RATES.items():
        assert int(((built_up == i) & (new == j)).sum()) == int(np.floor(rate))


def test_acceptance_gives_the_same_result_for_the_same_seed(acceptance_world):
    built_up, probs = acceptance_world
    a = scenarios.update_built_up_map(built_up, probs, ACCEPTANCE_RATES, rng=np.random.default_rng(7), randomness="acceptance")
    b = scenarios.update_built_up_map(built_up, probs, ACCEPTANCE_RATES, rng=np.random.default_rng(7), randomness="acceptance")
    assert np.array_equal(a, b)


def test_acceptance_only_changes_cells_with_a_potential_for_expansion(acceptance_world):
    built_up, probs = acceptance_world
    new = scenarios.update_built_up_map(built_up, probs, {(0, 1): 20.0}, rng=np.random.default_rng(0), randomness="acceptance")
    potential = core_functions.calc_transition_potentials(built_up, probs)[(0, 1)]
    assert np.all(potential[(built_up == 0) & (new == 1)] > 0)


def test_acceptance_respects_zones_and_water(acceptance_world):
    built_up, probs = acceptance_world
    zones = np.zeros(built_up.shape, dtype=np.float32)
    zones[:, :30] = 1
    water = np.zeros(built_up.shape, dtype=np.float32)
    water[:, 44:] = 1
    inside = {(0, 1): 4.0, (1, 2): 5.0, (1, 3): 2.0}
    outside = {(0, 1): 6.0, (1, 2): 7.0, (1, 3): 3.0}
    new = scenarios.update_built_up_map(built_up, probs, inside, rng=np.random.default_rng(1), randomness="acceptance",
                                        central_zones=zones, annual_rates_outside=outside, water_zone=water)
    for (i, j), asked in inside.items():
        moved = (built_up == i) & (new == j)
        assert int((moved & (zones == 1)).sum()) == asked
        assert int((moved & (zones == 0)).sum()) == outside[(i, j)]
    assert int(((new != built_up) & (water == 1)).sum()) == 0


def test_acceptance_reports_demand_that_cannot_be_met(world, capsys):
    built_up, probs, _, _ = world  # this map has no class 3 cells, so nothing can become class 3
    scenarios.update_built_up_map(built_up, probs, {(0, 3): 5.0}, rng=np.random.default_rng(0), randomness="acceptance")
    assert "Only 0 of 5 cells could be allocated for 0 -> 3" in capsys.readouterr().out


def test_acceptance_only_supports_the_four_class_transitions(acceptance_world):
    built_up, probs = acceptance_world
    with pytest.raises(ValueError, match="only supports"):
        scenarios.update_built_up_map(built_up, probs, {(3, 0): 5.0}, randomness="acceptance")


def test_unknown_randomness_is_refused(acceptance_world):
    built_up, probs = acceptance_world
    with pytest.raises(ValueError, match="randomness"):
        scenarios.update_built_up_map(built_up, probs, {(0, 1): 5.0}, randomness="other")


def test_every_scenario_accepts_the_option(acceptance_world):
    built_up, probs = acceptance_world
    rates = {(0, 1): 10.6, (0, 2): 6.2, (1, 2): 12.0, (1, 3): 5.0}
    zones = np.zeros(built_up.shape, dtype=np.float32)
    zones[:, :30] = 1
    runs = [
        quiet(scenarios.simulate_growth_based_scenario, 2020, 2022, built_up, probs, rates, "out", randomness="acceptance"),
        quiet(scenarios.simulate_density_based_scenario, 2020, 2022, built_up, probs, rates, {(0, 1): -1.0}, "out", randomness="acceptance"),
        quiet(scenarios.simulate_densification_only_scenario, 2020, 2022, built_up, probs, rates, "out", randomness="acceptance"),
        quiet(scenarios.simulate_centralities_scenario, 2020, 2022, built_up, probs, rates, zones, [0.2, 0.2], "out", randomness="acceptance"),
    ]
    for maps, paths, counts in runs:
        assert sorted(maps) == [2021, 2022]
    # the first year of the growth scenario moves exactly the demand rounded down
    first = runs[0][0][2021]
    for (i, j), rate in rates.items():
        assert int(((built_up == i) & (first == j)).sum()) == int(np.floor(rate))


# ---- demand rules and the central options of every scenario

def test_constant_demand_is_the_same_every_year():
    schedule = demand.constant_demand({(0, 1): 10.5, (1, 2): 4.0}, 3)
    assert len(schedule) == 3
    assert all(rates == {(0, 1): 10.5, (1, 2): 4.0} and freed == {} for rates, freed in schedule)


def test_trend_demand_changes_every_year_and_never_goes_below_zero():
    schedule = demand.trend_demand({(0, 1): 10.0, (1, 2): 5.0}, {(0, 1): -4.0, (1, 2): 1.5}, 4)
    assert [rates[(0, 1)] for rates, _ in schedule] == [10.0, 6.0, 2.0, 0.0]
    assert [rates[(1, 2)] for rates, _ in schedule] == [5.0, 6.5, 8.0, 9.5]


def test_densification_only_demand_phases_expansion_out_and_adds_it_to_densification():
    schedule = demand.densification_only_demand({(0, 1): 100.0, (0, 2): 40.0, (1, 2): 50.0, (1, 3): 10.0}, 10)
    first, freed_first = schedule[0]
    last, _ = schedule[-1]
    assert first[(0, 1)] == 90.0 and first[(0, 2)] == 36.0
    assert first[(1, 2)] == 60.0 and first[(1, 3)] == 14.0   # 1->2 gets the 10 freed from 0->1, 1->3 gets the 4 freed from 0->2
    assert freed_first == {(1, 2): 10.0, (1, 3): 4.0}
    assert last[(0, 1)] == 0.0 and last[(0, 2)] == 0.0


def test_central_shares_move_in_equal_yearly_steps():
    shares = demand.central_shares_by_year(2030, 2050, [0.2, 0.4], [0.6, 0.8])
    assert shares[2030] == [0.2, 0.4] and shares[2050] == [0.6, 0.8]
    assert shares[2040] == [0.4, 0.6] and shares[2031] == [0.22, 0.42]
    assert sorted(shares) == list(range(2030, 2051))
    assert demand.constant_central_shares(2030, 2032, [0.2, 0.4]) == {2030: [0.2, 0.4], 2031: [0.2, 0.4], 2032: [0.2, 0.4]}


def test_freed_demand_goes_inside_by_the_expansion_share():
    rates = {(0, 1): 100.0, (1, 2): 60.0}
    plain_inside, _ = demand.split_demand_by_zone(rates, [0.2, 0.5])
    inside, outside = demand.split_demand_by_zone(rates, [0.2, 0.5], freed={(1, 2): 10.0})
    assert plain_inside[(1, 2)] == 30.0                       # the densification share applies to everything
    assert inside[(1, 2)] == 27.0 and outside[(1, 2)] == 33.0  # 0.5 * 50 + 0.2 * 10
    assert inside[(0, 1)] == 20.0


@pytest.mark.parametrize("scenario, extra", [
    (scenarios.simulate_growth_based_scenario, ()),
    (scenarios.simulate_density_based_scenario, ({(0, 1): -2.0},)),
    (scenarios.simulate_densification_only_scenario, ()),
])
def test_central_options_need_zones_and_shares_together(world, scenario, extra):
    built_up, probs, zones, _ = world
    rates = {(0, 1): 10.0, (1, 2): 5.0}
    with pytest.raises(ValueError, match="together"):
        quiet(scenario, 2020, 2022, built_up, probs, rates, *extra, "out", central_zones=zones)
    with pytest.raises(ValueError, match="together"):
        quiet(scenario, 2020, 2022, built_up, probs, rates, *extra, "out", central_shares=[0.3, 0.3])


def test_a_forced_share_stays_the_same_every_year(world):
    built_up, probs, zones, water = world
    rates = {(0, 1): 60.0, (0, 2): 20.0, (1, 2): 40.0, (1, 3): 8.0, (2, 3): 6.0}
    sims, _, _ = quiet(scenarios.simulate_growth_based_scenario, 2020, 2025, built_up, probs, rates, "out",
                       water_zone=water, central_zones=zones, central_shares=[0.3, 0.5], seed=3)
    previous = built_up
    for year in range(2021, 2026):
        current = sims[year]
        expanded = (previous == 0) & (current > 0)
        densified = (previous >= 1) & (current > previous)
        assert int(expanded.sum()) == 80 and int((expanded & (zones == 1)).sum()) == 24   # 30% of 80
        assert int(densified.sum()) == 54 and int((densified & (zones == 1)).sum()) == 27  # 50% of 54
        previous = current


def test_business_as_usual_can_force_a_share_too(world):
    built_up, probs, zones, _ = world
    rates = {(0, 1): 60.0, (1, 2): 40.0}
    sims, _, _ = quiet(scenarios.simulate_density_based_scenario, 2020, 2023, built_up, probs, rates, {(0, 1): -10.0, (1, 2): -5.0}, "out",
                       central_zones=zones, central_shares=[0.5, 0.5], seed=3)
    first = sims[2021]
    assert int(((built_up == 0) & (first == 1)).sum()) == 60
    assert int(((built_up == 0) & (first == 1) & (zones == 1)).sum()) == 30
    second = sims[2022]
    assert int(((first == 0) & (second == 1)).sum()) == 50   # the demand fell by 10


def test_densification_only_with_centralities_follows_the_original_rule(world):
    built_up, probs, zones, _ = world
    rates = {(0, 1): 60.0, (0, 2): 20.0, (1, 2): 40.0, (1, 3): 8.0, (2, 3): 6.0}
    shares = [0.2, 0.5]
    sims, _, _ = quiet(scenarios.simulate_densification_only_scenario, 2020, 2030, built_up, probs, rates, "out",
                       central_zones=zones, central_shares=shares, seed=3)
    schedule = demand.densification_only_demand(rates, 10)
    previous = built_up
    for year, (year_rates, freed) in zip(range(2021, 2031), schedule):
        inside, outside = demand.split_demand_by_zone(year_rates, shares, freed)
        current = sims[year]
        for (i, j), cells in inside.items():
            moved = (previous == i) & (current == j)
            assert int((moved & (zones == 1)).sum()) == cells
            assert int((moved & (zones == 0)).sum()) == outside[(i, j)]
        previous = current
    # expansion is gone in the last year
    assert int(((sims[2029] == 0) & (sims[2030] > 0)).sum()) == 0


@pytest.mark.parametrize("randomness", allocation.RANDOMNESS)
def test_every_randomness_option_runs(acceptance_world, randomness):
    built_up, probs = acceptance_world
    new = scenarios.update_built_up_map(built_up, probs, {(0, 1): 10.0, (1, 2): 8.0}, rng=np.random.default_rng(0), randomness=randomness)
    assert int((new != built_up).sum()) == 18


def test_the_randomness_type_hints_match_the_options_that_are_accepted():
    assert allocation.RANDOMNESS == ("gumbel", "jitter", "none", "acceptance")
    for function in (allocation.update_built_up_map, scenarios.simulate_growth_based_scenario, scenarios.simulate_density_based_scenario,
                     scenarios.simulate_densification_only_scenario, scenarios.simulate_centralities_scenario):
        assert get_args(get_type_hints(function)["randomness"]) == allocation.RANDOMNESS, function.__name__
