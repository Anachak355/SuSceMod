"""Fast tests on small synthetic maps. They need no example data."""
import contextlib
import io

import numpy as np
import pytest

from SuSceMod.simulation import core_functions, input_handling, scenarios

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
        inside, outside = core_functions.split_demand_by_zone(rates, shares)
        for k, rate in rates.items():
            assert inside[k] + outside[k] == round(rate)
            assert inside[k] >= 0 and outside[k] >= 0


def test_split_demand_gives_the_rounding_remainder_to_the_inside():
    inside, outside = core_functions.split_demand_by_zone({(0, 1): 10.0, (1, 2): 10.0}, [0.75, 0.5])
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
