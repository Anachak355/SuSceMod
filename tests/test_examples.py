"""The three example scenarios must reproduce the results saved in SuSceMod/examples/output.

This catches any change that alters the simulated maps. It takes about a minute.
"""
import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from SuSceMod.simulation import data_utilities, input_handling, scenarios

EXAMPLES = Path(__file__).resolve().parent.parent / "SuSceMod" / "examples"


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


@pytest.fixture(scope="module")
def inputs():
    old_dir = os.getcwd()
    os.chdir(EXAMPLES)
    try:
        initial = data_utilities.load_raster_data("input_rasters/BU_2020.tif")
        prob_maps, _ = input_handling.load_prob_maps("suitability_maps/10_20/")
        change_10_20, _ = input_handling.calculate_change_maps("input_rasters/BU_2010.tif", "input_rasters/BU_2020.tif")
        change_00_10, _ = input_handling.calculate_change_maps("input_rasters/BU_2000.tif", "input_rasters/BU_2010.tif")
        rates, _, _ = quiet(input_handling.calculate_change_rates, change_10_20, [2010, 2020])
        rates_h, differential, _, _ = quiet(input_handling.calculate_change_rates_historical,
                                            change_00_10, change_10_20, [2000, 2010, 2020])
        yield initial, prob_maps, rates, rates_h, differential
    finally:
        os.chdir(old_dir)


def saved(scenario, year):
    path = EXAMPLES / "output" / scenario / f"sim_{year}.tif"
    if not path.exists():
        pytest.skip(f"{path} is not there; run the example notebook first")
    return data_utilities.load_raster_data(str(path))


def assert_same_as_saved(scenario, maps):
    for year in range(2021, 2031):
        assert np.array_equal(maps[year], saved(scenario, year), equal_nan=True), f"{scenario} differs in {year}"


def test_growth_based_matches_saved_output(inputs):
    initial, prob_maps, rates, _, _ = inputs
    maps = quiet(scenarios.simulate_growth_based_scenario, 2020, 2030, initial, prob_maps, rates, "out",
                 randomness="gumbel", seed=2118)[0]
    assert_same_as_saved("growth_based", maps)


def test_density_based_matches_saved_output(inputs):
    initial, prob_maps, _, rates_h, differential = inputs
    maps = quiet(scenarios.simulate_density_based_scenario, 2020, 2030, initial, prob_maps, rates_h, differential, "out",
                 randomness="gumbel", seed=2118)[0]
    assert_same_as_saved("density_based", maps)


def test_densification_only_matches_saved_output(inputs):
    initial, prob_maps, rates, _, _ = inputs
    maps = quiet(scenarios.simulate_densification_only_scenario, 2020, 2030, initial, prob_maps, rates, "out",
                 randomness="gumbel", seed=2118)[0]
    assert_same_as_saved("densification_only", maps)


def test_centralities_matches_saved_output(inputs):
    """The same steps as the centralities notebook: measure the historical share inside the centralities, run GAU to 2030
    with that share forced, then let the share move to 75% by 2050."""
    zones_path = EXAMPLES / "input_rasters" / "BU_centralities.tif"
    water_path = EXAMPLES / "input_rasters" / "ex_Zone_eau_binary.tif"
    if not (zones_path.exists() and water_path.exists()):
        pytest.skip("BU_centralities.tif and ex_Zone_eau_binary.tif are not in input_rasters")
    initial, prob_maps, rates, _, _ = inputs
    zones = data_utilities.load_raster_data(str(zones_path))
    water = data_utilities.load_raster_data(str(water_path))
    map_2010 = data_utilities.load_raster_data(str(EXAMPLES / "input_rasters" / "BU_2010.tif"))

    historical = input_handling.calculate_central_shares(map_2010, initial, zones)
    assert historical == [0.13, 0.26]  # the shares of chapter 7

    gau = quiet(scenarios.simulate_growth_based_scenario, 2020, 2030, initial, prob_maps, rates, "out",
                randomness="gumbel", seed=2118, water_zone=water, central_zones=zones, central_shares=historical)[0]
    shares = input_handling.calculate_central_shares(initial, gau[2030], zones)
    assert shares == [0.13, 0.26]  # forcing the share keeps it at 2030

    maps = quiet(scenarios.simulate_centralities_scenario, 2030, 2050, gau[2030], prob_maps, rates, zones, shares, "out",
                 final_central_shares=[0.75, 0.75], water_zone=water, randomness="gumbel", seed=2118)[0]
    for year in range(2031, 2051):
        assert np.array_equal(maps[year], saved("centralities", year), equal_nan=True), f"centralities differs in {year}"
