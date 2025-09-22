import numpy as np
import os
from . import core_functions

def update_built_up_map(
    built_up_map, 
    prob_maps_dict, 
    annual_rates,
    rng=None, 
    randomness="gumbel", 
    tau=0.1
):
    """
    randomness: "none" | "gumbel" | "jitter"
      - "gumbel": scores = pot + tau * Gumbel(0,1)
      - "jitter": scores = pot + Normal(0, tau * max(pot))
    tau: noise scale; try 0.05–0.2 for your potential range.
    """

    if rng is None:
        rng = np.random.default_rng(42)

    new_map = built_up_map.copy()
    # 1) base potentials
    potentials = core_functions.calc_transition_potentials(new_map, prob_maps_dict)

    # 2) add noise to create "scores" used for ranking
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
            noisy[key] = pot  # deterministic

    # 3) eligible donors sorted by descending *noisy* score
    elig = core_functions.calc_eligible_indices_generic(new_map, noisy, annual_rates, sort=True)

    # 4) allocate (same as before)
    H, W = new_map.shape
    assigned = np.zeros((H, W), dtype=bool)
    items = [((int(i), int(j)), max(0, int(round(v)))) for (i, j), v in annual_rates.items() if v > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)

    for (i, j), need in items:
        cand = elig.get((i, j))
        if cand is None or need <= 0:
            continue
        remaining = ~assigned.ravel()[cand]
        if not remaining.any():
            continue
        cand = cand[remaining]
        take = min(need, cand.size)
        if take <= 0:
            continue
        chosen = cand[:take]  # already ranked by noisy score
        new_map.flat[chosen] = float(j)
        assigned.flat[chosen] = True

    return new_map

def simulate_growth_based_scenario(
    initial_year,
    final_year,
    initial_built_up_map, 
    prob_maps_dict,
    change_years,
    annual_rates,
    output_folder,
    parent_folder,
    randomness,
    seed=2118
): 
    simulated_maps, output_paths, class_counts = {}, {}, {}
    
    current_built_up_map = initial_built_up_map.copy()
    
    print("\nDemands (cells/year):")
    for (i, j), rate in sorted(annual_rates.items()):
        print(f"    -> {i} → {j}: {rate}")

    for year in range(final_year - initial_year):
        # Remember previous simulation values to compare with new
        unique_old, counts_old = np.unique(current_built_up_map, return_counts=True)
        
        # Calculate the current year
        current_year = initial_year + year
        
        print("\n###################################################################\n")
        print(f"Current year: {current_year}")
        print(f"Simulating for year: {current_year + 1}")
            
        rng = np.random.default_rng(seed)
        # Run simulation, save map, and overwrite previous year map
        updated_map = update_built_up_map(current_built_up_map, prob_maps_dict, annual_rates,
                                  rng=rng, randomness=randomness, tau=0.1)
        
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

def simulate_density_based_scenario(
    initial_year,
    final_year,
    initial_built_up_map, 
    prob_maps_dict,
    change_years,
    annual_rates,          # baseline per-annum rates for year 1
    differential_change_rates,        # treat as signed change per year (rename to change_rates if you like)
    output_folder,
    parent_folder,
    randomness='gumbel',
    seed=2118
): 
    simulated_maps, output_paths, class_counts = {}, {}, {}
    current_built_up_map = initial_built_up_map.copy()

    # Interpret 'differential_change_rates' as signed per-year deltas (change rates)
    delta_rates = {k: float(v) for k, v in (differential_change_rates or {}).items()}

    # Start from the given baseline for Year 1
    current_rates = {k: float(v) for k, v in annual_rates.items()}

    # Helper: advance rates by one year using the per-year deltas, clamp to >= 0
    def step_rates(prev, delta):
        keys = set(prev) | set(delta)
        nxt = {}
        for k in keys:
            new_val = round(prev.get(k, 0.0) + delta.get(k, 0.0), 2)
            if new_val < 0:
                new_val = 0.0
            nxt[k] = new_val
        return nxt

    # Nice prints of baseline and the first "changed" rates (for Year 2)
    print("\nDemands (cells/year) — Year 1 (baseline):")
    for (i, j), rate in sorted(current_rates.items()):
        print(f"    -> {i} → {j}: {rate}")

    preview_year2 = step_rates(current_rates, delta_rates)
    print("\nDemands (cells/year) — Year 2 (after 1 year of change, clamped ≥ 0):")
    for (i, j), rate in sorted(preview_year2.items()):
        print(f"    -> {i} → {j}: {rate}")

    for year_idx in range(final_year - initial_year):
        # Choose rates for this simulation step
        use_rates = current_rates  # year_idx == 0 uses baseline; afterward it's already evolved

        # Snapshot counts from previous map
        unique_old, counts_old = np.unique(current_built_up_map, return_counts=True)
        
        current_year = initial_year + year_idx
        print("\n###################################################################\n")
        print(f"Current year: {current_year}")
        print(f"Simulating for year: {current_year + 1}")

        rng = np.random.default_rng(seed)

        # Run simulation for this step with the current (per-annum) rates
        updated_map = update_built_up_map(
            current_built_up_map,
            prob_maps_dict,
            use_rates,
            rng=rng,
            randomness=randomness,
            tau=0.1
        )
        
        simulated_maps[current_year + 1] = updated_map
        current_built_up_map = updated_map

        # Save path for this simulated year
        output_paths[current_year + 1] = os.path.join(output_folder, f"sim_{current_year + 1}.tif")

        # Debugging summary
        unique, counts = np.unique(current_built_up_map, return_counts=True)
        unique = [float(x) if not np.isnan(x) else 'nan' for x in unique]
        counts = [int(x) for x in counts]
        print('\n------------------------------------------------------')
        print(f"\nYear {current_year + 1}: Updated built-up map summary (value: count):")
        print(dict(zip(unique, counts)))
        if year_idx > 0:
            unique_old = [float(x) if not np.isnan(x) else 'nan' for x in unique_old]
            curr = {int(k): int(v) for k, v in zip(unique, counts) if k != 'nan'}
            prev = {int(k): int(v) for k, v in zip(unique_old, counts_old) if k != 'nan'}
            all_classes = sorted(set(curr) | set(prev))
            counts_diff = {c: curr.get(c, 0) - prev.get(c, 0) for c in all_classes}
            print("\nChanges from previous year summary (value: count):")
            print(counts_diff)

        # Store class counts
        class_counts[current_year + 1] = [unique, counts]

        # Evolve the rates for the *next* year (so Year 2 uses changed rates, Year 3 uses changed twice, etc.)
        current_rates = step_rates(current_rates, delta_rates)

    return simulated_maps, output_paths, class_counts