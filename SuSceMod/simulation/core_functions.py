# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:12:01 2024

@author: Administrator
"""

import numpy as np
import input_handling as inp

def neighborhood_counts_for_targets(built_up_map, target_classes):
    """
    For each target class j, compute the 8-neighbor count of class j for every cell.
    Vectorized, no loops. Uses zero padding (no wrap-around).

    Returns:
        counts_by_class: Dict[j:int] -> 2D np.ndarray (int16/32)
    """
    counts_by_class = {}
    H, W = built_up_map.shape
    a = built_up_map.astype(np.int32, copy=False)

    for j in sorted(set(int(x) for x in target_classes)):
        # Binary mask for class j
        M = (a == j).astype(np.uint8)

        # Zero-pad by 1 on all sides
        P = np.pad(M, 1, mode='constant', constant_values=0)

        # Sum 3x3 neighborhood (including center) via 9 shifted views
        S = (
            P[0:-2, 0:-2] + P[0:-2, 1:-1] + P[0:-2, 2:] +
            P[1:-1, 0:-2] + P[1:-1, 1:-1] + P[1:-1, 2:] +
            P[2:  , 0:-2] + P[2:  , 1:-1] + P[2:  , 2:]
        )

        # Exclude the center cell itself
        counts = S - M
        counts_by_class[j] = counts.astype(np.int16, copy=False)

    return counts_by_class


def calc_transition_potentials(built_up_map, prob_maps_dict):
    """
    Generic transition potentials for any (i->j) in prob_maps_dict.
    Potential formula (as in your original): sqrt( prob_ij * neigh_count_of_j )

    Args:
        built_up_map: 2D float array with NaNs allowed; class ids in integers
        prob_maps_dict: Dict[(i:int, j:int)] -> 2D float array (normalized [0,1], NaNs ok)

    Returns:
        potentials: Dict[(i,j)] -> 2D float array (potential; 0 outside donor class i or where prob is NaN/0)
    """
    H, W = built_up_map.shape
    a = built_up_map  # float (may contain NaNs)
    valid = ~np.isnan(a)

    # Collect the set of target classes we need neighborhood counts for
    target_classes = {int(j) for (_, j) in prob_maps_dict.keys()}
    neigh_counts = neighborhood_counts_for_targets(a, target_classes)

    potentials = {}
    for (i, j), prob in prob_maps_dict.items():
        i = int(i); j = int(j)
        # Donor mask: only cells currently in class i and valid
        donor_mask = (valid & (a.astype(np.int32) == i))

        # Probabilities: treat NaN/±inf as 0
        p = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # Neighborhood count for target class j
        n_j = neigh_counts[j]

        # Potential = sqrt(prob * neighbor_count_of_j) on donor cells, else 0
        pot = np.zeros((H, W), dtype=np.float32)
        if donor_mask.any():
            # compute only where donor_mask; elsewhere remains 0
            m = donor_mask
            # Avoid negative due to any weird input; clip
            s = np.sqrt(np.clip(p[m] * n_j[m], 0.0, None, dtype=np.float32), dtype=np.float32)
            pot[m] = s

        potentials[(i, j)] = pot

    return potentials


def calc_transition_acceptance(trans_pot_arrays):
    trans_bool_arrays = [trans_pot_arrays[k] > np.random.rand(*trans_pot_arrays[k].shape) for k in range(6)]
    trans_bool_exp0_1, trans_bool_exp0_2, trans_bool_exp0_3, trans_bool_dens1_2, trans_bool_dens1_3, trans_bool_dens2_3 = trans_bool_arrays
    
    trans_bool_exp = [trans_bool_exp0_1, trans_bool_exp0_2, trans_bool_exp0_3]
    trans_bool_dens = [trans_bool_dens1_2, trans_bool_dens1_3, trans_bool_dens2_3]
    
    return trans_bool_exp, trans_bool_dens

def calc_eligible_indices_generic(built_up_map, potentials, annual_rates, sort=True):
    """
    Build eligible donor indices per transition (i→j).

    Args:
        built_up_map : 2D float array (classes as ints; NaNs = nodata)
        potentials   : Dict[(i:int, j:int)] -> 2D float array (transition potential for i→j)
        annual_rates : Dict[(i:int, j:int)] -> float (cells/year); pairs <=0 are ignored
        sort         : If True, return indices sorted by descending potential

    Returns:
        elig : Dict[(i:int, j:int)] -> 1D np.ndarray of flat indices (eligible donors)
               (sorted by potential if sort=True; unsorted otherwise)
    """
    a = built_up_map
    valid = ~np.isnan(a)
    a_int = a.astype(np.int32, copy=False)

    elig = {}
    for (i, j), rate in annual_rates.items():
        if rate <= 0:
            continue

        i = int(i); j = int(j)
        pot = potentials.get((i, j), None)
        if pot is None:
            continue

        p = np.asarray(pot, dtype=np.float32)
        # donors: currently class i, valid cell, finite potential, positive potential
        donor_mask = (valid & (a_int == i) & np.isfinite(p) & (p > 0))
        if not donor_mask.any():
            continue

        flat_scores = p.ravel()
        cand = np.flatnonzero(donor_mask.ravel())

        if sort:
            order = np.argsort(flat_scores[cand])[::-1]  # highest potential first
            cand = cand[order]

        elig[(i, j)] = cand

    return elig



def calc_eligible_indices_fast(built_up_map, trans_pot_arrays, water_zone=None, central_zones=None, allocate_central=False, sorting=False):
    # Set defaults for optional arguments
    if water_zone is None:
        water_zone = np.zeros_like(built_up_map)
    if central_zones is None:
        central_zones = np.zeros_like(built_up_map)
    
    # Define masks for each condition
    is_empty = built_up_map == 0
    is_built = np.isin(built_up_map, [1, 2])
    not_in_water = water_zone != 1
    in_central = central_zones == 1
    not_in_central = central_zones == 0
    
    # Apply conditions based on arguments
    if np.any(central_zones):
        if allocate_central:
            zone_mask = in_central
        else:
            zone_mask = not_in_central
    else:
        zone_mask = np.ones_like(built_up_map, dtype=bool)  # No central zones, all valid

    valid_expansion_mask = is_empty & zone_mask & not_in_water
    valid_densification_mask = is_built & zone_mask & not_in_water

    # Get eligible indices
    expansions = np.argwhere(valid_expansion_mask)
    densifications = np.argwhere(valid_densification_mask)

    # Apply sorting if required
    if sorting:
        def expansion_key(pos):
            return sum(trans_pot_arrays[i][tuple(pos)] for i in range(3))
        
        def densification_key(pos):
            return sum(trans_pot_arrays[i + 3][tuple(pos)] for i in range(3))
        
        expansions = sorted(expansions, key=expansion_key, reverse=True)
        densifications = sorted(densifications, key=densification_key, reverse=True)

    return expansions, densifications

def calculate_decrease_rate(initial_year, final_year, annual_change_rates):
    decrease_rates = [round(a_r/((final_year - initial_year)), 2) for a_r in annual_change_rates]
    print("Annual decrease in rate of expansion: ", decrease_rates, "hectare/year")
    return decrease_rates

def calculate_central_percentages(initial_year, final_year, initial_central_percentages, final_central_percentages):
    
    inc_vals = [(final_central_percentages[i] - initial_central_percentages[i]) / (final_year - initial_year) for i in range(len(initial_central_percentages))  ]
    percentages = {}
    for i in range(final_year - initial_year + 1):
        year = initial_year + i
        percentages[year] = [
            round(initial_central_percentages[0] + inc_vals[0] * i, 2),
            round(initial_central_percentages[1] + inc_vals[1] * i, 2)
        ]
        
    return percentages

def check_decimals(current_demands_exp_central, current_demands_exp_non_central, current_demands_dens_central, current_demands_dens_non_central, current_demands_exp, current_demands_dens):
    cdec = np.floor(current_demands_exp_central).astype(int)
    cdenc = np.floor(current_demands_exp_non_central).astype(int)
    cddc = np.floor(current_demands_dens_central).astype(int)
    cddnc = np.floor(current_demands_dens_non_central).astype(int)
    cde = np.floor(current_demands_exp).astype(int)
    cdd = np.floor(current_demands_dens).astype(int)
    
    
    for i in range(len(cde)):
        if cdec[i] + cdenc[i] < cde[i]:
            current_demands_exp_central[i] += 1
            
    for i in range(len(cdd)):
        if cddc[i] + cddnc[i] < cdd[i]:
            current_demands_dens_central[i] += 1
    
    return current_demands_exp_central, current_demands_exp_non_central, current_demands_dens_central, current_demands_dens_non_central

def extrapolate_demand_yearly(parent_raster_folder, zoning_type, change_years):
    annual_change_rates_exp_1, annual_change_rates_dens_1 = inp.calculate_change_rates(parent_raster_folder, zoning_type, change_years[0])
    annual_change_rates_exp_2, annual_change_rates_dens_2 = inp.calculate_change_rates(parent_raster_folder, zoning_type, change_years[1])
    
    decrease_rates_exp = [(a-b)/10 for a, b in zip(annual_change_rates_exp_1, annual_change_rates_exp_2)]
    decrease_rates_dens = [(a-b)/10 for a, b in zip(annual_change_rates_dens_1, annual_change_rates_dens_2)]
    
    return decrease_rates_exp, decrease_rates_dens

def extrapolate_demand(parent_raster_folder, zoning_type, change_years):
    annual_change_rates_exp_1, annual_change_rates_dens_1 = inp.calculate_change_rates(parent_raster_folder, zoning_type, change_years[0])
    annual_change_rates_exp_2, annual_change_rates_dens_2 = inp.calculate_change_rates(parent_raster_folder, zoning_type, change_years[1])
    
    decrease_rates_exp = [(a-b) for a, b in zip(annual_change_rates_exp_1, annual_change_rates_exp_2)]
    decrease_rates_dens = [(a-b) for a, b in zip(annual_change_rates_dens_1, annual_change_rates_dens_2)]
    
    return decrease_rates_exp, decrease_rates_dens