import numpy as np

def neighborhood_counts_for_targets(built_up_map, target_classes):
    """
    For each target class j, compute the 8-neighbor count of class j for every cell.
    Vectorized, no loops. Uses zero padding (no wrap-around).

    Returns:
        counts_by_class: Dict[j:int] -> 2D np.ndarray (int16/32)
    """
    counts_by_class = {}
    H, W = built_up_map.shape
    a = np.nan_to_num(built_up_map, nan=-9999).astype(np.int32)

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
        donor_mask = (valid & (np.nan_to_num(a, nan=-9999).astype(np.int32) == i))

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

def calc_eligible_indices_generic(built_up_map, potentials, annual_rates, sort=True, zone_mask=None):
    """
    Build eligible donor indices per transition (i→j).

    Args:
        built_up_map : 2D float array (classes as ints; NaNs = nodata)
        potentials   : Dict[(i:int, j:int)] -> 2D float array (transition potential for i→j)
        annual_rates : Dict[(i:int, j:int)] -> float (cells/year); pairs <=0 are ignored
        sort         : If True, return indices sorted by descending potential
        zone_mask    : Optional 2D bool array; only cells where it is True can be donors

    Returns:
        elig : Dict[(i:int, j:int)] -> 1D np.ndarray of flat indices (eligible donors)
               (sorted by potential if sort=True; unsorted otherwise)
    """
    a = built_up_map
    valid = ~np.isnan(a)
    a_int = np.nan_to_num(a, nan=-9999).astype(np.int32)

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
        if zone_mask is not None:
            donor_mask &= zone_mask
        if not donor_mask.any():
            continue

        flat_scores = p.ravel()
        cand = np.flatnonzero(donor_mask.ravel())

        if sort:
            order = np.argsort(flat_scores[cand])[::-1]  # highest potential first
            cand = cand[order]

        elig[(i, j)] = cand

    return elig
