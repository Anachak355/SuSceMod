import os
import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from . import data_utilities as utils

_SUITABILITY_REGEX = re.compile(
    r"suitability.*?ref(\d+)_to_(\d+(?:\.\d+)?)", re.IGNORECASE
)
_DEFAULT_EXTS = (".tif", ".tiff", ".img", ".vrt", ".asc")

def normalize_probabilities(prob_maps, clip=True, eps=1e-9):
    out = {}
    for key, arr in prob_maps.items():
        a = np.array(arr, dtype=np.float32, copy=False)
        maxv = np.nanmax(a)
        if np.isnan(maxv) or maxv <= eps:
            # all-NaN or nonpositive → return zeros (keep NaNs as NaNs)
            z = np.zeros_like(a, dtype=np.float32)
            z[np.isnan(a)] = np.nan
            out[key] = z
        else:
            n = a / maxv
            if clip:
                n = np.clip(n, 0.0, 1.0)  # no negatives after normalization
            out[key] = n
    return out


def load_prob_maps(
    root_folder: str,
    *_unused,  # keeps compatibility with your current call signature
    allowed_exts: Tuple[str, ...] = _DEFAULT_EXTS,
    regex: re.Pattern = _SUITABILITY_REGEX,
):
    """
    Recursively find suitability rasters under `root_folder` whose filenames
    contain 'suitability' and match '...ref<from>_to_<to>' (e.g., '...ref11_to_41').
    
    Returns:
        maps_by_transition: Dict[(from_class:int, to_class:int), np.ndarray]
        paths_by_transition: Dict[(from_class:int, to_class:int), str]
    """
    maps_by_transition: Dict[Tuple[int, int], np.ndarray] = {}
    paths_by_transition: Dict[Tuple[int, int], str] = {}

    for dirpath, _, files in os.walk(root_folder):
        for fname in files:
            lower = fname.lower()
            if "suitability" not in lower:
                continue
            if not lower.endswith(allowed_exts):
                continue

            m = regex.search(lower)
            if not m:
                # Name has 'suitability' but doesn't match the ref/to pattern
                continue

            from_id = int(m.group(1))
            to_raw = m.group(2)
            # Handle decimals like "41.0"
            try:
                to_id = int(float(to_raw))
            except ValueError:
                # Skip weird tokens we can't parse to int
                continue

            fpath = os.path.join(dirpath, fname)
            arr = utils.load_raster_data(fpath)  # returns np.ndarray

            key = (from_id, to_id)
            if key in maps_by_transition:
                print(
                    f"[load_prob_maps] Duplicate transition {key} — "
                    f"replacing {paths_by_transition[key]} with {fpath}"
                )

            maps_by_transition[key] = arr
            paths_by_transition[key] = fpath

    if not maps_by_transition:
        raise FileNotFoundError(
            f"No suitability rasters found under '{root_folder}'. "
            f"Expected filenames like 'suitability_ref<classA>_to_<classB>.tif'."
        )
        
    maps_by_transition = normalize_probabilities(maps_by_transition)

    return maps_by_transition, paths_by_transition

def calculate_change_maps(initial_map_for_change_path, final_map_for_change_path):
    """
    Create 'change maps' per origin class:
        change_maps[i] is a raster where pixels that STARTED as class i
        are set to the FINAL class id; everywhere else is NaN.

    Returns:
        change_maps: Dict[int, np.ndarray]
        classes:     Sorted list of all class ids seen in either map (ints)
    """
    init_arr = utils.load_raster_data(initial_map_for_change_path)  # float with NaNs
    final_arr = utils.load_raster_data(final_map_for_change_path)   # float with NaNs

    if init_arr.shape != final_arr.shape:
        raise ValueError(
            f"Initial and final maps have different shapes: "
            f"{init_arr.shape} vs {final_arr.shape}"
        )

    # Unique class ids across both maps (ignore NaNs), cast to int
    init_classes = np.unique(init_arr[~np.isnan(init_arr)]).astype(np.int32)
    final_classes = np.unique(final_arr[~np.isnan(final_arr)]).astype(np.int32)
    classes = sorted(np.unique(np.concatenate([init_classes, final_classes])).tolist())

    change_maps = {}
    for i in classes:
        mask_i = (init_arr == i)
        if not np.any(mask_i):
            # This origin class doesn't exist in the initial map
            continue
        ch = np.full(init_arr.shape, np.nan, dtype=np.float32)
        ch[mask_i] = final_arr[mask_i]
        change_maps[int(i)] = ch

    return change_maps, classes

def compute_crosstab_from_change_maps(change_maps_dict):
    """
    Build a full i→j crosstab (counts) from change_maps_dict where
    each value is a raster of final-class IDs for pixels that STARTED as class i.
    Returns a pandas DataFrame with rows=origin classes, cols=destination classes.
    """
    # Collect all classes seen as origins or destinations
    classes = set(change_maps_dict.keys())
    for ch in change_maps_dict.values():
        vals = np.unique(ch[~np.isnan(ch)])
        classes.update(vals.astype(np.int32).tolist())
    classes = sorted(int(c) for c in classes)

    # Fill the matrix
    idx = {c: k for k, c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=np.int64)

    for i, ch in change_maps_dict.items():
        valid = ~np.isnan(ch)
        if not np.any(valid):
            continue
        dest_vals, dest_counts = np.unique(ch[valid].astype(np.int32), return_counts=True)
        for j, cnt in zip(dest_vals.tolist(), dest_counts.tolist()):
            mat[idx[int(i)], idx[int(j)]] += int(cnt)

    # DataFrame for convenience (rows: from, cols: to)
    crosstab_df = pd.DataFrame(mat, index=classes, columns=classes)
    return crosstab_df


def calculate_change_rates(change_maps_dict, change_years, print_summary=True):
    """
    Convert the crosstab into annual demands (cells/year) for i→j, i≠j.
    change_years = [start_year, end_year]  → years = end - start (no div-by-zero).
    Returns: (annual_change_rates, period_counts, crosstab_df)
    """
    start_year, end_year = int(change_years[0]), int(change_years[1])
    years = max(1, end_year - start_year)

    crosstab_df = compute_crosstab_from_change_maps(change_maps_dict)

    # Period totals for actual transitions (exclude diagonal / no-change)
    period_counts = {}
    for i in crosstab_df.index:
        for j in crosstab_df.columns:
            if i == j:
                continue
            cnt = int(crosstab_df.loc[i, j])
            if cnt > 0:
                period_counts[(int(i), int(j))] = cnt

    # Annualize
    annual_change_rates = {k: round(v / years, 2) for k, v in period_counts.items()}

    if print_summary:
        print("\nCrosstab (counts, incl. no-change on diagonal):")
        print(crosstab_df)
        print("\nAnnual change rates (cells/year):")
        for (i, j), rate in sorted(annual_change_rates.items()):
            print(f"    -> {i} → {j}: {rate}")

    return annual_change_rates, period_counts, crosstab_df

def calculate_change_rates_historical(change_maps_dict_1, change_maps_dict_2, change_years, print_summary=True):
    if not isinstance(change_years, (list, tuple)) or len(change_years) != 3:
        raise ValueError("change_years must be [start1, split, end2].")

    start1, split, end2 = map(int, change_years)

    # Use your existing function (no duplication)
    annual_rates_1, _, _ = calculate_change_rates(
        change_maps_dict_1, [start1, split], print_summary=False
    )
    annual_rates_2, period_counts_2, crosstab_df_2 = calculate_change_rates(
        change_maps_dict_2, [split, end2], print_summary=False
    )

    # Decrease = positive drop from dict_1 → dict_2
    all_keys = set(annual_rates_1).union(annual_rates_2)
    change_rates = {}
    for k in all_keys:
        dec = round(annual_rates_2.get(k, 0.0) - annual_rates_1.get(k, 0.0), 2)
        if dec < 0:
            change_rates[k] = dec

    if print_summary:
        print(f"\nDict_2 crosstab (counts incl. diagonal) for {split}–{end2}:")
        print(crosstab_df_2)
        print("\nDict_2 annual change rates (cells/year):")
        for (i, j), rate in sorted(annual_rates_2.items()):
            print(f"    -> {i} → {j}: {rate}")
        if change_rates:
            print(f"\nChanges vs. dict_1 [{start1}–{split}] (cells/year):")
            for (i, j), dec in sorted(change_rates.items()):
                print(f"    ↓ {i} → {j}: {dec}")
        else:
            print(f"\nNo Changes vs. dict_1 [{start1}–{split}].")

    return annual_rates_2, change_rates, period_counts_2, crosstab_df_2