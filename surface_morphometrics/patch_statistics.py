#! /usr/bin/env python
"""Per-region statistics for membrane patches or connected components.

This reads the per-triangle CSVs written by `generate_patches.py` (edited in place
into the membrane `*.AVV_rh*.csv`) or `label_connected_components.py`
(`*_components.csv`) and computes area-weighted summary statistics for each region
(each patch or component), for any set of per-triangle properties.

Because both tools tag triangles with an integer region id (`<prefix>_patch_number`,
`<prefix>_random_patch_number`, or `component_number`), the same aggregation works
for all of them — protein patches, their random controls, and whole connected
components — producing one tidy CSV for downstream plotting/statistics. The
`region_type` column keeps the prefix (e.g. ribo_patch, ribo_random_patch), so
multiple patch types measured into one surface stay distinct. Real-patch rows also
get `min_protein_distance` — the closest approach of the membrane midplane to that
patch's protein — and, when membrane thickness was available, `headgroup_distance`,
the distance to the true membrane edge (headgroups) measured at the patch's central
triangle (min_protein_distance minus half the effective thickness there).

Usage:
  patch_statistics.py config.yml                          # batch over work_dir CSVs
  patch_statistics.py config.yml --csv TS1_IMM.AVV_rh9.csv
  patch_statistics.py config.yml --properties curvedness_VV,thickness --output out.csv

Region id 0 (triangles in no patch / dropped components) is always excluded.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

import click
import numpy as np
import pandas as pd
import yaml

from .config_utils import load_config

def detect_label_columns(columns):
    """Map each region-label column present in `columns` to its region_type.

    generate_patches writes prefixed columns (e.g. ribo_patch_number,
    ribo_random_patch_number); the region_type keeps the prefix so several patch
    types measured into one surface (ribo_patch vs atp_patch) stay distinct in the
    output. label_connected_components writes component_number. Legacy unprefixed
    patch columns are still recognized for older CSVs.
    """
    detected = {}
    for c in columns:
        if c == "component_number":
            detected[c] = "component"
        elif c == "patch_number":            # legacy (unprefixed) real patches
            detected[c] = "patch"
        elif c == "patch_random_number":     # legacy (unprefixed) random patches
            detected[c] = "random"
        elif c.endswith("_random_patch_number"):
            detected[c] = c[:-len("_number")]   # e.g. ribo_random_patch
        elif c.endswith("_patch_number"):
            detected[c] = c[:-len("_number")]   # e.g. ribo_patch
    return detected


def _weighted_mean(values, weights):
    """Area-weighted mean, ignoring NaN weights/values."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total == 0:
        return np.nan
    return float(np.sum(values * weights) / total)


def aggregate_regions(df, label_col, value_cols, area_col="area",
                      drop_zero_values=True, min_cols=None):
    """Aggregate per-triangle data into per-region rows for one label column.

    Parameters
    ----------
    df : DataFrame of per-triangle properties (one row per triangle).
    label_col : column holding the integer region id (0 = excluded).
    value_cols : list of property columns to summarize.
    area_col : column with per-triangle area for area weighting. If missing,
        falls back to unweighted means (each triangle weight 1).
    drop_zero_values : if True, drop triangles where the value is 0 or NaN
        before summarizing that property (matches the original per-patch
        scripts, where 0 means "unmeasured").
    min_cols : optional {output_name: source_column} mapping. For each entry, the
        per-region NaN-ignoring minimum of source_column is written to output_name
        (e.g. {"min_protein_distance": "ribo_protein_distance"}). Used for the
        closest membrane-to-protein and membrane-edge-to-protein approaches.

    Returns
    -------
    DataFrame: one row per region with n_triangles, total_area,
    `{prop}_mean` (area-weighted) + `{prop}_median` (area-weighted) per property,
    plus each requested `min_cols` output.
    """
    from .morphometrics_stats import weighted_median

    have_area = area_col in df.columns
    min_cols = {name: src for name, src in (min_cols or {}).items() if src in df.columns}
    rows = []
    region_ids = sorted(int(r) for r in df[label_col].unique() if r != 0)
    for rid in region_ids:
        region = df[df[label_col] == rid]
        area_all = region[area_col].to_numpy(dtype=float) if have_area else \
            np.ones(len(region))
        row = {
            "region_id": rid,
            "n_triangles": len(region),
            "total_area": float(area_all.sum()),
        }
        for out_name, src_col in min_cols.items():
            # Closest approach within this patch. The source column is NaN outside
            # the patch, so ignore NaN and take the min.
            mv = region[src_col].to_numpy(dtype=float)
            mv = mv[~np.isnan(mv)]
            row[out_name] = float(np.min(mv)) if len(mv) else np.nan
        for col in value_cols:
            if col not in region.columns:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                continue
            vals = region[col].to_numpy(dtype=float)
            weights = area_all
            mask = ~np.isnan(vals)
            if drop_zero_values:
                mask &= vals != 0
            vals, weights = vals[mask], weights[mask]
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                continue
            row[f"{col}_mean"] = _weighted_mean(vals, weights)
            row[f"{col}_median"] = float(weighted_median(vals, weights))
        rows.append(row)
    return pd.DataFrame(rows)


def _real_patch_prefix(label_col):
    """Prefix for a real-patch label column (<prefix>_patch_number), else None.

    Random patches and connected components have no associated protein, so they
    return None. Legacy unprefixed `patch_number` returns "" (empty prefix).
    """
    if label_col.endswith("_random_patch_number"):
        return None
    if label_col == "patch_number":                       # legacy (unprefixed)
        return ""
    if label_col.endswith("_patch_number"):
        return label_col[:-len("_patch_number")]
    return None


def protein_distance_col(label_col, columns):
    """The protein-distance column matching a real-patch label column, else None."""
    prefix = _real_patch_prefix(label_col)
    if prefix is None:
        return None
    cand = f"{prefix}_protein_distance" if prefix else "protein_distance"
    return cand if cand in columns else None


def compute_headgroup_distances(df, label_col, protein_col):
    """Per-patch distance from the protein to the true membrane edge (headgroups).

    Returns {region_id: headgroup_distance}. Measured at the patch's central triangle
    (the nearest triangle to the protein): the closest protein distance minus half
    the effective membrane thickness there. The thickness is the center triangle's
    own thickness, or — if that is unmeasured — the patch's distance-from-center
    weighted mean thickness, or NaN if the patch has no measured thickness. Computing
    it at a single triangle (rather than min-over-per-triangle-values) guarantees
    headgroup_distance <= min_protein_distance.
    """
    from .generate_patches import effective_thickness

    prefix = _real_patch_prefix(label_col)
    if prefix is None or "thickness" not in df.columns:
        return {}
    center_col = f"{prefix}_patch_center" if prefix else "patch_center"
    cdist_col = f"{prefix}_patch_center_distance" if prefix else "patch_center_distance"
    have_center = center_col in df.columns
    have_cdist = cdist_col in df.columns

    result = {}
    for rid in sorted(int(r) for r in df[label_col].unique() if r != 0):
        region = df[df[label_col] == rid]
        protein = region[protein_col].to_numpy(dtype=float)
        protein = protein[~np.isnan(protein)]
        if len(protein) == 0:
            continue
        center_protein = float(np.min(protein))  # nearest triangle == patch center
        center_thick = float("nan")
        if have_center:
            crow = df.loc[df[center_col] == rid, "thickness"].to_numpy(dtype=float)
            if len(crow):
                center_thick = float(crow[0])
        th = region["thickness"].to_numpy(dtype=float)
        cd = region[cdist_col].to_numpy(dtype=float) if have_cdist else np.zeros(len(region))
        t_eff = effective_thickness(center_thick, th, cd)
        result[rid] = center_protein - t_eff / 2.0 if np.isfinite(t_eff) else float("nan")
    return result


def aggregate_csv(csv_file, value_cols, label_col=None, drop_zero_values=True):
    """Aggregate one CSV across all applicable label columns.

    Returns a tidy DataFrame tagged with `source` (the CSV base name) and
    `region_type` (e.g. ribo_patch / ribo_random_patch / component). Real-patch
    rows also carry `min_protein_distance` and (when thickness is available)
    `headgroup_distance`.
    """
    df = pd.read_csv(csv_file)
    base = os.path.basename(csv_file)
    for suffix in ("_patches.csv", "_components.csv", ".csv"):
        if base.endswith(suffix):
            source = base[: -len(suffix)]
            break
    else:
        source = base

    if label_col is not None:
        label_cols = {label_col: detect_label_columns([label_col]).get(label_col, label_col)}
    else:
        label_cols = detect_label_columns(df.columns)

    if not label_cols:
        print(f"  no region-label column in {base}; skipping")
        return pd.DataFrame()

    out = []
    for col, region_type in label_cols.items():
        min_cols = {}
        pcol = protein_distance_col(col, df.columns)
        if pcol:
            min_cols["min_protein_distance"] = pcol
        part = aggregate_regions(df, col, value_cols,
                                 drop_zero_values=drop_zero_values,
                                 min_cols=min_cols)
        if part.empty:
            continue
        # Headgroup distance: measured at each patch's central triangle (so it is
        # always <= min_protein_distance), with the thickness fallback.
        if pcol:
            hg = compute_headgroup_distances(df, col, pcol)
            if hg:
                idx = part.columns.get_loc("min_protein_distance") + 1
                part.insert(idx, "headgroup_distance", part["region_id"].map(hg))
        part.insert(0, "source", source)
        part.insert(1, "region_type", region_type)
        out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


@click.command()
@click.argument("configfile", type=click.Path(exists=True))
@click.option("--csv", "csv_file", type=click.Path(exists=True), default=None,
              help="Single CSV to process (overrides batch mode).")
@click.option("--properties", default=None,
              help="Comma-separated per-triangle properties to summarize "
                   "(defaults to patch_analysis.statistics_properties).")
@click.option("--label-col", "label_col", default=None,
              help="Force a single region-label column (e.g. ribo_patch_number, "
                   "component_number). Default: auto-detect any <prefix>_patch_number "
                   "/ <prefix>_random_patch_number / component_number column.")
@click.option("--pattern", default=None,
              help="Glob (within work_dir) of CSVs to aggregate in batch mode "
                   "(default: *.AVV_rh<radius_hit>.csv, the in-place membrane CSVs).")
@click.option("--output", "output_csv", default=None,
              help="Output CSV path (default: work_dir/patch_statistics.csv).")
@click.option("--keep-zeros", is_flag=True, default=False,
              help="Keep triangles whose value is 0 (default drops them, like the "
                   "original per-patch scripts).")
def patch_statistics_cli(configfile, csv_file, properties, label_col, pattern,
                         output_csv, keep_zeros):
    """Compute per-region (patch/component) area-weighted statistics.

    CONFIGFILE: path to config.yml.
    """
    config = load_config(configfile, require=("work_dir",))

    pa_config = config.get("patch_analysis", {})
    work_dir = config.get("work_dir", config.get("seg_dir", "./"))
    if not work_dir.endswith("/"):
        work_dir += "/"
    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)

    if properties is not None:
        value_cols = [p.strip() for p in properties.split(",") if p.strip()]
    else:
        value_cols = pa_config.get("statistics_properties",
                                   ["curvedness_VV", "thickness"])

    drop_zero_values = not keep_zeros
    print("Patch statistics settings:")
    print(f"  Properties: {value_cols}")
    print(f"  Drop zero values: {drop_zero_values}")
    print(f"  Region label column: {label_col or 'auto-detect'}")

    if csv_file is not None:
        files = [csv_file]
    else:
        # generate_patches now edits the membrane CSVs in place, so the default
        # scans the enriched AVV surfaces; those without patch columns are skipped.
        pattern = pattern or f"*.AVV_rh{radius_hit}.csv"
        files = sorted(glob(work_dir + pattern))
        if not files:
            print(f"No CSVs matching {work_dir}{pattern}")
            return
    print(f"  Processing {len(files)} CSV file(s)")

    results = []
    for f in files:
        print(f"Aggregating {os.path.basename(f)}")
        part = aggregate_csv(f, value_cols, label_col=label_col,
                             drop_zero_values=drop_zero_values)
        if not part.empty:
            results.append(part)

    if not results:
        print("No regions aggregated.")
        return
    combined = pd.concat(results, ignore_index=True)

    if output_csv is None:
        output_csv = os.path.join(work_dir, "patch_statistics.csv")
    combined.to_csv(output_csv, index=False)
    print(f"\nWrote {len(combined)} region rows to {output_csv}")
    # Brief summary per region type.
    for region_type, grp in combined.groupby("region_type"):
        print(f"  {region_type}: {len(grp)} regions across {grp['source'].nunique()} source(s)")


if __name__ == "__main__":
    patch_statistics_cli()
