#! /usr/bin/env python
"""Generate membrane patches centered on protein particles from a STAR file.

For each particle in a STAR file, the nearest triangle of a membrane graph is
found and used as the center of a circular "patch" (all triangles within
`patch_radius`).  Each patch is identified by the particle's STAR line ID, so
patches map directly back to particles.  Optionally, matched random control
patches are generated, and the STAR file is annotated with the per-particle
distance to the membrane for downstream particle-side filtering (e.g. selecting
cotranslating ribosomes).

This generalizes GrotjahnLab/patch_analysis/find_IMM_patches_for_ATP_synthase.py
to any membrane and any particle set, driven by config.yml like the rest of the
surface morphometrics pipeline.

STAR compatibility: RELION 3/4 only. Their rlnCoordinateX/Y/Z are absolute
tomogram-frame coordinates that match the surface frame. RELION 5 tomography STAR
files (rlnCenteredCoordinate*Angst, relative to the tomogram center) are NOT yet
supported and must be converted to absolute coordinates first.

Two usage modes:
1. Batch over all tomograms in work_dir for the configured membrane label:
     generate_patches.py config.yml
2. A single membrane graph + STAR file:
     generate_patches.py config.yml --graph TS1_IMM.AVV_rh8.gt --star TS1.star

IMPORTANT - which particles are used:
By default EVERY particle in the STAR file is matched against the membrane, so
the STAR must contain only the particles for that one tomogram. If you have a
combined STAR with particles from many tomograms, set `star_tomo_column` (the
column holding the tomogram/micrograph name, e.g. rlnMicrographName or
rlnTomoName) so the tool keeps only the rows that match the tomogram. Matching is
by basename (bidirectional substring), so it tolerates directory paths, .mrc
extensions, and pixel-size/bin suffixes (e.g. `TS_004` matches
`/data/TS_004.mrc_6.65Apx.mrc`). `patch_id` still refers to the row number in the
original STAR.

Outputs are written IN PLACE: the patch properties are added back into the input
membrane .gt / .vtp / .csv (no separate *_patches file set). Measurement names are
prefixed by `patch_analysis.measurement_prefix` (default "ribo"), so different
patch types (ribosome, ATP synthase, ...) can be measured into the same surface
without colliding — just re-run with a different prefix.

Vertex properties added (0 means "not in any patch"; <p> = the configured prefix):
  <p>_patch_number, <p>_patch_center             - real patches (= STAR line ID)
  <p>_patch_center_distance, <p>_protein_distance
  <p>_random_patch_number, <p>_random_patch_center, <p>_random_patch_center_distance

When per-triangle `thickness` is present, a headgroup distance (distance to the true
membrane edge = protein distance - thickness/2, measured at the patch's central
triangle) is also written per-particle into the annotated STAR as
<p>_headgroup_distance, and reported per-patch by `morphometrics patch_statistics`.
"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import os
from glob import glob

import click
import numpy as np
import yaml

from .config_utils import load_config

# ---------------------------------------------------------------------------
# Pure-geometry helpers (no pycurv/graph-tool dependency, so they are unit
# testable on plain numpy arrays).
# ---------------------------------------------------------------------------


def nearest_triangle(triangle_xyz, particle_xyz):
    """Nearest membrane triangle for each particle.

    Parameters
    ----------
    triangle_xyz : (T, 3) array of triangle-center coordinates (membrane units).
    particle_xyz : (P, 3) array of particle coordinates (same units).

    Returns
    -------
    (min_d, min_i) : distance to and index of the nearest triangle per particle.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(triangle_xyz)
    min_d, min_i = tree.query(particle_xyz, k=1)
    return min_d, min_i


def assign_patches(triangle_xyz, center_indices, center_ids, patch_radius,
                   particle_xyz=None):
    """Assign triangles to patches around the given centers.

    A triangle inside more than one patch is assigned to the patch whose center
    is closest; equal distances are broken by last-added (centers later in
    `center_indices` win ties).

    Parameters
    ----------
    triangle_xyz : (T, 3) array of triangle-center coordinates.
    center_indices : sequence of triangle indices used as patch centers.
    center_ids : sequence of integer ids (one per center) written into the
        label array for that patch's triangles. Must be > 0.
    patch_radius : float, patch radius in the same units as the coordinates.
    particle_xyz : optional (len(center_indices), 3) array of the particle
        position for each patch; if given, returns protein_distance too.

    Returns
    -------
    dict with arrays of length T:
      number              - patch id per triangle (0 = none)
      center              - patch id only on the center triangle (0 elsewhere)
      center_distance     - distance to assigned center (nan where number == 0)
      protein_distance    - distance to the patch's particle (nan; only if
                            particle_xyz given)
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(triangle_xyz)

    n = len(triangle_xyz)
    number = np.zeros(n, dtype=np.int64)
    center = np.zeros(n, dtype=np.int64)
    center_distance = np.full(n, np.inf)
    protein_distance = np.full(n, np.nan)

    for k, (ci, cid) in enumerate(zip(center_indices, center_ids)):
        center_xyz = triangle_xyz[ci]
        member_idx = tree.query_ball_point(center_xyz, patch_radius)
        if not member_idx:
            continue
        member_idx = np.asarray(member_idx)
        dists = np.linalg.norm(triangle_xyz[member_idx] - center_xyz, axis=1)
        # Nearest-center-wins: take this patch where it is closer than (or tied
        # with, last-added) the currently assigned center.
        take = dists <= center_distance[member_idx]
        chosen = member_idx[take]
        number[chosen] = cid
        center_distance[chosen] = dists[take]
        if particle_xyz is not None:
            pxyz = particle_xyz[k]
            protein_distance[chosen] = np.linalg.norm(
                triangle_xyz[chosen] - pxyz, axis=1)
        # The center triangle carries the patch id in the center array.
        center[ci] = cid

    center_distance[number == 0] = np.nan
    return {
        "number": number,
        "center": center,
        "center_distance": center_distance,
        "protein_distance": protein_distance,
    }


def choose_random_centers(triangle_xyz, n_centers, min_distance, rng,
                          exclude_indices=None):
    """Pick n_centers triangle indices at least min_distance apart.

    Selection is over triangle indices directly (no float-coordinate lookups).
    Returns fewer than n_centers (with a warning left to the caller) if the
    min-distance constraint cannot be satisfied.
    """
    n = len(triangle_xyz)
    order = rng.permutation(n)
    exclude = set(int(i) for i in exclude_indices) if exclude_indices is not None else set()
    chosen = []
    chosen_xyz = []
    for idx in order:
        if len(chosen) >= n_centers:
            break
        if int(idx) in exclude:
            continue
        cxyz = triangle_xyz[idx]
        if chosen_xyz:
            d = np.linalg.norm(np.asarray(chosen_xyz) - cxyz, axis=1)
            if np.any(d < min_distance):
                continue
        chosen.append(int(idx))
        chosen_xyz.append(cxyz)
    return chosen


def effective_thickness(center_thickness, thicknesses, center_distances):
    """Membrane thickness to use for a patch's headgroup distance.

    Prefer the central triangle's own thickness. If that is unmeasured (NaN or
    <= 0), fall back to the distance-from-center weighted mean of the patch's
    measured thicknesses (closer triangles weighted more, weight = 1/(1+distance)).
    If the patch has no measured thickness at all, return NaN.
    """
    if np.isfinite(center_thickness) and center_thickness > 0:
        return float(center_thickness)
    th = np.asarray(thicknesses, dtype=float)
    cd = np.asarray(center_distances, dtype=float)
    mask = np.isfinite(th) & (th > 0)
    if not mask.any():
        return float("nan")
    w = 1.0 / (1.0 + cd[mask])
    return float(np.sum(th[mask] * w) / np.sum(w))


def _particle_headgroup_distances(triangle_xyz, thickness, min_d, min_i, patch_radius):
    """Per-particle distance to the true membrane edge (headgroups).

    For each particle: its mesh distance minus half the effective thickness at its
    nearest (patch-center) triangle, using the triangles within `patch_radius` of
    that center as the fallback for `effective_thickness`. NaN when the patch has no
    measured thickness.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(triangle_xyz)
    out = np.full(len(min_i), np.nan)
    for p in range(len(min_i)):
        ci = int(min_i[p])
        nbr = np.asarray(tree.query_ball_point(triangle_xyz[ci], patch_radius))
        cd = np.linalg.norm(triangle_xyz[nbr] - triangle_xyz[ci], axis=1)
        t_eff = effective_thickness(thickness[ci], thickness[nbr], cd)
        if np.isfinite(t_eff):
            out[p] = min_d[p] - t_eff / 2.0
    return out


# ---------------------------------------------------------------------------
# STAR / config helpers
# ---------------------------------------------------------------------------


def _tomo_stem(name):
    """Basename of a tomogram reference, minus directory and a trailing .mrc."""
    base = os.path.basename(str(name).strip().replace("\\", "/").rstrip("/"))
    if base.lower().endswith(".mrc"):
        base = base[:-4]
    return base


def tomo_name_matches(star_value, tomo_name):
    """Whether a STAR tomogram/micrograph value refers to `tomo_name`.

    Matches by basename with bidirectional substring, so it tolerates directory
    paths, .mrc extensions, and pixel-size / bin suffixes (e.g. a graph named
    `TS_004` matching a STAR micrograph `/data/TS_004.mrc_6.65Apx.mrc`, or vice
    versa).
    """
    sv = _tomo_stem(star_value)
    tn = _tomo_stem(tomo_name)
    if not sv or not tn:
        return False
    return tn in sv or sv in tn


def load_particle_coordinates(star_file, pa_config, angstroms, tomo_name=None):
    """Load particle coordinates from a STAR file.

    Returns (df, coords_nm, line_ids), where line_ids are the 1-based row numbers
    in the original STAR (stable even when the STAR is filtered to one tomogram).
    coords are converted to the membrane's units: nm by default (Angstrom values
    divided by 10), or left in Angstrom when `angstroms` is True.

    If `star_tomo_column` is set in the config and `tomo_name` is given, only rows
    whose value in that column contains `tomo_name` are kept (for combined STAR
    files spanning many tomograms).

    NOTE: RELION 3/4 STAR files only. Those store absolute tomogram-frame
    coordinates (rlnCoordinateX/Y/Z in pixels from voxel 0), which line up with the
    surface frame directly. RELION 5 tomography STAR files use
    rlnCenteredCoordinate*Angst (relative to the tomogram center) and are NOT
    currently supported: they would need the tomogram box size added as a
    half-extent offset. Convert them to absolute coordinates before use.
    """
    import starfile
    star = starfile.read(star_file)
    coord_cols = pa_config.get("star_coord_columns",
                               ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"])
    # starfile.read can return a dict of blocks; take the particles block.
    if isinstance(star, dict):
        block = None
        for value in star.values():
            if all(c in value.columns for c in coord_cols):
                block = value
                break
        if block is None:
            # Fall back to the last block (RELION convention for particles).
            block = list(star.values())[-1]
        star = block

    # 1-based original row numbers, preserved across filtering for stable patch_id.
    star = star.reset_index(drop=True)
    line_ids = star.index.to_numpy() + 1

    # Optionally keep only rows belonging to this tomogram (combined STAR files).
    tomo_col = pa_config.get("star_tomo_column")
    if tomo_col and tomo_name:
        if tomo_col not in star.columns:
            print(f"  WARNING: star_tomo_column '{tomo_col}' not found in STAR; "
                  f"using all {len(star)} particles.")
        else:
            mask = np.array([tomo_name_matches(v, tomo_name) for v in star[tomo_col]])
            n_before = len(star)
            star = star[mask]
            line_ids = line_ids[mask]
            print(f"  Filtered STAR by {tomo_col} matching tomogram '{tomo_name}' "
                  f"(basename match): {len(star)}/{n_before} particles")

    coords = star[coord_cols].to_numpy(dtype=float)
    if pa_config.get("star_coords_in_pixels", True):
        px_col = pa_config.get("star_pixelsize_column", "rlnPixelSize")
        pixel_size = star[px_col].to_numpy(dtype=float)  # Angstrom/pixel
        coords = coords * pixel_size[:, None]
    # coords are now in Angstrom; convert to nm unless the surfaces are in Angstrom.
    if not angstroms:
        coords = coords / 10.0
    star = star.reset_index(drop=True)
    return star, coords, line_ids


# ---------------------------------------------------------------------------
# Single-graph driver
# ---------------------------------------------------------------------------


def generate_patches_single(graph_file, star_file, pa_config, radius_hit,
                            angstroms, output_dir, label, generate_random, seed,
                            tomo_name=None):
    """Generate patches for one membrane graph + STAR file.

    tomo_name, with `star_tomo_column` in the config, restricts a combined STAR
    file to the rows belonging to this tomogram.
    """
    from pycurv import TriangleGraph, io
    from graph_tool import load_graph

    np.bool = bool  # pycurv/graph-tool compatibility shim (deprecated numpy alias)

    patch_radius = pa_config.get("patch_radius", 12)
    particle_max_distance = pa_config.get("particle_max_distance", None)
    random_min_distance = pa_config.get("random_min_distance", patch_radius)
    annotate_star = pa_config.get("annotate_star", True)
    # Measurement names are prefixed (default "ribo") so several kinds of patches
    # (e.g. ribosome vs ATP synthase) can be measured into the SAME surface without
    # colliding — each run adds its own <prefix>_* properties in place.
    prefix = pa_config.get("measurement_prefix", "ribo")

    print(f"Processing graph: {graph_file}")
    print(f"  STAR file: {star_file}")
    tg = TriangleGraph()
    tg.graph = load_graph(graph_file)
    triangle_xyz = tg.graph.vp.xyz.get_2d_array([0, 1, 2]).transpose()
    n_triangles = triangle_xyz.shape[0]
    print(f"  {n_triangles} triangles")

    # If per-triangle membrane thickness is present (from measure_thickness), also
    # emit <prefix>_headgroup_distance = (protein/mesh distance) - thickness/2: the
    # distance to the true membrane edge (lipid headgroups) rather than the bilayer
    # midplane that the surface sits on.
    have_thickness = "thickness" in tg.graph.vp
    thickness = tg.graph.vp["thickness"].get_array() if have_thickness else None
    if have_thickness:
        print(f"  Thickness present -> adding {prefix}_headgroup_distance")

    star, particle_xyz, line_ids = load_particle_coordinates(
        star_file, pa_config, angstroms, tomo_name=tomo_name)
    n_particles = particle_xyz.shape[0]
    print(f"  {n_particles} particles")
    if n_particles == 0:
        print("  No particles to process for this graph; skipping.")
        return

    # Nearest triangle for every particle (for STAR annotation + patch centers).
    min_d, min_i = nearest_triangle(triangle_xyz, particle_xyz)

    # Patch id = 1-based row number in the original STAR (0 reserved for "no patch").
    line_ids = line_ids.astype(np.int64)

    # Annotate the STAR with mesh distance and nearest-triangle id (all particles).
    if annotate_star:
        star = star.copy()
        star["patch_id"] = line_ids
        star["mesh_distance"] = min_d
        star["mesh_neighbor_id"] = min_i
        if have_thickness:
            # Distance from each particle to the true membrane edge (headgroups),
            # measured at its nearest (patch-center) triangle; see
            # effective_thickness for the center->patch-mean->NaN thickness fallback.
            star[f"{prefix}_headgroup_distance"] = _particle_headgroup_distances(
                triangle_xyz, thickness, min_d, min_i, patch_radius)
        star_stem = os.path.splitext(os.path.basename(star_file))[0]
        annotated_path = os.path.join(output_dir, f"{star_stem}_{label}_meshannotated.star")
        _write_star(star, annotated_path)
        print(f"  Wrote annotated STAR: {annotated_path}")

    # Patch centers: optionally drop particles whose membrane distance is too large.
    if particle_max_distance is not None:
        keep = min_d <= particle_max_distance
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(f"  Dropped {n_dropped} particle(s) farther than "
                  f"{particle_max_distance} nm from the membrane")
    else:
        keep = np.ones(n_particles, dtype=bool)

    center_indices = min_i[keep]
    center_ids = line_ids[keep]
    center_particle_xyz = particle_xyz[keep]
    print(f"  Generating {len(center_indices)} patch(es)")

    real = assign_patches(triangle_xyz, center_indices, center_ids, patch_radius,
                          particle_xyz=center_particle_xyz)
    _set_int_vp(tg, f"{prefix}_patch_number", real["number"])
    _set_int_vp(tg, f"{prefix}_patch_center", real["center"])
    _set_float_vp(tg, f"{prefix}_patch_center_distance", real["center_distance"])
    _set_float_vp(tg, f"{prefix}_protein_distance", real["protein_distance"])
    # Note: the headgroup distance is a per-patch quantity measured at the central
    # triangle (protein distance - effective_thickness/2), not a per-triangle field.
    # It is emitted per-particle in the annotated STAR (above) and per-patch by
    # `morphometrics patch_statistics` (from protein_distance + thickness), so it is
    # not stored as a per-triangle vertex property here.

    if generate_random:
        rng = np.random.default_rng(seed)
        rand_centers = choose_random_centers(
            triangle_xyz, len(center_indices), random_min_distance, rng)
        if len(rand_centers) < len(center_indices):
            print(f"  WARNING: only placed {len(rand_centers)} of "
                  f"{len(center_indices)} random patches given the "
                  f"{random_min_distance} nm spacing constraint")
        # Random patches reuse the paired real patch ids so number i <-> random i.
        rand_ids = center_ids[:len(rand_centers)]
        rand = assign_patches(triangle_xyz, rand_centers, rand_ids, patch_radius)
        _set_int_vp(tg, f"{prefix}_random_patch_number", rand["number"])
        _set_int_vp(tg, f"{prefix}_random_patch_center", rand["center"])
        _set_float_vp(tg, f"{prefix}_random_patch_center_distance", rand["center_distance"])

    # Edit the input graph/surface/CSV IN PLACE: write the patch properties back
    # into the existing AVV files rather than creating a separate *_patches set.
    # This keeps a single enriched surface per membrane and lets repeated runs with
    # different prefixes accumulate multiple patch types on it.
    base_path = os.path.splitext(graph_file)[0]
    tg.graph.save(f"{base_path}.gt")
    surf = tg.graph_to_triangle_poly()
    io.save_vtp(surf, f"{base_path}.vtp")
    from .intradistance_verticality import export_csv
    export_csv(tg, f"{base_path}.csv")
    print(f"  Updated in place ({prefix}_patch*): {base_path}.gt / .vtp / .csv")


def _write_star(df, path):
    """Write a STAR file, falling back to CSV-like if starfile.write is absent."""
    import starfile
    starfile.write(df, path, overwrite=True)


def _set_int_vp(tg, name, array):
    vp = tg.graph.new_vertex_property("int")
    vp.a = array.astype(np.int64)
    tg.graph.vp[name] = vp


def _set_float_vp(tg, name, array):
    vp = tg.graph.new_vertex_property("double")
    vp.a = array.astype(float)
    tg.graph.vp[name] = vp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("configfile", type=click.Path(exists=True))
@click.option("--graph", "graph_file", type=click.Path(exists=True), default=None,
              help="Single membrane .gt graph to process (overrides batch mode).")
@click.option("--star", "star_file", type=click.Path(exists=True), default=None,
              help="STAR file of particle coordinates (required with --graph).")
@click.option("--label", default=None,
              help="Membrane label (e.g. IMM). Defaults to patch_analysis.membrane_label.")
@click.option("--output-dir", "output_dir", default=None,
              help="Directory for the annotated STAR (defaults to work_dir). The "
                   "membrane .gt/.vtp/.csv are edited in place, not written here.")
@click.option("--no-random", is_flag=True, default=False,
              help="Skip generation of random control patches.")
@click.option("--seed", type=int, default=None,
              help="Random seed (defaults to patch_analysis.random_seed).")
@click.option("--star-tomo-column", "star_tomo_column", default=None,
              help="STAR column holding the tomogram/micrograph name; if set, a "
                   "combined STAR is filtered to rows matching the tomogram "
                   "(overrides patch_analysis.star_tomo_column).")
@click.option("--tomo-name", "tomo_name", default=None,
              help="With --graph: the tomogram name to match in --star-tomo-column "
                   "(default: inferred from the graph filename).")
def generate_patches_cli(configfile, graph_file, star_file, label, output_dir,
                         no_random, seed, star_tomo_column, tomo_name):
    """Generate protein-centered membrane patches from a STAR file.

    CONFIGFILE: path to config.yml.

    By default every particle in the STAR is matched against the membrane, so the
    STAR must hold only that tomogram's particles. For a combined multi-tomogram
    STAR, set star_tomo_column (config or --star-tomo-column) to filter by tomogram.
    """
    config = load_config(configfile, require=("work_dir",))

    pa_config = config.get("patch_analysis", {})
    if star_tomo_column is not None:
        pa_config = {**pa_config, "star_tomo_column": star_tomo_column}
    work_dir = config.get("work_dir", config.get("seg_dir", "./"))
    if not work_dir.endswith("/"):
        work_dir += "/"
    if output_dir is None:
        output_dir = work_dir
    elif not output_dir.endswith("/"):
        output_dir += "/"
    os.makedirs(output_dir, exist_ok=True)

    radius_hit = config.get("curvature_measurements", {}).get("radius_hit", 9)
    angstroms = config.get("surface_generation", {}).get("angstroms", False)
    label = label or pa_config.get("membrane_label", "IMM")
    generate_random = (not no_random) and pa_config.get("generate_random", True)
    seed = seed if seed is not None else pa_config.get("random_seed", 0)

    print("Patch generation settings:")
    print(f"  Work directory: {work_dir}")
    print(f"  Output directory (annotated STAR): {output_dir}")
    print(f"  Membrane .gt/.vtp/.csv: edited in place")
    print(f"  Measurement prefix: {pa_config.get('measurement_prefix', 'ribo')}")
    print(f"  Membrane label: {label}")
    print(f"  Patch radius: {pa_config.get('patch_radius', 12)} nm")
    mdist = pa_config.get("particle_max_distance", None)
    print(f"  Particle max distance: {'disabled' if mdist is None else f'{mdist} nm'}")
    print(f"  Random control patches: {generate_random}")

    # Single-file mode.
    if graph_file is not None:
        if star_file is None:
            raise click.UsageError("--star is required when --graph is given.")
        # Infer the tomogram name from the graph filename if not given (used only
        # when star_tomo_column is set, to filter a combined STAR).
        single_tomo = tomo_name
        if single_tomo is None:
            single_tomo = os.path.basename(graph_file).split(f"_{label}")[0]
        generate_patches_single(graph_file, star_file, pa_config, radius_hit,
                                angstroms, output_dir, label, generate_random, seed,
                                tomo_name=single_tomo)
        return

    if star_file is not None:
        raise click.UsageError("--graph is required when --star is given.")

    # Batch mode: find each membrane graph and its matching STAR file.
    star_dir = pa_config.get("star_dir", work_dir)
    if not star_dir.endswith("/"):
        star_dir += "/"
    star_pattern = pa_config.get("star_pattern", "{tomo}.star")

    graphs = sorted(glob(f"{work_dir}*_{label}.AVV_rh{radius_hit}.gt"))
    if not graphs:
        print(f"No membrane graphs matching *_{label}.AVV_rh{radius_hit}.gt in {work_dir}")
        return
    print(f"Found {len(graphs)} membrane graph(s) to process")

    for gpath in graphs:
        base = os.path.basename(gpath)
        tomo = base.split(f"_{label}.AVV_rh{radius_hit}.gt")[0]
        spath = os.path.join(star_dir, star_pattern.format(tomo=tomo))
        if not os.path.exists(spath):
            print(f"  SKIP {tomo}: no STAR file at {spath}")
            continue
        generate_patches_single(gpath, spath, pa_config, radius_hit, angstroms,
                                output_dir, label, generate_random, seed,
                                tomo_name=tomo)


if __name__ == "__main__":
    generate_patches_cli()
