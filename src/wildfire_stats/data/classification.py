"""Assign land-cover classes to fire polygons and build biome groupings."""

from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio import mask as rio_mask

from wildfire_stats.config import resolve

# MODIS IGBP (MCD12Q1 LC_Type1) code -> GFA land-cover label
MODIS_TO_GFA = {
    0: "Water",
    1: "Evergreen Needleleaf forest",
    2: "Evergreen Broadleaf forest",
    3: "Deciduous Needleleaf forest",
    4: "Deciduous Broadleaf forest",
    5: "Mixed forest",
    6: "Closed shrublands",
    7: "Open shrublands",
    8: "Woody savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent wetlands",
    12: "Croplands",
    13: "Urban and built-up",
    14: "Cropland/Natural vegetation mosaic",
    15: "Snow and ice",
    16: "Barren or sparsely vegetated"
}

# Collapse rare MODIS/GFA classes (notebook 3)
MODIS_MERGE_MAP = {
    "Cropland/Natural vegetation mosaic": "Croplands",
    "Permanent wetlands": "Other",
    "Snow and ice": "Other",
    "Water": "Other",
    "Other": "Other"
}

# The 8 core biome classes used for validation, counts, and maps
CORE_BIOMES = [
    "Deciduous Broadleaf forest",
    "Evergreen Broadleaf forest",
    "Mixed forest",
    "Savannas",
    "Woody savannas",
    "Evergreen Needleleaf forest",
    "Open shrublands",
    "Grasslands",
]

# Biomes used by the high-res Poisson tail-trend scan (no savannas)
POISSON_BIOMES = [
    "Deciduous Broadleaf forest",
    "Evergreen Broadleaf forest",
    "Mixed forest",
    "Evergreen Needleleaf forest",
    "Open shrublands",
    "Grasslands",
]


def _majority_class(src, geom, code_map, nodata):
    """Most common mapped class of the raster pixels under ``geom``; None if none.

    Pixels equal to ``src.nodata`` or ``nodata`` are dropped before the vote.
    """
    try:
        out_image, _ = rasterio.mask.mask(src, [geom], crop=True)
    except ValueError:
        # Skip fires that fall outside raster extent
        return None

    data = out_image[0]
    data = data[(data != src.nodata) & (data != nodata)]
    if data.size == 0:
        return None

    mapped = [code_map.get(int(val), "Other") for val in data]
    return Counter(mapped).most_common(1)[0][0]


def classify_with_modis(fires_gdf, year_col, modis_by_year, modis_to_gfa=None, out_col="modis_class_timevary"):
    """Classify fires by majority MODIS land-cover type from that year's rasters.

    Args:
        fires_gdf (GeoDataFrame): Fire polygons with a year column.
        year_col (str): Column name specifying fire year.
        modis_by_year (dict[int, list[str]]): Mapping of year to MODIS raster paths.
        modis_to_gfa (dict[int, str]): MODIS code -> label (default ``MODIS_TO_GFA``).
        out_col (str): Output column (notebook 4 used ``"modis_class"``).

    Returns:
        GeoDataFrame: Copy of ``fires_gdf`` with ``out_col``; fires with no valid
        pixels are ``"Unknown"``. A fire spanning several tiles gets the last tile's class.
    """
    modis_to_gfa = MODIS_TO_GFA if modis_to_gfa is None else modis_to_gfa
    nodata = resolve(None, "modis_nodata")
    results = []

    # Loop over fires by year and raster file
    for year, fires in fires_gdf.groupby(fires_gdf[year_col]):
        if year not in modis_by_year:
            continue

        for modis_file in modis_by_year[year]:
            with rasterio.open(modis_file) as src:
                # Project fires to match raster CRS
                fires_proj = fires.to_crs(src.crs)
                for idx, fire in fires_proj.iterrows():
                    majority = _majority_class(src, fire.geometry, modis_to_gfa, nodata)
                    if majority is not None:
                        results.append((idx, majority))

    fires_out = fires_gdf.copy()
    fires_out[out_col] = "Unknown"
    for idx, cls in results:
        fires_out.at[idx, out_col] = cls
    return fires_out


def classify_with_static(fires_gdf, static_tile_dict, modis_to_gfa=None, out_col="modis_class_static"):
    """Classify fire polygons using static (per-tile) MODIS rasters.

    Args:
        fires_gdf: Fire perimeters (geometry must be present).
        static_tile_dict: tile_id -> static raster path (from ``build_static_modis_tiles``).
        modis_to_gfa: MODIS code -> label (default ``MODIS_TO_GFA``).
        out_col: Output column.

    Returns:
        Copy of ``fires_gdf`` with ``out_col``.
    """
    modis_to_gfa = MODIS_TO_GFA if modis_to_gfa is None else modis_to_gfa
    nodata = resolve(None, "modis_nodata")
    results = []

    # Loop over each static MODIS tile raster
    for tile_id, raster_path in static_tile_dict.items():
        with rasterio.open(raster_path) as src:
            # Reproject fire perimeters into the same CRS as the raster
            fires_proj = fires_gdf.to_crs(src.crs)
            for idx, fire in fires_proj.iterrows():
                majority = _majority_class(src, fire.geometry, modis_to_gfa, nodata)
                if majority is not None:
                    results.append((idx, majority))

    fires_out = fires_gdf.copy()
    fires_out[out_col] = "Unknown"
    for idx, cls in results:
        fires_out.at[idx, out_col] = cls
    return fires_out


def build_gfa_modis_validation(gfa, modis_by_year, modis_to_gfa=None, crs=None):
    """MODIS majority class next to GFA's own ``landcover_s`` for each fire (notebook 3).

    One row per (fire, MODIS tile) hit, with columns
    ``fire_ID, year, gfa_class, modis_class, geometry, area_km2``.
    """
    modis_to_gfa = MODIS_TO_GFA if modis_to_gfa is None else modis_to_gfa
    nodata = resolve(None, "modis_nodata")
    results = []
    for year, fires in gfa.groupby("YEAR"):
        if year not in modis_by_year:
            continue

        for modis_file in modis_by_year[year]:
            with rasterio.open(modis_file) as src:
                fires_proj = fires.to_crs(src.crs)

                for idx, fire in fires_proj.iterrows():
                    majority = _majority_class(src, fire.geometry, modis_to_gfa, nodata)
                    if majority is not None:
                        results.append({
                            "fire_ID": fire["fire_ID"],
                            "year": fire["YEAR"],
                            "gfa_class": fire.get("landcover_s", "Other"),
                            "modis_class": majority,
                            "geometry": fire.geometry
                        })

    # TODO(bug): geometries are in the MODIS raster CRS (sinusoidal metres) but
    # labelled with the GFA CRS; area_km2 is right only because sinusoidal is equal-area.
    results_df = gpd.GeoDataFrame(results, crs=resolve(crs, "gfa_crs"))
    results_df["area_km2"] = results_df.geometry.area / 1e6
    return results_df


def merge_classes(df, cols, mapping=None):
    """Copy of ``df`` with ``mapping`` (default ``MODIS_MERGE_MAP``) applied to each of ``cols``."""
    mapping = MODIS_MERGE_MAP if mapping is None else mapping
    df = df.copy()
    for col in cols:
        df[col] = df[col].replace(mapping)
    return df


def classify_with_evt(
    fires_gdf: gpd.GeoDataFrame,
    conus_raster_path: str,
    alaska_raster_path: str,
    column_name: str = "EVT_class",
) -> gpd.GeoDataFrame:
    """
    Classify each fire polygon using the most common EVT category found
    in the 2024 CONUS + 2024 Alaska EVT rasters.

    Assumptions:
      - fires_gdf.geometry has a valid CRS (e.g., EPSG:6933 for MTBS)
      - CONUS and AK rasters may have different projected CRSs
    """
    if "geometry" not in fires_gdf:
        raise ValueError("fires_gdf must contain a geometry column.")
    if fires_gdf.crs is None:
        raise ValueError("fires_gdf must have a CRS set (fires_gdf.crs).")

    fires_out = fires_gdf.copy()

    with rasterio.open(conus_raster_path) as src_conus, rasterio.open(alaska_raster_path) as src_ak:
        # Reproject fires to each raster's CRS
        fires_conus = fires_out.to_crs(src_conus.crs)
        fires_ak = fires_out.to_crs(src_ak.crs)

        evt_vals: list[float] = []

        for geom_conus, geom_ak in zip(fires_conus.geometry, fires_ak.geometry):
            # Skip empty geometries
            if geom_conus is None or geom_conus.is_empty:
                evt_vals.append(np.nan)
                continue

            got_value = False

            # -----------------------
            # Try CONUS raster first
            # -----------------------
            try:
                out_image, _ = rio_mask.mask(src_conus, [geom_conus], crop=True)
                arr = out_image[0]

                nod = src_conus.nodata
                if nod is not None:
                    arr = arr[arr != nod]

                if arr.size > 0:
                    vals, counts = np.unique(arr, return_counts=True)
                    mode_val = vals[np.argmax(counts)]
                    evt_vals.append(int(mode_val))
                    got_value = True
            except ValueError:
                # Polygon outside CONUS raster extent
                pass

            if got_value:
                continue

            # -----------------------
            # Try Alaska raster
            # -----------------------
            try:
                out_image, _ = rio_mask.mask(src_ak, [geom_ak], crop=True)
                arr = out_image[0]

                nod = src_ak.nodata
                if nod is not None:
                    arr = arr[arr != nod]

                if arr.size > 0:
                    vals, counts = np.unique(arr, return_counts=True)
                    mode_val = vals[np.argmax(counts)]
                    evt_vals.append(int(mode_val))
                    got_value = True
            except ValueError:
                # Polygon outside Alaska raster extent
                pass

            if not got_value:
                evt_vals.append(np.nan)

    fires_out[column_name] = evt_vals
    return fires_out


def _split_labels(biome, split):
    """(west, east) labels for a longitude split, e.g. 'Open shrublands W of 130W'."""
    deg = f"{abs(split):g}"
    return f"{biome} W of {deg}W", f"{biome} E of {deg}W"


def assign_refined_label(biome, lon, base_biomes=None, lon_splits=None):
    """Refined label for one fire: split biomes by longitude, keep other base biomes, else None."""
    base_biomes = resolve(base_biomes, "refined_base_biomes")
    lon_splits = resolve(lon_splits, "refined_lon_splits")
    if biome not in base_biomes:
        return None
    if biome in lon_splits:
        west, east = _split_labels(biome, lon_splits[biome])
        return west if lon < lon_splits[biome] else east
    return biome


def subset_union(df, members, new_label, col="biome_refined"):
    """Concatenate subset DataFrames and assign a new label."""
    sub = df[df[col].isin(members)].copy()
    sub[col] = new_label
    return sub


def build_refined_groups(df, biome_col="modis_class_static", lon_col="LONGITUDE",
                         out_col="biome_refined", lon_splits=None, verbose=True):
    """Build the nine (overlapping) refined biome groupings from notebook 5.

    Groups: Deciduous Broadleaf; Evergreen Broadleaf (all, E, W of the split);
    Grasslands; Open shrublands (all, W, E of the split); Grasslands + Open
    shrublands E. Union groups duplicate rows, so the result is for fitting
    only -- don't save it as the fire table.
    """
    lon_splits = resolve(lon_splits, "refined_lon_splits")
    base_biomes = resolve(None, "refined_base_biomes")

    df = df[df[biome_col].isin(base_biomes)].copy()
    if lon_col not in df.columns:
        raise ValueError(f"Expected column '{lon_col}' not found in input")

    df[out_col] = [
        assign_refined_label(b, lon, base_biomes, lon_splits)
        for b, lon in zip(df[biome_col], df[lon_col])
    ]

    ebf_w, ebf_e = _split_labels("Evergreen Broadleaf forest", lon_splits["Evergreen Broadleaf forest"])
    osh_w, osh_e = _split_labels("Open shrublands", lon_splits["Open shrublands"])

    def only(label):
        return df[df[out_col] == label].copy()

    group_frames = [
        only("Deciduous Broadleaf forest"),
        subset_union(df, [ebf_e, ebf_w], "Evergreen Broadleaf forest", out_col),
        only(ebf_e),
        only(ebf_w),
        only("Grasslands"),
        subset_union(df, [osh_e, osh_w], "Open shrublands", out_col),
        only(osh_w),
        only(osh_e),
        subset_union(df, ["Grasslands", osh_e], f"Grasslands + {osh_e}", out_col),
    ]
    df_refined = pd.concat(group_frames, ignore_index=True)

    if verbose:
        print("Final refined grouping counts:")
        print(df_refined[out_col].value_counts())
        print(f"\nTotal fires included: {len(df_refined)}")
    return df_refined
