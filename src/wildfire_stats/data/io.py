"""Load and prepare fire perimeter datasets (MTBS, Idaho HFD, GFA, LANDFIRE EVT)."""

import os
import pickle
import re

import geopandas as gpd
import pandas as pd

from wildfire_stats.config import resolve

# Rough CONUS bounding box (lon/lat) used to subset GFA in notebook 3
CONUS_BBOX = {"minx": -125.0, "maxx": -66.0, "miny": 24.0, "maxy": 50.0}

# Shapefiles truncate column names to 10 characters; this restores the ones we write
SHAPEFILE_RENAMES = {
    "modis_cl": "modis_class_timevary",
    "modis_cl_1": "modis_class_static",
    "biome_refi": "modis_class_static_grouped",
}

GFA_FOLDER_PATTERN = re.compile(r"Global_fire_atlas_V1_perimeter_\d{4}")


def load_shapefile(shp_path, projection, area_col=None, equal_area_crs=None):
    """Load a vector file, set its CRS, reproject to equal-area, and optionally compute area.

    Args:
        shp_path (str): Path to the shapefile (or other vector file) to load.
        projection (str): CRS string or EPSG code for the source projection.
        area_col (str, optional): Name of column to store polygon area in km².
            If provided and missing, it will be computed.
        equal_area_crs (str, optional): Target CRS; defaults to ``equal_area_crs`` in the yaml.

    Returns:
        GeoDataFrame: Reprojected to the equal-area CRS with optional area column.
    """
    equal_area_crs = resolve(equal_area_crs, "equal_area_crs")
    gdf = gpd.read_file(shp_path)
    gdf = gdf.set_crs(projection)
    gdf = gdf.to_crs(equal_area_crs)

    # Compute area in km² if requested and not already present
    if area_col is not None and area_col not in gdf.columns:
        gdf[area_col] = gdf.geometry.area / 1e6

    return gdf


def load_mtbs(path, area_col="AREA_SQKM", crs=None):
    """Load MTBS burn-area boundaries in the equal-area CRS.

    The file's CRS is overridden with ``mtbs_crs`` before reprojecting. With the
    default ``area_col="AREA_SQKM"`` the vendor's area column is kept; pass
    ``area_col="area_km2"`` to compute polygon areas instead (notebook 1).
    """
    return load_shapefile(path, resolve(crs, "mtbs_crs"), area_col=area_col)


def load_idaho(path, layer="Fires1950_Present", equal_area_crs=None):
    """Load the Idaho Historical Fires Database layer, reproject, and add ``area_km2``."""
    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf.to_crs(resolve(equal_area_crs, "equal_area_crs"))
    gdf["area_km2"] = gdf.geometry.area / 1e6
    return gdf


def load_gfa(folder, years=None, bbox=None, to_equal_area=True, verbose=True, crs=None, equal_area_crs=None):
    """Load and concatenate the yearly Global Fire Atlas perimeter shapefiles.

    Args:
        folder: Directory containing ``Global_fire_atlas_V1_perimeter_YYYY/`` folders.
        years: Optional inclusive ``(start, end)`` year range.
        bbox: Optional lon/lat box ``{"minx", "maxx", "miny", "maxy"}`` (e.g. ``CONUS_BBOX``).
        to_equal_area: Reproject to the equal-area CRS and add ``area_km2`` / ``perim_km``.
            If False, geometries stay in the GFA CRS and no areas are computed.
        verbose: Print each file as it is read.

    Returns:
        GeoDataFrame with a ``YEAR`` column.
    """
    crs = resolve(crs, "gfa_crs")
    equal_area_crs = resolve(equal_area_crs, "equal_area_crs")

    gfa_all_data = []
    for gfa_folder_name in os.listdir(folder):
        if not GFA_FOLDER_PATTERN.match(gfa_folder_name):
            continue
        gfa_year = int(re.search(r"\d{4}", gfa_folder_name).group())
        if years is not None and not (years[0] <= gfa_year <= years[1]):
            continue
        gfa_shp_path = os.path.join(folder, gfa_folder_name, gfa_folder_name + ".shp")
        if not os.path.exists(gfa_shp_path):
            continue

        if verbose:
            print(f"Processing {gfa_shp_path} ...")
        gfa_gdf = gpd.read_file(gfa_shp_path).set_crs(crs)
        if to_equal_area:
            gfa_gdf = gfa_gdf.to_crs(equal_area_crs)
            gfa_gdf["area_km2"] = gfa_gdf.geometry.area / 1e6
            gfa_gdf["perim_km"] = gfa_gdf.geometry.length / 1000
        gfa_gdf["YEAR"] = gfa_year
        if bbox is not None:
            gfa_gdf = gfa_gdf.cx[bbox["minx"]:bbox["maxx"], bbox["miny"]:bbox["maxy"]]
        gfa_all_data.append(gfa_gdf)

    out_crs = equal_area_crs if to_equal_area else crs
    if not gfa_all_data:
        return gpd.GeoDataFrame(columns=["fire_ID", "geometry", "YEAR", "landcover_s"], crs=out_crs)
    return gpd.GeoDataFrame(pd.concat(gfa_all_data, ignore_index=True), crs=out_crs)


def filter_years(df, year_col, years):
    """Rows with ``years[0] <= df[year_col] <= years[1]``."""
    return df[(df[year_col] >= years[0]) & (df[year_col] <= years[1])]


def overlap_with(gdf, other, dedup_col):
    """Fires in ``gdf`` that intersect any geometry in ``other``, one row per ``dedup_col``."""
    out = gpd.sjoin(gdf, other[["geometry"]], predicate="intersects")
    return out.drop_duplicates(subset=dedup_col)


def restore_truncated_columns(gdf, mapping=None):
    """Undo shapefile 10-character column truncation (missing keys are ignored)."""
    return gdf.rename(columns=SHAPEFILE_RENAMES if mapping is None else mapping)


def load_evt_attributes(path):
    """Read the LANDFIRE EVT attribute table CSV."""
    return pd.read_csv(path)


def attach_evt_attributes(gdf, attrs, code_col="EVT_class", key_col="VALUE"):
    """Left-merge EVT attributes (EVT_NAME, EVT_LF, EVT_PHYS, EVT_GP_N, ...) onto fires."""
    return gdf.merge(attrs, how="left", left_on=code_col, right_on=key_col)


def save_pickle(obj, path, verbose=True):
    """Pickle ``obj`` to ``path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if verbose:
        print(f"Saved to: {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_vector(gdf, path, verbose=True):
    """Write a GeoDataFrame with ``to_file`` (creating parent directories)."""
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    gdf.to_file(path)
    if verbose:
        print(f"Saved GeoDataFrame to {path}")
