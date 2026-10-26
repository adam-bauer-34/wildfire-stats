"""Build and index land-cover rasters (MODIS MCD12Q1, LANDFIRE EVT)."""

import glob
import os
import re
import subprocess
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.merge import merge
from scipy import stats

from wildfire_stats.config import load_paths, resolve


def get_lc_type1_subdataset(hdf_path):
    """Return the LC_Type1 subdataset string using gdalinfo output."""
    result = subprocess.run(
        ["gdalinfo", hdf_path],
        capture_output=True,
        text=True,
        check=True
    )
    for line in result.stdout.splitlines():
        if "LC_Type1" in line and "SUBDATASET" in line:
            return line.split("=")[1].strip()
    raise RuntimeError(f"LC_Type1 subdataset not found in {hdf_path}")


def convert_modis_hdf_to_tif(modis_dir, tif_dir):
    """Convert every MODIS ``.hdf`` in ``modis_dir`` to an LC_Type1 GeoTIFF in ``tif_dir``.

    Shells out to ``gdalinfo`` / ``gdal_translate``; existing outputs are skipped.
    """
    os.makedirs(tif_dir, exist_ok=True)
    for fname in os.listdir(modis_dir):
        if fname.endswith(".hdf"):
            hdf_path = os.path.join(modis_dir, fname)
            base = os.path.splitext(fname)[0]
            tif_path = os.path.join(tif_dir, f"{base}_LC_Type1.tif")

            if os.path.exists(tif_path):
                continue

            subdataset = get_lc_type1_subdataset(hdf_path)

            cmd = ["gdal_translate", subdataset, tif_path, "-of", "GTiff"]

            print("Converting:", hdf_path)
            subprocess.run(cmd, check=True)


def index_modis_by_year(tif_dir):
    """Map year -> list of MODIS GeoTIFF paths, using the ``A<YYYY>`` token in filenames."""
    modis_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith(".tif")]
    modis_by_year = {}
    for f in modis_files:
        match = re.search(r"A(\d{4})", os.path.basename(f))
        if match:
            year = int(match.group(1))
            modis_by_year.setdefault(year, []).append(f)
    return modis_by_year


# TODO(bug): pixels that are NaN in every year become 0 ("Water") after the
# uint8 cast (HANDOFF 4m).
def build_static_modis_tiles(modis_by_year: dict[int, list[str]], out_dir: str | None = None) -> dict[str, str]:
    """Build static MODIS (per tile) rasters by pixelwise mode across years.

    Args:
        modis_by_year: Mapping from year → list of MODIS raster file paths.
        out_dir: Output directory for static rasters (default ``<logged_output>/static_modis_tiles``).

    Returns:
        Dict mapping tile_id → written GeoTIFF path.
    """
    if out_dir is None:
        out_dir = os.path.join(load_paths().logged_output, "static_modis_tiles")
    default_nodata = resolve(None, "modis_nodata")
    water_class = resolve(None, "modis_water_class")
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Regex to extract MODIS tile IDs (e.g., h10v04) from filenames
    tile_pattern = re.compile(r"h\d{2}v\d{2}")

    # collect rasters by MODIS tile id
    # Build a mapping from tile_id to all rasters (across years) covering that tile
    tile_files: dict[str, list[str]] = {}
    for year, files in modis_by_year.items():
        for f in files:
            m = tile_pattern.search(os.path.basename(f))
            if not m:
                continue
            tile_id = m.group()
            tile_files.setdefault(tile_id, []).append(f)

    # Dict to store output GeoTIFF paths per tile
    out_paths: dict[str, str] = {}

    # For each tile, compute pixelwise mode across its raster stack to form a static map
    for tile_id, files in tile_files.items():
        arrays: list[np.ndarray] = []
        ref_profile: dict | None = None
        nodata = default_nodata

        # Read raster bands for this tile and store them in a stack
        for f in files:
            with rasterio.open(f) as src:
                # Set the reference profile from the first raster
                if ref_profile is None:
                    ref_profile = src.profile.copy()
                    nodata = ref_profile.get("nodata", default_nodata)
                arr = src.read(1).astype(float)
                arrays.append(arr)

        # stack and mask nodata + class 17 as NaN (per your convention)
        # Stack arrays along a time axis and mask out nodata and a specific class (17)
        stack = np.stack(arrays, axis=0)
        stack = np.where((stack == nodata) | (stack == water_class), np.nan, stack)

        # pixelwise mode across time
        # Use SciPy's mode to get the most frequent land cover class at each pixel
        mode_map, _ = stats.mode(stack, axis=0, nan_policy="omit")
        static_map = np.squeeze(mode_map).astype(np.uint8)

        # Construct output filename for the static tile map
        out_path = os.path.join(out_dir, f"static_modis_mode_{tile_id}.tif")

        # Write the static map as a GeoTIFF with the reference metadata
        assert ref_profile is not None
        ref_profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=nodata)
        with rasterio.open(out_path, "w", **ref_profile) as dst:
            dst.write(static_map, 1)
        out_paths[tile_id] = out_path

    # Return mapping from tile_id to static GeoTIFF path
    return out_paths


def build_static_evt_region(evt_files: list[str], out_path: str) -> str:
    """Compute pixelwise mode across a stack of EVT rasters using block windows.

    Note from the original author (notebook 7): this was never run to completion
    and may have bugs; the analysis used the 2024 rasters directly.
    """
    if not evt_files:
        raise ValueError("No EVT rasters provided for region.")

    with ExitStack() as stack:
        srcs = [stack.enter_context(rasterio.open(f)) for f in evt_files]

        ref = srcs[0]
        profile = ref.profile.copy()
        nodata = profile.get("nodata", None)

        profile.update(count=1, compress="lzw")

        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in ref.block_windows(1):
                block_stack = []
                for src in srcs:
                    arr = src.read(1, window=window).astype("float32")
                    if nodata is not None:
                        arr = np.where(arr == nodata, np.nan, arr)
                    block_stack.append(arr)

                block_stack = np.stack(block_stack, axis=0)
                all_nan = np.isnan(block_stack).all(axis=0)

                mode_block, _ = stats.mode(
                    block_stack, axis=0, nan_policy="omit", keepdims=False
                )
                mode_block = mode_block.astype(profile["dtype"])

                if nodata is not None:
                    mode_block[all_nan] = nodata

                dst.write(mode_block, 1, window=window)

    return out_path


def build_static_evt_us_auto(root_dir: str,
                             out_dir: str | None = None) -> dict[str, str]:
    """
    Automatically scans *every subfolder* in root_dir, identifies
    whether it is CONUS or Alaska (based on file name patterns),
    and creates a static EVT map (mode across years) for each region.
    Then mosaics CONUS+AK into one US raster.

    Region detection logic:
       - If file contains 'AK' or 'Alaska': goes to Alaska group
       - Otherwise: goes to CONUS group

    Args:
        root_dir: directory containing all year subfolders.
        out_dir: output directory for the static maps.

    Returns:
        { "conus": <path>, "ak": <path>, "us": <path> }

    Note from the original author (notebook 7): this was never run to completion
    and may have bugs. Region detection matches "ak" anywhere in the full path.
    """
    if out_dir is None:
        out_dir = os.path.join(load_paths().logged_output, "static_evt")
    os.makedirs(out_dir, exist_ok=True)

    conus_files = []
    ak_files = []

    # Iterate through ALL subfolders
    for item in os.listdir(root_dir):
        folder = os.path.join(root_dir, item)
        if not os.path.isdir(folder):
            continue

        tif_dir = os.path.join(folder, "Tif")
        if not os.path.isdir(tif_dir):
            continue

        tifs = glob.glob(os.path.join(tif_dir, "*.tif"))
        if not tifs:
            continue

        for tif in tifs:
            tif_lower = tif.lower()
            if "ak" in tif_lower or "alaska" in tif_lower:
                ak_files.append(tif)
            else:
                conus_files.append(tif)

    if not conus_files:
        raise RuntimeError("Could not find any CONUS tifs in the dataset.")
    if not ak_files:
        raise RuntimeError("Could not find any Alaska tifs in the dataset.")

    static_conus = os.path.join(out_dir, "static_evt_CONUS.tif")
    static_ak = os.path.join(out_dir, "static_evt_AK.tif")

    static_conus = build_static_evt_region(conus_files, static_conus)
    static_ak = build_static_evt_region(ak_files, static_ak)

    us_path = os.path.join(out_dir, "static_evt_US.tif")
    with rasterio.open(static_conus) as src_c, rasterio.open(static_ak) as src_a:
        nodata = src_c.nodata

        mosaic, transform = merge([src_c, src_a], nodata=nodata)

        profile = src_c.profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            dtype=mosaic.dtype,
        )
        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(us_path, "w", **profile) as dst:
            dst.write(mosaic, 1)

    return {"conus": static_conus, "ak": static_ak, "us": us_path}
