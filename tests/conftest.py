"""Shared fixtures: synthetic fires, tiny rasters, and an isolated config dir."""

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import box  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
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


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Copy the tracked config yamls and write a paths.yaml pointing into tmp_path."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    for f in (REPO / "config").glob("*.yaml"):
        if f.name != "paths.yaml":
            shutil.copy(f, cfg / f.name)
    (cfg / "paths.yaml").write_text(
        f"DATA_HEAD: {tmp_path / 'data'}\n"
        "MODIS_SUBDIR: modis/\nMTBS_SUBDIR: mtbs/\nGFA_SUBDIR: gfa/\nIDAHO_SUBDIR: idaho/\n"
        "EVT_SUBDIR: evt/\nLOGGED_OUTPUT_SUBDIR: logged_output/\nOUTPUT_SUBDIR: output/\n"
        f"FIGS_DIR: {tmp_path / 'figs'}\n"
    )
    monkeypatch.setenv("WILDFIRE_STATS_CONFIG_DIR", str(cfg))
    return cfg


def make_fires(n_per=300, seed=0, biomes=CORE_BIOMES):
    """Lognormal fire sizes (some below 4 km²), years 1984–2024, lon/lat per biome."""
    rng = np.random.default_rng(seed)
    frames = []
    for b in biomes:
        frames.append(pd.DataFrame({
            "modis_class_static": b,
            "area_km2": np.exp(rng.normal(1.8, 1.1, n_per)) + 0.5,
            "year": rng.integers(1984, 2025, n_per),
            "LONGITUDE": rng.uniform(-150, -70, n_per),
            "LATITUDE": rng.uniform(25, 65, n_per),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def fires():
    return make_fires()


def fake_overall_results(groups, dists=("lognormal", "generalized_pareto", "truncated_power_law")):
    """Static-fit dicts shaped like fit_static_by_group output (no real fitting)."""
    static = {
        "lognormal": (1.5, 1.0),
        "generalized_pareto": (0.3, 5.0),
        "truncated_power_law": (1.6, 0.01),
        "weibull": (0.8, 6.0),
    }
    out = {}
    for g in groups:
        params = pd.DataFrame(
            {
                "p1": [static[d][0] for d in dists],
                "p1_se": 0.1,
                "p2": [static[d][1] for d in dists],
                "p2_se": 0.1,
                "reduces_to": None,
            },
            index=list(dists),
        )
        llm = pd.DataFrame({"best": [0.0] * len(dists)}, index=list(dists))
        out[g] = {"params": params, "likelihood_matrix": llm, "best_fit": dists[0]}
    return out


def fake_timevary_results(groups, seed=1):
    """Nested results[group][dist][mode] with a mix of significant / null slopes."""
    rng = np.random.default_rng(seed)
    res = {}
    for g in groups:
        res[g] = {}
        for d in ("lognormal", "generalized_pareto", "truncated_power_law"):
            res[g][d] = {}
            for mode, k in (("both", 4), ("p1_only", 3), ("p2_only", 3)):
                coeffs = rng.normal(0, 1, k)
                ses = np.abs(rng.normal(0, 0.5, k))
                res[g][d][mode] = {"coeffs": coeffs, "ses": ses, "loglik": -100.0, "n": 500}
    return res


@pytest.fixture
def modis_rasters(tmp_path):
    """Two tiny EPSG:4326 MODIS-like class rasters (one tile per year) plus fire polygons."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(3)
    d = tmp_path / "modis"
    d.mkdir()
    by_year = {}
    for year in (2005, 2006):
        arr = rng.choice([1, 4, 7, 10, 12, 17, 255], size=(40, 40), p=[.2, .2, .2, .2, .1, .05, .05]).astype("uint8")
        path = d / f"MCD12Q1.A{year}001.h09v05.061_LC_Type1.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=40, width=40, count=1, dtype="uint8",
            crs="EPSG:4326", transform=from_origin(-110.0, 40.0, 0.25, 0.25), nodata=255,
        ) as dst:
            dst.write(arr, 1)
        by_year[year] = [str(path)]

    polys = []
    for i in range(30):
        x0 = rng.uniform(-110.5, -101.5)
        y0 = rng.uniform(29.5, 39.0)
        polys.append(box(x0, y0, x0 + rng.uniform(0.2, 1.5), y0 + rng.uniform(0.2, 1.5)))
    gdf = gpd.GeoDataFrame(
        {
            "year": rng.choice([2005, 2006, 2010], size=30),
            "fire_ID": np.arange(30),
            "landcover_s": rng.choice(CORE_BIOMES, size=30),
        },
        geometry=polys,
        crs="EPSG:4326",
    )
    gdf["YEAR"] = gdf["year"]
    return by_year, gdf


def notebook_cell_source(nb_glob, cell):
    """Source of one cell of an original notebook in dev/original_nbs."""
    path = next((REPO / "dev" / "original_nbs").glob(nb_glob))
    nb = json.loads(path.read_text())
    return "".join(nb["cells"][cell]["source"])


def nested_function_source(module_source, name, occurrence=0):
    """Dedented source of a (possibly nested) function definition."""
    import ast
    import textwrap

    tree = ast.parse(module_source)
    nodes = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name),
        key=lambda n: n.lineno,
    )
    n = nodes[occurrence]
    lines = module_source.splitlines()
    return textwrap.dedent("\n".join(lines[n.lineno - 1 : n.end_lineno]))


def old_module_path(name):
    return REPO / "src" / "wildfire_stats" / "old" / f"{name}.py"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


os.environ.setdefault("MPLBACKEND", "Agg")
