"""Old (archived utils / original notebook) vs new package: identical outputs.

These need the gitignored ``wildfire_stats.old`` archive and ``dev/original_nbs``;
they skip when either is missing.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import (
    CORE_BIOMES,
    fake_overall_results,
    fake_timevary_results,
    make_fires,
    nested_function_source,
    notebook_cell_source,
    old_module_path,
)

TA = pytest.importorskip("wildfire_stats.old.temporal_analysis")
LS = pytest.importorskip("wildfire_stats.old.landcover_segmentation")
DC = pytest.importorskip("wildfire_stats.old.dataset_comparison")

from wildfire_stats.data import classification as dcls  # noqa: E402
from wildfire_stats.data import io as dio  # noqa: E402
from wildfire_stats.fitting import fits  # noqa: E402
from wildfire_stats.fitting import powerlaw as wfpl  # noqa: E402
from wildfire_stats.fitting.stats_helpers import ccdf, kuiper, logpdf  # noqa: E402
from wildfire_stats.reporting import summaries as rsum  # noqa: E402
from wildfire_stats.reporting import tables as rtab  # noqa: E402


def assert_nested_equal(a, b):
    assert type(a) is type(b) or (isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray)))
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys())
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, np.ndarray)):
        np.testing.assert_array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b)
    else:
        assert a == b


# ------------------------------------------------------------------ stats helpers

@pytest.mark.parametrize("name,args", [
    ("ccdf_exponential", (0.3,)),
    ("ccdf_power_law", (1.8,)),
    ("ccdf_truncated_power_law", (1.6, 0.05)),
    ("ccdf_stretched_exponential", (0.2, 0.7)),
    ("ccdf_weibull", (0.8, 6.0)),
    ("ccdf_lognormal", (1.5, 1.0)),
    ("ccdf_genpareto", (0.3, 5.0)),
])
def test_ccdf_functions(name, args):
    x = np.logspace(np.log10(4), 3, 200)
    np.testing.assert_array_equal(getattr(ccdf, name)(x, *args, 4), getattr(TA, name)(x, *args, 4))


@pytest.mark.parametrize("name,params", [
    ("logpdf_lognormal", (1.5, 1.0)),
    ("logpdf_trunc_powerlaw", (1.6, 0.05)),
    ("logpdf_genpareto", (0.3, 5.0)),
    ("logpdf_genpareto", (-0.5, 2.0)),
    ("logpdf_weibull", (0.8, 6.0)),
    ("logpdf_stretched_exp", (0.2, 0.7)),
])
@pytest.mark.parametrize("occurrence", [0, 1])  # analyze_time_varying_mle, then _refined
def test_logpdf_functions(name, params, occurrence):
    src = old_module_path("temporal_analysis").read_text()
    ns = {}
    exec("import numpy as np\nfrom scipy.special import gamma, gammaincc\n"
         + nested_function_source(src, name, occurrence), ns)
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(1.8, 1.1, 300)) + 4.0  # MLE passes tail data only (x >= xmin)
    t = rng.normal(0, 1, 300)
    p1 = params[0] + 0.05 * t
    p2 = params[1] + 0.05 * t
    try:
        expected = ns[name](x, p1, p2, 4)
    except ValueError as e:
        # e.g. GPD with xi < 0: array params fail to broadcast once a point leaves the support
        with pytest.raises(ValueError, match=str(e).split(" with shapes")[0]):
            getattr(logpdf, name)(x, p1, p2, 4)
        return
    np.testing.assert_array_equal(getattr(logpdf, name)(x, p1, p2, 4), expected)


def test_kuiper():
    a = make_fires(seed=1)[:500]
    b = make_fires(seed=2)[:400]
    alphas = np.linspace(0, 30, 50)
    _, min_alpha, min_k = kuiper.kuiper_scan(a, b, alphas)
    assert (min_alpha, min_k) == DC.plot_alpha_vs_kuiper_with_min_cdf(a, b, alphas)
    assert kuiper.kuiper_statistic(a.area_km2, b.area_km2) == DC.kuiper_statistic(a.area_km2, b.area_km2)


# ------------------------------------------------------------------ static fits / grouping

def test_fit_static_by_group_matches_notebook_loop(fires):
    df = fires[fires.modis_class_static.isin(CORE_BIOMES[:2])]
    new = fits.fit_static_by_group(df, "modis_class_static", R=3, years=(1990, 2016))

    old = {}
    sub = df[(df["year"] >= 1990) & (df["year"] <= 2016)]
    for modis_cat, subset in sub.groupby("modis_class_static"):
        data = subset["area_km2"].values
        params = wfpl.summarize_parameters_bootstrap(data, R=3, xmin=4, random_state=42)
        R, best = wfpl.likelihood_matrix_and_best(data, xmin=4)
        old[modis_cat] = {"params": params, "likelihood_matrix": R, "best_fit": best}

    assert list(new) == list(old)
    for k in old:
        pd.testing.assert_frame_equal(new[k]["params"], old[k]["params"])
        pd.testing.assert_frame_equal(new[k]["likelihood_matrix"], old[k]["likelihood_matrix"])
        assert new[k]["best_fit"] == old[k]["best_fit"]


def _nb5_refined_groups(mtbs_classified):
    """Run the grouping half of original notebook 5 cell 33."""
    src = notebook_cell_source("5_*.ipynb", 33)
    src = src[: src.index("# === Step 4")]
    ns = {"pd": pd, "mtbs_classified": mtbs_classified}
    exec(src, ns)
    return ns["df_refined"]


def test_build_refined_groups(fires):
    new = dcls.build_refined_groups(fires, verbose=False)
    old = _nb5_refined_groups(fires)
    pd.testing.assert_frame_equal(new, old)


# ------------------------------------------------------------------ time-varying MLE

DISTS = ("lognormal", "generalized_pareto", "truncated_power_law")


def _small_fires():
    return make_fires(n_per=120, seed=4)


def test_time_varying_mle_original():
    df = _small_fires()
    overall = fake_overall_results(["Grasslands"], DISTS)
    kw = dict(xmin=4, R_boot=1, min_total=50, verbose=False)
    np.random.seed(0)
    old = TA.analyze_time_varying_mle(df, overall, **kw)
    np.random.seed(0)
    new = fits.analyze_time_varying_mle(df, overall, **kw)
    assert set(old["Grasslands"]) == set(DISTS)
    assert_nested_equal(new, old)


def test_time_varying_mle_refined():
    df = _small_fires()
    overall = fake_overall_results(["Open shrublands W of 130W"], DISTS[:2])
    kw = dict(xmin=4, R_boot=1, min_total=20, verbose=False)
    np.random.seed(0)
    old = TA.analyze_time_varying_mle_refined(df, overall, **kw)
    np.random.seed(0)
    new = fits.analyze_time_varying_mle(
        dcls.build_refined_groups(df, verbose=False), overall, group_col="biome_refined", **kw
    )
    assert list(old) == ["Open shrublands W of 130W"]
    assert_nested_equal(new, old)


def test_time_varying_mle_evtname():
    src = notebook_cell_source("7_*.ipynb", 12)
    src = src[: src.index("timevary_results_evt = ")]
    ns = {}
    exec("import numpy as np\nimport pandas as pd\nimport geopandas as gpd\n" + src, ns)

    df = _small_fires()
    df["EVT_NAME"] = np.where(df["LONGITUDE"] < -110, "West shrub", "East grass")
    overall = fake_overall_results(["West shrub"], DISTS[:2])
    kw = dict(group_col="EVT_NAME", xmin=4, R_boot=1, min_total=50, verbose=False)
    np.random.seed(0)
    old = ns["analyze_time_varying_mle_evtname"](df, overall, **kw)
    np.random.seed(0)
    new = fits.analyze_time_varying_mle(df, overall, restrict_to_fitted=True, **kw)
    assert list(old) == ["West shrub"]
    assert_nested_equal(new, old)


# ------------------------------------------------------------------ summaries

GROUPS = CORE_BIOMES + ["Croplands"]


@pytest.fixture
def mode_tables():
    tv = fake_timevary_results(GROUPS)
    return tv, [TA.summarize_timevary_results_mode(tv, mode=m) for m in ("both", "p1_only", "p2_only")]


def test_summarize_timevary_results_mode(mode_tables):
    tv, old = mode_tables
    for new_df, old_df in zip(rsum.summarize_timevary_modes(tv), old):
        pd.testing.assert_frame_equal(new_df, old_df)
    # sanity: the fixture produces both significant and non-significant slopes
    assert old[0]["p1_slope_sig"].nunique() == 2


@pytest.mark.parametrize("old_name,kw", [
    ("summarize_ecoregion_fits", dict(include="primary")),
    ("summarize_other_fits", dict(include="other")),
    ("summarize_all_fits", dict(include="all", order="primary_first")),
    ("summarize_all_fits_general", dict(include="all", order="delta_ll")),
])
def test_summarize_fits(mode_tables, old_name, kw):
    _, (df_both, df_p1, df_p2) = mode_tables
    overall = fake_overall_results(GROUPS)
    # vary Δloglik so the delta_ll ordering is exercised
    for i, g in enumerate(GROUPS):
        overall[g]["likelihood_matrix"]["best"] = [0.0, 1.5 + i % 3, 0.7]
    old = getattr(TA, old_name)(df_both, df_p1, df_p2, overall)
    new = rsum.summarize_fits(df_both, df_p1, df_p2, overall, **kw)
    assert (new["Δp1 ± se"] != "").any()
    pd.testing.assert_frame_equal(new, old)


# ------------------------------------------------------------------ Poisson tail trend

def test_poisson_highres(fires):
    old = TA.fit_poisson_tail_trend_by_biome_highres(fires)
    new = fits.fit_poisson_tail_trend(fires, valid_groups=dcls.POISSON_BIOMES)
    pd.testing.assert_frame_equal(new, old)


def test_poisson_refined(fires):
    old = TA.fit_poisson_tail_trend_by_biome_refined(fires)
    new = fits.fit_poisson_tail_trend(
        dcls.build_refined_groups(fires, verbose=False),
        group_col="biome_refined",
        label_col="biome_refined",
    )
    pd.testing.assert_frame_equal(new, old)


# ------------------------------------------------------------------ classification

def test_classify_with_modis(modis_rasters):
    by_year, gdf = modis_rasters
    new = dcls.classify_with_modis(gdf, "year", by_year)
    pd.testing.assert_frame_equal(new, TA.classify_with_modis(gdf, "year", by_year, dcls.MODIS_TO_GFA))
    new_ls = dcls.classify_with_modis(gdf, "year", by_year, out_col="modis_class")
    pd.testing.assert_frame_equal(new_ls, LS.classify_with_modis(gdf, "year", by_year, dcls.MODIS_TO_GFA))
    assert (new["modis_class_timevary"] != "Unknown").sum() > 5


def test_classify_with_static(modis_rasters, tmp_path):
    from wildfire_stats.data import rasters

    by_year, gdf = modis_rasters
    tiles = rasters.build_static_modis_tiles(by_year, out_dir=str(tmp_path / "static_new"))
    old_tiles = TA.build_static_modis_tiles(by_year, out_dir=str(tmp_path / "static_old"))
    import rasterio

    for t in tiles:
        with rasterio.open(tiles[t]) as a, rasterio.open(old_tiles[t]) as b:
            np.testing.assert_array_equal(a.read(1), b.read(1))
    new = dcls.classify_with_static(gdf, tiles)
    pd.testing.assert_frame_equal(new, TA.classify_with_static(gdf, old_tiles, dcls.MODIS_TO_GFA))


def test_build_gfa_modis_validation(modis_rasters):
    import geopandas as gpd
    import rasterio
    import rasterio.mask
    from collections import Counter

    by_year, gdf = modis_rasters
    gfa_global = gdf.drop(columns="year")
    src = notebook_cell_source("3_*.ipynb", 5)
    src = src[src.index("results = []"): src.index("gfa_classified = ")]
    ns = dict(gpd=gpd, rasterio=rasterio, Counter=Counter, gfa_global=gfa_global,
              modis_by_year=by_year, modis_to_gfa=dcls.MODIS_TO_GFA)
    exec(src, ns)
    new = dcls.build_gfa_modis_validation(gfa_global, by_year)
    pd.testing.assert_frame_equal(new, ns["results_df"])


@pytest.mark.parametrize("old_name,kind,cols", [
    ("evaluate_modis_against_gfa", "sensitivity", ("gfa_class", "modis_class")),
    ("evaluate_modis_reliability", "reliability", ("modis_class", "gfa_class")),
])
def test_classification_metrics(old_name, kind, cols):
    from sklearn.metrics import confusion_matrix

    ns = {"np": np, "pd": pd, "confusion_matrix": confusion_matrix}
    exec(nested_function_source(notebook_cell_source("3_*.ipynb", 6), old_name), ns)
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "gfa_class": rng.choice(CORE_BIOMES + ["Croplands"], 400),
        "modis_class": rng.choice(CORE_BIOMES + ["Croplands"], 400),
    })
    new = rtab.classification_metrics(df, ref_col=cols[0], pred_col=cols[1], kind=kind)
    pd.testing.assert_frame_equal(new, ns[old_name](df))


def test_load_shapefile_same_as_old(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    g = gpd.GeoDataFrame({"YEAR": [2000, 2001]}, geometry=[box(-100, 40, -99, 41), box(-90, 35, -89.5, 35.5)])
    p = tmp_path / "f.shp"
    g.to_file(p)
    pd.testing.assert_frame_equal(
        dio.load_mtbs(p, area_col="area_km2"), TA.load_shapefile(str(p), "EPSG:4269", area_col="area_km2")
    )
