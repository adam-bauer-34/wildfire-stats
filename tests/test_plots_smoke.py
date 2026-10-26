"""Every plotting entry point runs on synthetic data (Agg backend)."""

import numpy as np
import pandas as pd
import pytest

from conftest import CORE_BIOMES, fake_overall_results, fake_timevary_results

from wildfire_stats.data import classification as dcls
from wildfire_stats.fitting import fits
from wildfire_stats.plotting import dataset_comp as pdc
from wildfire_stats.plotting import distributions as pdist
from wildfire_stats.plotting import landcover as plc
from wildfire_stats.plotting import temporal as ptemp
from wildfire_stats.reporting import summaries as rsum
from wildfire_stats.reporting import tables as rtab


@pytest.fixture
def tv_tables():
    tv = fake_timevary_results(CORE_BIOMES)
    df_both, df_p1, df_p2 = rsum.summarize_timevary_modes(tv)
    return df_both, df_p1, df_p2


def test_dataset_comp(fires):
    a, b = fires[:800], fires[800:1500]
    pdc.plot_cdf_vs_area(a, b, labels=("A", "B"))
    alpha, k = pdc.plot_alpha_vs_kuiper_with_min_cdf(a, b, np.linspace(0, 20, 30))
    assert 0 <= alpha <= 20 and k >= 0


def test_static_fit_plots_and_table(fires):
    df = fires[fires.modis_class_static.isin(CORE_BIOMES[:2])].rename(columns={"modis_class_static": "modis_class"})
    res = fits.fit_static_by_group(df, "modis_class", R=2)
    counts = df["modis_class"].value_counts()
    results_dicts = {"MTBS": res, "GFA": res, "Idaho": res}
    counts_dicts = {"MTBS": counts, "GFA": counts, "Idaho": counts}
    pdist.plot_distribution_params(results_dicts, counts_dicts, min_count=100)
    table = rtab.build_comparison_table(results_dicts, counts_dicts, min_n=100)
    assert not table.empty
    assert pdist.plot_modis_category_ccdf(df, CORE_BIOMES[0], which=["power_law", "lognormal"]) is not None


def test_group_ccdf_cdf_qq(fires, tv_tables, tmp_path, isolated_config):
    df_both, df_p1, df_p2 = tv_tables
    overall = fake_overall_results(CORE_BIOMES)
    summary = rsum.summarize_fits(df_both, df_p1, df_p2, overall)
    # nonstationary Q–Q needs every fire >= xmin (see the TODO(bug) in the function)
    tail = fires[fires.area_km2 >= 4]
    fig = pdist.plot_group_ccdf_cdf_qq(tail, "Grasslands", col_name="modis_class_static", which=("lognormal",),
                                       timevary_summary=summary, timevary_dist="lognormal")
    assert len(fig.axes) == 4
    fig = pdist.plot_group_ccdf_cdf_qq(fires, "Grasslands", col_name="modis_class_static",
                                       which=("lognormal", "power_law"), nonstationary=False)
    assert len(fig.axes) == 3
    assert len(list((tmp_path / "figs").glob("*.png"))) == 2


def test_tail_scan(fires):
    fig, info = pdist.plot_tail_scan_ccdf_cdf_qq(
        fires, "Grasslands", col_name="modis_class_static", which=("lognormal",),
        candidate_percentiles=[50, 80], min_tail_n=20, R_boot_tail=2,
    )
    assert info["biome"] == "Grasslands" and fig is not None


def test_temporal_plots(fires, tv_tables):
    df_both, df_p1, df_p2 = tv_tables
    ptemp.plot_fire_counts_faceted(fires)
    ptemp.fire_threshold_analysis(fires, thresholds=[4, 8])
    ptemp.plot_distribution_evolution_ccdf_pdf(df_both, fires, df_p1=df_p1, df_p2=df_p2)
    counts_long, counts_wide = rsum.fires_time_series_by_ecoregion(fires, period_years=4, plot=True)
    assert set(counts_wide.columns) == set(CORE_BIOMES)
    out = fits.fit_poisson_tail_trend(dcls.build_refined_groups(fires, verbose=False),
                                      group_col="biome_refined", label_col="biome_refined")
    assert out["biome_refined"].nunique() == 9


def test_landcover_plots(fires):
    plc.plot_biome_facets(fires)
    df = fires.assign(EVT_NAME=np.where(fires.LONGITUDE < -110, "W", "E"))
    plc.plot_category_facets(df, col_name="EVT_NAME", min_fires=100)
    plc.plot_category_counts(df.assign(EVT_class=df.EVT_NAME), "EVT_class")

    rng = np.random.default_rng(0)
    v = pd.DataFrame({
        "gfa_class": rng.choice(CORE_BIOMES, 500),
        "modis_class": rng.choice(CORE_BIOMES, 500),
        "year": rng.integers(2003, 2017, 500),
    })
    confusion = pd.crosstab(v["gfa_class"], v["modis_class"])
    plc.plot_confusion_matrix(confusion)
    plc.plot_confusion_matrix(confusion, normalize=True)
    rtab.print_overall_metrics(confusion)
    metrics = plc.plot_confusion_by_period(v, {"a": (2003, 2008), "b": (2009, 2016)})
    rtab.print_period_metrics(metrics)
    assert set(metrics) == {"a", "b"}


def test_classification_agreement():
    df = pd.DataFrame({"year": [2003, 2004, 2010, 2020], "a": ["x", "y", "Unknown", "x"], "b": ["x", "z", "x", "y"]})
    rate, comp = rsum.classification_agreement(df, "a", "b")
    assert rate == 0.5 and len(comp) == 2
