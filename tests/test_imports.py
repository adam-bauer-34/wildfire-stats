"""Every package module imports, and no code uses sys.path hacks or the old wfpl name."""

import importlib
import pkgutil
import re

import pytest

import wildfire_stats
from conftest import REPO

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(wildfire_stats.__path__, "wildfire_stats.")
    if ".old" not in m.name
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_expected_modules_present():
    for name in [
        "wildfire_stats.config",
        "wildfire_stats.data.io",
        "wildfire_stats.data.rasters",
        "wildfire_stats.data.classification",
        "wildfire_stats.fitting.fits",
        "wildfire_stats.fitting.powerlaw",
        "wildfire_stats.fitting.stats_helpers.ccdf",
        "wildfire_stats.fitting.stats_helpers.qq",
        "wildfire_stats.fitting.stats_helpers.kuiper",
        "wildfire_stats.fitting.stats_helpers.logpdf",
        "wildfire_stats.plotting.dataset_comp",
        "wildfire_stats.plotting.distributions",
        "wildfire_stats.plotting.landcover",
        "wildfire_stats.plotting.temporal",
        "wildfire_stats.reporting.summaries",
        "wildfire_stats.reporting.tables",
    ]:
        assert name in MODULES


def _sources():
    for p in (REPO / "src").rglob("*.py"):
        yield p, p.read_text()
    for p in (REPO / "dev").glob("*.ipynb"):  # wrappers only, not dev/original_nbs
        yield p, p.read_text()


@pytest.mark.parametrize("pattern", [r"sys\.path\.(append|insert)", r"^\W*import wildfire_powerlaw", r"^\W*from utils\b"])
def test_no_legacy_imports(pattern):
    # ^\W* also catches statements inside notebook JSON strings ('    "import ...')
    hits = [str(p) for p, text in _sources() if re.search(pattern, text, flags=re.M)]
    assert not hits, hits


def test_config_resolve(isolated_config):
    from wildfire_stats import config as cfg

    assert cfg.resolve(None, "min_fire") == 4
    assert cfg.resolve(7, "min_fire") == 7
    assert isinstance(cfg.load_assumptions()["prior_weight"], float)
    de = cfg.load_dist_bounds()["de_bounds"]
    assert all(isinstance(v, float) for b in de.values() for pair in b for v in pair)
    p = cfg.load_paths()
    assert p.mtbs.name == "mtbs" and p.logged_output.name == "logged_output"
