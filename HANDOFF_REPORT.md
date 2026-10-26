# `wildfire-stats` — Handoff Review

Prepared 2026-07-31. Based on a static read of the repo at `4d826c6` (no code executed).

---

## 1. What this project does

One scientific question, asked three ways:

> **What probability distribution describes wildfire burned areas, does it differ by land-cover type, and is it changing over time?**

The core machinery is a vendored + extended fork of Jeff Alstott's `powerlaw` package
([utils/wildfire_powerlaw/powerlaw.py](utils/wildfire_powerlaw/powerlaw.py), 3208 lines). It MLE-fits **nine
distribution families** to fire areas above a cutoff `xmin`:

| Family | Params (p1, p2) |
|---|---|
| `power_law` | α |
| `exponential` | λ |
| `truncated_power_law` | α, λ |
| `lognormal` | μ, σ |
| `lognormal_excess` | μ, σ fit to *x − xmin* |
| `stretched_exponential` | λ, β |
| `weibull` | k, λ |
| `weibull_excess` | k, λ fit to *x − xmin* |
| `generalized_pareto` | ξ, σ |

Two public entry points do all the work:

- `wfpl.summarize_parameters_bootstrap(data, xmin, R, random_state)` → DataFrame indexed by
  distribution with `p1, p1_se, p2, p2_se, reduces_to`. SEs come from `R` nonparametric bootstrap
  refits. `reduces_to` flags nesting (e.g. TPL whose 95% CI on λ covers 0 → "this is just a power law").
- `wfpl.likelihood_matrix_and_best(data, xmin)` → `(Δloglik table, best_name)`. Best = highest total
  tail log-likelihood. Everything downstream calls a fit "good" if **n ≥ 400**, **Δloglik ≤ 2.0**, and
  **`reduces_to` is null**.

Throughout, `xmin = 4 km²` and areas are computed in **EPSG:6933** (equal-area).

### Notebook pipeline

| # | Notebook | Purpose | Writes |
|---|---|---|---|
| 1 | `1_dataset_comparison` | Compares MTBS / GFA / Idaho CDFs over their shared space-time window; minimises the Kuiper statistic over a size threshold α. **This is where `xmin = 4 km²` comes from.** | figures only |
| 2 | `2_fitting_routine` | Demo of `wfpl` on raw MTBS; sanity-check that all three datasets give similar fits | figures only |
| 3 | `3_land_cover_classification` | `gdal_translate` MODIS HDF → GeoTIFF; validates MODIS-majority land cover against GFA's own `landcover_s` attribute (confusion matrices, per-period stability) | `gfa_2003_2016_usa_classified_timevarying.shp` |
| 4 | `4_land_cover_segmentation` | Applies MODIS classification to MTBS + Idaho (which lack a land-cover field); static fits per biome per dataset | 3 `.shp`, 3 `*_static_fits_by_biome_2003_2016.pkl` |
| 5 | `5_temporal_analysis` | **The main result.** Builds a *static* (pixelwise-mode-over-years) MODIS raster so MTBS 1984–2024 can be classified outside 2003–2016; static fits per biome; **time-varying MLE** where each parameter is linear in year; Poisson tail-trend regressions; refined biome groupings split by longitude | `mtbs_classified_static.shp`, `mtbs_classified.shp`, 3 `.pkl` |
| 6 | `6_spatial_analysis` | Global GFA on 5°×5° tiles, static + time-varying. Author's own header: *"I've included this notebook for completeness but we ended up moving away from all of this."* | `gfa_spatial_*.pkl` |
| 7 | `7_evt_classifier` | Re-does the biome analysis with LANDFIRE EVT vegetation types instead of MODIS IGBP classes. Functions are defined inline in the notebook, not in `utils/` | `overall_results_evtname*.pkl` |

### The time-varying model (the novel contribution)

`analyze_time_varying_mle` in [temporal_analysis.py:447](utils/temporal_analysis.py#L447) — years are
centred and scaled to **decades** (`year_c = (year − mean)/10`), then each distribution parameter is
made affine in `year_c`:

```
lognormal:            μ(t) = a1 + b1·t      σ(t) = a2 + b2·t
truncated power law:  α(t) = 1 + exp(a1 + b1·t)   λ(t) = exp(a2 + b2·t)
```

The joint negative log-likelihood over all fires (each contributing its own year) is minimised with
`scipy.optimize.differential_evolution`, plus a weak quadratic prior (`prior_weight = 1e-3`) pulling
toward the static fit. Three **modes** are fit per distribution — `both`, `p1_only`, `p2_only` — and a
slope is called significant if its bootstrap 95% CI excludes zero.

---

## 2. Data you need to download

Nothing but code and small `.pkl` result files is in the repo. All five raw inputs must be fetched.
Sizes are order-of-magnitude estimates — verify before allocating.

| # | Dataset | Source | Coverage | Rough size | Used by |
|---|---|---|---|---|---|
| 1 | **MTBS Burned Area Boundaries** (`S_USA.MTBS_BURN_AREA_BOUNDARY.shp`) | <https://mtbs.gov/direct-download> → "Burned Areas Boundaries Dataset" | USA 1984–2024 | ~0.5–1 GB | 1, 2, 4, 5 |
| 2 | **ORNL DAAC Global Fire Atlas v1** perimeters | <https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-cms-global-fire-atlas-1642-1> | Global 2003–2016 | ~5–10 GB | 1, 2, 3, 6 |
| 3 | **Idaho State Historical Fires Database** (`HFD_HistoricFiresDatabase7_0.gdb`) | <https://giscenter.isu.edu/research/Techpg/HFD/> → "Download HFD" | 11 western states, 1950–2016 | ~0.2 GB | 1, 2, 4 |
| 4 | **MODIS MCD12Q1 v061** Land Cover Type Yearly 500 m | <https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12q1-061>, Earthdata Search, 2003-01-01 → 2016-12-31 | ~14 CONUS + ~5 AK tiles × 14 yr ≈ 250–300 HDFs | **~10–20 GB** | 3, 4, 5 |
| 5 | **LANDFIRE EVT** full-extent rasters, CONUS + Alaska, 2016–2024, plus the **EVT Attribute Data Table CSV** | <https://landfire.gov/data/FullExtentDownloads> and <https://landfire.gov/vegetation/evt> | USA | **~5–15 GB per version-year**; notebook 7 in practice only needs the **2024** CONUS + AK rasters | 7 |

Accounts required: NASA **Earthdata Login** for #2 and #4.

**Column-schema dependencies to verify on download** (these break loudly if the vendor changed the schema):

- MTBS: `YEAR`, `AREA_SQKM`, `FIRE_ID`, and **`LATITUDE` / `LONGITUDE`** — the last two are required by
  the refined longitude-split groupings in notebook 5 and the map plots. The shapefile is read with
  `.set_crs("EPSG:4269")`, i.e. the code assumes the CRS is *missing or wrong* on disk and overrides it.
  Check that this is still correct or you will silently mis-project everything.
- GFA: `fire_ID`, `landcover_s`, and folders named `Global_fire_atlas_V1_perimeter_YYYY/` each containing
  a same-named `.shp`.
- Idaho: read via layer `Fires1950_Present`, column `FIRE_YEAR`.
- MODIS: filenames must contain `A<YYYY>` and an `h##v##` tile id — both regexes are load-bearing.

**Hard-coded absolute paths to replace** (all `/Users/lukevonkapff/...`; most are flagged with a
`# REPLACE WITH...` comment, but not all):

```
notebooks/1: cells 3, 6      notebooks/5: cells 3, 4
notebooks/2: cells 2, 9      notebooks/7: cells 1, 3 (conus_2024 / alaska_2024), 6 (LF2024_EVT.csv)
notebooks/3: cells 3, 5
notebooks/4: cell 1 (modis_folder), 3, 5
notebooks/6: cell 2
```

---

## 3. How I'd reproduce it

### Step 0 — Fix the environment first (blocking)

Both dependency files are broken for this machine:

- `environment.yml` is a fully build-pinned **macOS (osx-64)** export. It contains `appnope`, Darwin build
  hashes, and `prefix: /Users/lukevonkapff/opt/anaconda3/envs/modis_env`. `conda env create -f
  environment.yml` **will not solve on Linux.**
- `requirements.txt` is a `pip freeze` full of `@ file:///Users/runner/miniforge3/...` local paths. Also
  unusable.
- **`cartopy` is missing from both**, yet [temporal_analysis.py:10](utils/temporal_analysis.py#L10) and
  [spatial_analysis.py:9](utils/spatial_analysis.py#L9) import it at module scope. Every notebook that does
  `from utils.temporal_analysis import *` fails on import without it.

Write a fresh loose environment instead:

```yaml
name: wildfire
channels: [conda-forge]
dependencies:
  - python=3.10
  - numpy pandas scipy matplotlib statsmodels scikit-learn
  - geopandas rasterio fiona pyogrio shapely
  - cartopy mpmath
  - gdal            # notebook 3 shells out to gdalinfo / gdal_translate
  - jupyterlab ipykernel
```

Then install both packages. **`utils/wildfire_powerlaw` must be installed separately** — the utils modules
import it as a *top-level* name (`import wildfire_powerlaw as wfpl`), not as `utils.wildfire_powerlaw`, so
the root `pip install -e .` alone is not enough:

```bash
pip install -e .
pip install -e utils/wildfire_powerlaw
```

Also confirm `gdalinfo` and `gdal_translate` are on `PATH` — notebook 3 calls them via `subprocess`.

### Step 1 — Understand what you are actually re-running

Three facts materially change the plan:

1. **No notebook ever loads a `.pkl` back.** There is not a single `pickle.load` or `read_pickle` in the
   repo. The README's promise that "results are saved so you only have to run them once" is not
   implemented — the `.pkl` files are write-only artifacts. The only resume path that exists is a
   *commented-out* `gpd.read_file` in notebook 5 cell 6. **Every notebook must be run end-to-end from raw
   data**, or you add the load-shortcuts yourself (recommended — see Step 3).
2. **The notebooks were not run top-to-bottom.** Saved execution counts for notebook 6 are
   `[1, 2, 3, 14, 24, 6, 7, 8, 9, 33, 34, 35, 13]` and for notebook 7 `[56, None, 2, 21, 24, ...]`. They were
   built across many sessions against an *evolving* `utils/` module. Some saved outputs were therefore
   produced by code that no longer exists (see §4.1). Notebook 5 is the cleanest (`1…31`, mostly ordered).
3. **Notebook 1 cell 3 and notebook 7 cell 1 were never executed** (`execution_count: None`). Notebook 7
   cell 1 carries the author's own note: *"THIS MAY HAVE BUGS, I WAS UNABLE TO RUN IT IN TIME SO I
   PROCEEDED WITH 2024 RASTER ALONE FOR REST OF NOTEBOOK."*

### Step 2 — Run order

Dependencies between notebooks are only via files on disk:

```
1 ──(establishes xmin = 4)──▶ (all)
3 ──gfa_2003_2016_usa_classified_timevarying.shp──▶ 4
5 ──mtbs_classified.shp──▶ 7
2, 6  standalone
```

So: **3 → 4 → 5 → 7**, with 1, 2, 6 runnable any time. Notebook 3's GeoTIFF conversion must precede 4 and 5.

### Step 3 — Practical advice before you burn cluster hours

- **Add resume points.** Wrap each expensive cell in `if os.path.exists(path): load else: compute; save`.
  Highest value targets: the MODIS GeoTIFF conversion (nb 3), `classify_with_modis` / `classify_with_static`
  (nbs 4, 5), and every `wfpl` fitting loop.
- **The classification loops are the bottleneck and are quadratic-ish.** `classify_with_modis`
  ([landcover_segmentation.py:35](utils/landcover_segmentation.py#L35)) and `classify_with_static`
  ([temporal_analysis.py:1183](utils/temporal_analysis.py#L1183)) loop over *every raster tile × every fire*,
  calling `rasterio.mask.mask` each time, with a `.to_crs()` on the full frame per tile. For ~30k MTBS fires
  × ~20 tiles that's ~600k individual mask operations. Expect many hours. Consider `rasterstats` /
  `exact_extract`, or at minimum spatially pre-filter fires to each tile's bounds before the inner loop.
- **Test on a subset first.** Run notebook 5 on one biome with `R_boot=2` before committing to `R_boot=50`
  — the differential-evolution bootstrap is `n_biomes × n_dists × 3 modes × R_boot` global optimisations.
- **Verify the fitting core reproduces first.** Notebook 2 is fast and self-contained. If its numbers match
  the stored outputs, `wfpl` is behaving; if not, stop and debug there before running anything expensive.

### Step 4 — Compare against stored outputs

Every notebook has its outputs committed, so you can diff numerically cell-by-cell. That is your ground
truth for "did I reproduce it." `git show HEAD:notebooks/5_temporal_analysis.ipynb` if you want a
pristine copy before you start re-running.

---

## 4. Bugs and correctness concerns

Ordered by how much they'd change a published number. I have not executed anything; items marked
**[verify]** are high-confidence from reading but worth a one-line check.

### 4.1 Confirmed — will change results

**(a) The time-varying bootstrap breaks the area↔year pairing, invalidating every reported trend
significance.** [temporal_analysis.py:823-828](utils/temporal_analysis.py#L823)

```python
idx = rng_global.choice(len(data), size=len(data), replace=True)
boot_data = data[idx]                                  # areas resampled
... differential_evolution(lambda p: neg_loglik(p, data=boot_data), ...)
```

`neg_loglik` reads `years` from the enclosing scope — it is **never resampled**. So each bootstrap replicate
pairs resampled fire sizes with the *original, unpermuted* year vector. This destroys exactly the
association the model is estimating. The resulting SEs on `p1'`/`p2'` are not the sampling distribution of
the slope, and the `p1_slope_sig` / `p2_slope_sig` flags produced by
`summarize_timevary_results_mode` ([temporal_analysis.py:321](utils/temporal_analysis.py#L321)) are
therefore unreliable. Direction of bias is most likely toward *over*-declaring significance. Fix: `boot_data
= data[idx]; boot_years = years[idx]` and pass both. Present in **both**
`analyze_time_varying_mle` (L823) and `analyze_time_varying_mle_refined` (L2987), and in
notebook 7's inline `analyze_time_varying_mle_evtname`.

**(b) Notebook 7 runs on a double-counted, biome-truncated subset of MTBS.** Notebook 5 cell 42 saves
`df_refined` — the *nine overlapping groupings* — to `../data/mtbs_classified.shp`. Notebook 7 cell 2 reads
that file as if it were the fire table. Consequences:

- Fires appear **multiple times**: Grasslands rows are in both `Grasslands` and `Grasslands + Open
  shrublands E of 130W`; Open shrublands rows can appear up to **three** times.
- Only the four `valid_biomes` survive — Evergreen Needleleaf, Mixed forest, Savannas, Woody savannas,
  Closed shrublands, Croplands etc. are **silently dropped** from the entire EVT analysis.

Every EVT fit, count, and `n ≥ 400` gate in notebook 7 is computed on this distorted table. Fix: save the
un-duplicated `mtbs_classified` for downstream use and keep `df_refined` as an analysis-only frame.

**(c) Duplicate function definitions — later ones silently override earlier ones, and one notebook call
now silently no-ops.** `plot_parameter_heatmap` is defined **three times** in
[spatial_analysis.py](utils/spatial_analysis.py) at L394, L599, and L1298 — with *incompatible signatures*:

```python
L599:   plot_parameter_heatmap(best_fits_gfa, biome, distribution, ...)
L1298:  plot_parameter_heatmap(best_fits_gfa, df_tv, biome=..., distribution=...)   # wins
```

Notebook 6 cell 11 calls the old form, `plot_parameter_heatmap(best_fits_gfa, "Savannas",
"truncated_power_law")`, which now binds `df_tv="Savannas"`, `biome="truncated_power_law"`. The saved
output confirms this already happened: the cell produced **no figure**, only the text
`No data for truncated_power_law – truncated_power_law`. Same pattern:
`fit_poisson_tail_trend_by_biome_highres` is defined twice in
[temporal_analysis.py](utils/temporal_analysis.py) (L1837, L2403); `classify_with_modis` and
`load_shapefile` exist in both `landcover_segmentation.py` and `temporal_analysis.py`, so a
`from ... import *` of both shadows one with the other.

**(d) The `*_excess` distributions are fit to one model and scored under a different one.**
`Lognormal_Excess.fit` ([powerlaw.py:3103](utils/wildfire_powerlaw/powerlaw.py#L3103)) sets μ, σ from
`log(x − xmin)`. But it inherits `Lognormal.pdf` / `loglikelihoods`, which evaluate a lognormal in the
**unshifted** `x`, renormalised over `[xmin, ∞)`. So the parameters are not the MLE of the likelihood
used to score them. Identical issue in `Weibull_Excess` (L3135), which optimises its own shifted NLL on
`y = x − xmin` and is then scored by `Distribution.loglikelihoods` on unshifted `x`. Since
`likelihood_matrix_and_best` ranks families by exactly that likelihood, **the two `_excess` families are
systematically and unfairly penalised** in every "best fit" determination in the project.

**(e) Model selection uses raw log-likelihood with no complexity penalty.**
[powerlaw.py:3200](utils/wildfire_powerlaw/powerlaw.py#L3200): `best = max(totals, ...)`, tie-broken by
parameter count only on *exact* ties. A 2-parameter family that nests a 1-parameter family can never lose.
The `Δloglik ≤ 2` cutoff used downstream is a rough stand-in for ΔAIC ≈ 4, and the `reduces_to` CI check
partially compensates — but the reported `best_fit` string is biased toward 2-parameter families. Worth
reporting AIC/BIC alongside.

**(f) `min_n` is applied before the `xmin` cut in the spatial path, but after it in the temporal path.**
`analyze_gfa_by_grid` ([spatial_analysis.py:71](utils/spatial_analysis.py#L71)) tests
`len(box_df) < min_n` on the **unfiltered** tile, then fits with `xmin=4`. Since most GFA fires are well
under 4 km², a tile with 400 total fires may have only a handful of tail points. `analyze_time_varying_mle`
([temporal_analysis.py:624](utils/temporal_analysis.py#L624)) does it the other way round. The "≥ 400
fires" criterion therefore means different things in notebook 6 vs notebook 5.

### 4.2 Likely bugs — **[verify]**

**(g) The time-varying truncated-power-law normaliser probably evaluates to NaN.**
[temporal_analysis.py:539](utils/temporal_analysis.py#L539):

```python
Z = (lambd ** (1 - alpha)) * np.exp(lambd * xmin) * gammaincc(1 - alpha, lambd * xmin) * gamma(1 - alpha)
Z = np.clip(Z, 1e-300, 1e300); logZ = np.log(Z)
```

Two problems. `scipy.special.gammaincc(a, x)` requires `a ≥ 0` and returns `nan` otherwise; for any α > 1
(the entire regime of interest) `1 − α < 0`. And `np.clip` **propagates NaN** rather than clamping it, so
the `try/except` never fires — the NaN flows silently into `logZ` and out through the whole log-likelihood.
Separately, the analytic form looks wrong regardless: for a density ∝ `x^(−α)·exp(−λ(x − xmin))` on
`[xmin, ∞)` the normaliser is `exp(λ·xmin)·λ^(α−1)·Γ(1−α, λ·xmin)`, i.e. **`λ^(α−1)`, not `λ^(1−α)`**. Check
with one line: `scipy.special.gammaincc(-0.5, 1.0)`. If it's NaN, every time-varying TPL fit in notebooks 5
and 7 is degenerate.

**(h) `power_law` in the time-varying routine will crash or return garbage.**
[temporal_analysis.py:781](utils/temporal_analysis.py#L781) gives `power_law` only **two** bounds
`[(0.0, 5), (-1, 1)]`, but:
- mode `"both"` unpacks `a1, b1, a2, b2 = params` from a length-2 vector → `ValueError` → swallowed by the
  bare `except` → returns `np.inf` for *every* candidate → DE returns an arbitrary point;
- modes `"p1_only"` / `"p2_only"` index `bounds[2]` / `bounds[3]` **outside any try block** → uncaught
  `IndexError` that kills the whole run.

Latent today only because `power_law` rarely clears the `Δloglik ≤ 2` candidate filter. It will bite the
moment it does.

**(i) The weak prior is on the wrong scale for every non-TPL family.**
[temporal_analysis.py:762](utils/temporal_analysis.py#L762): `prior_center_a1 = np.log(max(p1_static,
1e-6))`. For the TPL that's correct — `a1` genuinely lives on a `log(α−1)` scale. But for lognormal,
`μ(t) = a1 + b1·t` is on the **raw** scale, so the prior pulls μ toward `log(μ_static)`. Worse: lognormal μ
can be negative, in which case `max(μ, 1e-6)` → `log(1e-6)` = **−13.8**, a wildly wrong prior centre. The
weight is small (`1e-3`) so the damage is bounded, but it is a genuine misspecification.

**(j) Time-varying log-densities are not truncation-corrected at `xmin`, so they aren't comparable to the
static fits.** In `analyze_time_varying_mle`, `logpdf_lognormal` (L497) and `logpdf_weibull` (L573) return
the **plain, untruncated** density and merely mask `x ≤ xmin`. The static `wfpl` path renormalises over
`[xmin, ∞)`. `logpdf_powerlaw`, `logpdf_stretched_exp` and `logpdf_genpareto` *are* correctly tail-normalised.
Consequences: (i) fitting a lognormal to data truncated at 4 km² without the correction biases μ and σ;
(ii) static and time-varying log-likelihoods are on different scales and must not be compared directly;
(iii) even *within* the time-varying comparison, families are scored inconsistently against each other.

**(k) `logpdf_genpareto` divides by ξ with no ξ→0 branch.** [temporal_analysis.py:557](utils/temporal_analysis.py#L557):
`-(1/xi + 1) * log1p(xi*y/sigma)` with `xi = np.clip(xi, -1, 2)` — ξ = 0 is inside the allowed range and is
exactly the exponential limit the optimiser will drift toward. Compare the static
`Generalized_Pareto._pdf_base_function` ([powerlaw.py:2259](utils/wildfire_powerlaw/powerlaw.py#L2259)),
which *does* have an `isclose(xi, 0)` special case.

**(l) `logpdf_powerlaw` clips α to `[0, 5]` and takes `log(abs(C))`.**
[temporal_analysis.py:521](utils/temporal_analysis.py#L521). α < 1 is an improper (non-normalisable)
power law; taking the absolute value of the normaliser lets the optimiser wander into that region and
score it as if it were valid.

**(m) All-NaN pixels become land-cover class 0 = "Water" in the static MODIS raster.**
[temporal_analysis.py:1166](utils/temporal_analysis.py#L1166): `stats.mode(..., nan_policy="omit")`
returns NaN where every year is NaN (having masked nodata and class 17 = Water Bodies), then
`.astype(np.uint8)` on NaN is undefined behaviour — in practice 0 on x86. `classify_with_static` only
filters `!= nodata` and `!= 255`, so 0 survives and `modis_to_gfa[0]` = `"Water"`. Meanwhile
`classify_with_modis` leaves 17 unmapped, so it becomes `"Other"`. The two classifiers therefore disagree
by construction on water pixels — which slightly deflates the "agreement between time-varying and static
MODIS classifications" figure reported in notebook 5 cell 8.

### 4.3 Hazards and code smells

- **Import-time global side effects.** [powerlaw.py:32-37](utils/wildfire_powerlaw/powerlaw.py#L32)
  runs `warnings.filterwarnings("ignore")` and **`sys.stderr = open(os.devnull, "w")`** at module scope.
  Merely importing `wfpl` silently blackholes stderr for the rest of the Python session. This will hide
  warnings from *unrelated* libraries and make debugging your reproduction unnecessarily hard. Neutralise
  it (`SUPPRESS_OUTPUT = False`) before you start. Several `utils` functions additionally call
  `warnings.filterwarnings("ignore", category=RuntimeWarning)` as a global side effect.
- **Shapefile 10-character column truncation.** Round-tripping through `.shp` mangles column names, and
  the code compensates inconsistently. `modis_class` → `modis_clas` is handled by hand in notebook 4
  cell 9 (GFA uses `"modis_clas"`, MTBS/Idaho in memory use `"modis_class"`), and notebook 7 cell 2
  renames `"modis_cl"`/`"modis_cl_1"` — but `pandas.rename` **silently ignores keys that don't exist**, so
  if the real truncated name is `modis_clas` that rename is a no-op and `modis_class_timevary` never
  materialises. Switch the intermediate artifacts to GeoParquet or GeoPackage.
- **Non-deterministic tile assignment.** Both classifiers loop tiles-outer / fires-inner and write with
  `fires_gdf.at[idx, col] = cls`, so a fire straddling two tiles gets whichever tile is visited last —
  dict-iteration order, i.e. `os.listdir` order. Majority vote is computed *per tile*, never pooled across
  tiles. Fires near tile seams are effectively arbitrary.
- **Orphan artifacts in `data/`.** `mtbs_static_fits_by_biome_no_savanna.pkl`,
  `gfa_spatial_time_varying_savannas_tpl.pkl`, `..._shrublands_lognormal.pkl` and
  `..._evergreen_broadleaf_gp.pkl` are referenced by **no notebook** — they came from variants of
  notebook 6 cell 14 that were never committed. Note that
  `gfa_spatial_time_varying_evergreen_broadleaf_gp.pkl` is **5 bytes** — an empty pickled dict, i.e. a run
  that produced nothing. Also, `overall_results_evtname_collapsed.pkl` is written by notebook 7 cell 16
  but is **absent** from `data/`.
- **Bootstrap SEs use `ddof=0`** for the time-varying path (`np.std(boot_params, axis=0)`,
  [L2757](utils/temporal_analysis.py#L2757)) but `ddof=1` in `summarize_parameters_bootstrap`
  ([powerlaw.py:3026](utils/wildfire_powerlaw/powerlaw.py#L3026)). Negligible at R = 50, inconsistent
  regardless.
- `summarize_parameters_bootstrap` will raise `KeyError` on `'p1_se'` if called with `R = 0`.
- `likelihood_matrix_and_best` takes an `alpha=0.05` argument that is never used.
- `Fit.__init__` does `self.xmin_distribution.pdf_ends_at_xmax = ...` — mutating a *class* attribute
  globally. Inherited from upstream, but it makes `Fit` objects non-independent.
- `utils/__init__.py` is empty while `setup.py` declares `packages=find_packages(include=["utils", ...])`
  and lists the unrelated PyPI `powerlaw` in `install_requires` — harmless but misleading, since the
  vendored fork is what's actually used.
- Notebook 7's analysis functions (`analyze_time_varying_mle_evtname`, its own copy of
  `plot_refined_biome_ccdf_cdf_qq`) are defined **inline in the notebook**, diverging from the `utils/`
  versions. Any fix to `temporal_analysis.py` must be mirrored there by hand.

---

## 5. Suggested first three moves

1. **Build a working Linux environment and get notebook 2 to reproduce its stored outputs.** Cheap, fast,
   and it validates the entire fitting core before you spend anything on data.
2. **Fix (a) — the bootstrap year pairing — before re-running notebook 5.** It's a two-line change and it
   is the difference between publishable and not-publishable trend significance.
3. **Download MTBS + MODIS (the notebook 3→4→5 spine) and add `.pkl`/`.parquet` resume points as you go.**
   Defer GFA (notebook 6, which the author abandoned) and LANDFIRE (notebook 7, which additionally needs
   bug (b) fixed) until the main MTBS result reproduces.
