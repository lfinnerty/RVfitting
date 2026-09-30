# RV fitting for hot-Jupiter hosts

Tools to combine public radial-velocity (RV) archives for a star, fit the orbit of its
short-period planet, and turn the fit into observing numbers: the transit time and its
uncertainty on a target date, how much new RVs would improve that, and whether the RVs
show an outer planet or long-term trend.

The code was written for planning high-resolution cross-correlation spectroscopy (HRCCS)
of hot Jupiters, whose conjunction times must be known to a few hours or better. It pairs
with [`hrccs_planner`](https://github.com/lfinnerty/ObsTools) (KPIC/ObsTools) for window
selection.

**What it does**

- **Reads eight RV sources:** HIRES (Teklu et al. 2025), the California Legacy Survey,
  HARPS RVBank (2024 and 2020 releases), the SOPHIE archive, Hébrard et al. 2016, NEID, and
  the NASA Exoplanet Archive's published RV tables.
- **Matches star names across sources** through cached SIMBAD aliases.
- **Removes duplicate exposures,** preferring the most recent analysis of each one.
- **Fits the orbit** with one offset and one jitter term per instrument. It compares circular
  and eccentric (Keplerian) orbits and adopts the eccentric one only when e is significant.
  The transit conjunction and its full-covariance uncertainty are propagated to a target date.
- **Flags outer planets and trends** in the residuals, vets them, and fits them jointly with
  the inner planet: as a sinusoid, a second Keplerian, or a quadratic trend.
- **Forecasts** how much new HIRES points would tighten the transit time, surveys
  Exoplanet Archive hot-Jupiter hosts for unlisted outer planets, and exports target lists
  for `hrccs_planner`.

## Contents

| File | Purpose |
|---|---|
| `plot_rvs.py` | Fit and plot one star (the main entry point); also the fitting library |
| `rv_io.py` | Database readers, SIMBAD cache, de-duplication (`load_datasets`, `combine_rv_data`) |
| `download_rv_databases.py` | Download or refresh every RV database into `RVdatabases/` |
| `list_rv_systems.py` | List every system in the databases by SIMBAD main identifier |
| `generate_synthetic_rvs.py` | Draw synthetic RVs on chosen dates from a saved fit |
| `forecast_phase_uncertainty.py` | Forecast how 3–5 new HIRES RVs change the transit-time uncertainty |
| `survey_hot_jupiter_outer_planets.py` | Search Exoplanet Archive hot-Jupiter hosts for unlisted outer planets |
| `make_hrccs_targetlist.py` | Write an `hrccs_planner` target list from the saved fits |
| `RVdatabases/DATABASES.md` | Where each database comes from, how it is read, and how to rebuild it |
| `plots/` | Fit outputs (`<star>_rv_fit_plot.png`, `<star>_rv_fit_parameters.json`) and result tables |

## Installation

Python ≥ 3.10 is required (tested with 3.11).

```bash
git clone git@github.com:lfinnerty/RVfitting.git
cd RVfitting
```

With conda:

```bash
conda create -n rvfit python=3.11
conda activate rvfit
pip install -r requirements.txt
```

Or with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependencies are numpy, scipy, astropy, astroquery (≥ 0.4.8, for SIMBAD's batched
`query_objects`), matplotlib and requests. The scripts are run from the repository
directory; there is nothing to build.

Two optional pieces:
- **`hrccs_planner`**, for the HRCCS steps: install it from
  [ObsTools](https://github.com/lfinnerty/ObsTools) with `pip install -e .`.
- **The `MPLBACKEND=Agg` environment variable**, for running on a machine without a display.
  `plot_rvs.py` calls `plt.show()` after saving; with this backend it only warns.

## Setting up the data

The Teklu HIRES table and the Exoplanet Archive RV tables are in the repository. Everything
else is downloaded, not committed:

```bash
python download_rv_databases.py            # everything (skips files already present)
python download_rv_databases.py vizier     # just the VizieR catalogs
python download_rv_databases.py sophie     # just the SOPHIE archive
python download_rv_databases.py neid       # NEID RVs for the stars with fits in plots/
```

| Group | What it fetches | Size | Notes |
|---|---|---|---|
| `vizier` | Teklu (HIRES), HARPS RVBank 2024 and 2020, CLS, Hébrard 2016 | ~270 MB | The 2020 HARPS RVBank table is the largest file (~180 MB). |
| `exoarchive` | Any Exoplanet Archive RV tables missing from `RVdatabases/exoarchive/` | ~7 MB | Uses the archive's bulk-download list. |
| `sophie` | The public SOPHIE CCF table, in 144 RA bins | ~10 MB | The server is slow: about 30 minutes. |
| `neid` | NEID L2 RVs, from the archive's metadata table | < 1 MB | Only stars that have a fit JSON in `plots/`; run again after fitting new stars. |

Downloads resume where they stopped and are checked before being saved, so an
interrupted run can simply be repeated. See `RVdatabases/DATABASES.md` for the provenance
of each source and the details of how it is read.

**SOPHIE data policy:** the SOPHIE archive forbids organized redistribution of its data, so
`RVdatabases/SOPHIE/` is git-ignored. Do not commit it or share it in bulk. Publications
using those RVs should include the archive's acknowledgement: *"Based on data retrieved
from the SOPHIE archive at Observatoire de Haute-Provence (OHP), available at
atlas.obs-hp.fr/sophie"*.

**SIMBAD cache:** star names are resolved through SIMBAD, and the results are cached in
`RVdatabases/simbad_cache.json` (git-ignored). The first fit of a new star makes one small
query. Failed lookups are not cached, so they are retried next time.

**Exoplanet Archive table** (needed only by the hot-Jupiter survey):

```bash
curl -G "https://exoplanetarchive.ipac.caltech.edu/TAP/sync" \
  --data-urlencode "query=select pl_name,hostname,pl_orbper,pl_bmassj,pl_bmassprov,pl_radj,pl_rvamp,pl_orbeccen,sy_pnum,discoverymethod,disc_year,ra,dec from pscomppars" \
  --data-urlencode "format=csv" -o RVdatabases/exoplanet_archive_pscomppars_20260929.csv
```

The survey script expects that exact file name.

## Fitting a star

```bash
python plot_rvs.py "HD 217107"
```

Names are matched through SIMBAD, so `HD217107`, `HD 217107` and other SIMBAD identifiers
all work. The fit prints a summary:

```
HD 217107: loaded 918 RV measurements from all source(s) (453 duplicate(s) removed)
HD 217107: adopted keplerian orbit (e/sigma_e = 117.63)
HD 217107: best-fit period = 7.12687 +/- 3.81526e-06 days
HD 217107: RV semiamplitude = 141.971 +/- 0.158921 m/s
HD 217107: transit conjunction = BJD 2458359.389863 +/- 0.002645
HD 217107: orbital phase on 2027-07-01 = 0.949470 +/- 0.000457 cycles (+/- 0.078164 hours)
HD 217107: eccentricity = 0.125733 +/- 0.0010689
HD 217107: argument of periastron = 23.5807 +/- 0.47811 degrees
HD 217107: time of periastron = BJD 2458358.326291 +/- 0.009480
HD 217107: jitter (m/s): CLS Lick 13.2, ExoArchive 9.4, Teklu 4.8, CLS HIRES-j 2.0, CLS APF 2.4, SOPHIE+ 2.4, NEID-pre 0.9, NEID-post 1.3
HD 217107: outer-signal flag = outer_planet; fitted outer planet, eccentric Keplerian (e = 0.382 +/- 0.005, omega = -155 deg): P = 5115 +/- 10 d, K = 51.1 +/- 0.5 m/s, m sin i = 4.01 Mjup (M*/Msun)^(2/3)
HD 217107: reduced chi2 (with jitter) = 1.01889
```

The first run of a new star also prints a line when it queries SIMBAD.

It writes two files to `plots/`:
- **`HD 217107_rv_fit_plot.png`:** the combined RVs, the periodogram, the RVs folded on the
  inner planet's period, and, when there is an outer signal, a fourth panel. That panel
  shows the outer planet folded on its own period, a fitted trend, or, for candidates that
  were not fitted, the residuals with seasonal means. Every title says which models were
  used.
- **`HD 217107_rv_fit_parameters.json`:** every fitted quantity (see
  [Output](#output-json)).

Options:

```bash
python plot_rvs.py HD2638 --source harps          # one source only
python plot_rvs.py HD143105 --outer-model none    # never add an outer term
python plot_rvs.py "HD 83443" --outer-model sinusoid   # force a circular outer planet
python plot_rvs.py HD2638 --synthetics            # include RVdatabases/Synthetics/ points
```

- **`--source`** takes `all` (the default), `teklu`, `exoarchive`, `cls`, `harps`,
  `harps2020`, `sophie`, `hebrard`, `neid`, `espresso` or `neveuvanmalle`.
- **`--outer-model`** takes `auto` (the default), `none`, `sinusoid`, `keplerian` or `trend`.
  With `auto`, a term is added only when an outer signal passes vetting. Forcing a model
  still runs the vetting, and a forced term that fails it is labelled a candidate.
- **`--synthetics`:** fits that include synthetic points are saved with a
  `_with_synthetics` suffix, so they never overwrite real-data fits.

### Using it from Python

```python
import rv_io
import plot_rvs

datasets = rv_io.load_datasets("HD 217107", plot_rvs.OBSERVED_SOURCES)
bjd, rv, rv_error, labels = rv_io.combine_rv_data(datasets)
fit = plot_rvs.fit_system(bjd, rv, rv_error, labels)
print(fit.parameters["period_days"], fit.parameters["phase_uncertainty_hours_on_2027-07-01"])
```

- **`fit_system`** also accepts `outer_model=` and `initial_period=`. `initial_period`
  starts the period search from a known value, e.g. a transit period, instead of the
  periodogram peak.
- **`plot_rvs.plot_fit(star, bjd, rv, rv_error, labels, fit)`** returns the figure.

## How the fit works

1. **Loading and de-duplication (`rv_io`).** Each source is read with its own conventions:
   units, time frame, barycentric state, zero points, and mask choice for SOPHIE.
   - **Name matching:** stars are matched through their SIMBAD identifiers.
   - **Duplicates:** when the same exposure appears in two datasets (same instrument, within
     5 minutes, or 30 minutes for Lick/Hamilton), the copy from the more recent analysis is
     kept. Matching is one-to-one, and points within one dataset are never merged.
2. **Offset groups.** Every instrument and zero-point era gets its own offset, e.g.
   `CLS HIRES-j`, `HARPS-post` (after the 2015 fibre upgrade), `SOPHIE+` (after 2011),
   and NEID before and after the 2022 Contreras fire.
3. **Period.** The inner period is the highest peak of an error-weighted Lomb–Scargle
   periodogram between 1.2 and 8 days (`plot_rvs.PERIOD_RANGE`). Obvious 5σ outliers are
   excluded.
4. **Orbit.** The model is (γ, K, P, Tc, √e cos ω, √e sin ω, offsets), fitted jointly with
   per-instrument jitter (iterated to convergence).
   - **Conjunction:** Tc is the transit conjunction (f + ω = π/2), placed at the orbit nearest
     the weighted centre of the data.
   - **Circular vs eccentric:** both are fitted, and the eccentric orbit is adopted only if
     e > 2.45 σ_e (Lucy & Sweeney 1971).
   - **Target-date timing:** the phase on the target date (2027-07-01; `plot_rvs.TARGET_DATE`)
     uses the full Tc–P covariance.
5. **Outer signals.** The one-planet residuals are searched for long-period signals
   (60 d – 3× baseline) and activity-like ones (10–60 d). A detection is fitted jointly and
   kept only if it passes all of the following:
   - K/σ_K ≥ 20 in a refit to nightly-binned residuals;
   - at least 10 independent nights and 5 observing seasons;
   - a period not within 5% of 1, ½ or ⅓ year;
   - for trends, an implied amplitude of at least 3× the per-point noise.

   A vetted outer planet is refitted as a Keplerian. That eccentric orbit is adopted if e is
   significant, e ≤ 0.8, and its K stays within 2× the circular fit's. Anything that fails
   is reported as a candidate (`outer_candidate`, `trend_candidate`) and not fitted.
   Short-period residual peaks are reported as `possible_activity`.

## Other tools

**Synthetic RVs** on given dates, drawn from a saved fit and the star's empirical noise.
They're written to `RVdatabases/Synthetics/` in ExoArchive table format:

```bash
python generate_synthetic_rvs.py HD2638 2027-06-15 2027-06-18 2027-07-02
```

**2027A forecast:** synthetic 3, 4 or 5-point HIRES realizations on observable Maunakea
nights, re-fitted with the full pipeline. Writes `plots/forecast_2027A_hires.csv`:

```bash
python forecast_phase_uncertainty.py                          # targets in plots/rv_target_summary.csv
python forecast_phase_uncertainty.py --targets HD2638 HD143105 --counts 3 5 --realizations 10
```

**Phase uncertainty on later dates:** each summary target's conjunction-time uncertainty,
propagated from its saved fit with no new data. Writes `plots/phase_uncertainty_2027B.csv`
(2027B start, middle and end by default):

```bash
python propagate_phase_uncertainty.py
python propagate_phase_uncertainty.py --dates 2028-02-01 2028-05-01 2028-07-31 --output plots/phase_uncertainty_2028A.csv
```

**Hot-Jupiter outer-planet survey:** every Exoplanet Archive hot-Jupiter host with at least
20 RVs over at least a year, compared with the Archive's listed planets. Writes
`plots/hot_jupiter_outer_survey.csv`. It resumes if interrupted:

```bash
python survey_hot_jupiter_outer_planets.py
python survey_hot_jupiter_outer_planets.py --plot-candidates   # forced-fit plots in plots/survey_candidates/
```

**System list** from all databases (SIMBAD main identifiers), or the ExoArchive
host-to-file table:

```bash
python list_rv_systems.py --output system_list.dat
python list_rv_systems.py --file-lists
```

**HRCCS planning with `hrccs_planner`:** export the fitted targets, then ask how much S/N
the best dayside windows lose to ephemeris uncertainty:

```bash
python make_hrccs_targetlist.py            # writes plots/hrccs_targetlist.csv
hrccs-plan phase-sensitivity plots/hrccs_targetlist.csv --site keck2 \
    --start 2027-08-01 --end 2028-01-31 --airmass-k 0.05 --seeing-exponent 0.6 \
    --sigma-column "sigma_t now (h)" --sigma-column "sigma_t +3 RVs (h)" \
    --csv plots/hrccs_phase_sensitivity_2027B.csv
```

## Output JSON

The main keys of `plots/<star>_rv_fit_parameters.json`:

| Key | Meaning |
|---|---|
| `orbit_model` | `circular` or `keplerian` (the adopted inner orbit) |
| `period_days`, `semiamplitude_m_per_s`, `conjunction_bjd` | Adopted orbit, each with `_uncertainty` |
| `eccentricity`, `omega_degrees`, `periapsis_bjd` | Eccentric elements (ω is the star's); `null` for circular orbits |
| `phase_on_2027-07-01`, `phase_uncertainty_hours_on_2027-07-01` | Phase from transit on the target date and its uncertainty |
| `jitter_m_per_s`, `source_offsets_m_per_s` | Per-instrument jitter and offsets |
| `reduced_chi_squared` | Including jitter (≈ 1 by construction) |
| `circular_fit`, `keplerian_fit` | Both inner-orbit fits in full |
| `outer_signal` | `flag`, the residual peaks, `vetting` (significances and rejection reasons), the fitted outer term (`fit`), and the circular vs Keplerian `outer_model_comparison` |
| `includes_synthetics` | Whether synthetic points were in the fit |

## Limitations

- **Period range.** The inner period is searched between 1.2 and 8 days. Pass
  `initial_period` to `fit_system` to use a known period instead.
- **Uncertainties** assume white noise plus per-instrument jitter. Unmodelled planets or
  activity inflate the jitter rather than being fitted, unless they pass the outer-signal
  vetting.
- **Activity.** Outer-signal candidates are not checked against stellar activity
  indicators; magnetic cycles can mimic long-period planets.
- **Pipeline data quality.** SOPHIE archive RVs are pipeline products. Gross outliers are
  removed and an instrumental error floor is added, but there is no quality flag. Data mixed
  across observing modes can show season-to-season offsets.
- **Archive errors are not corrected.** For example, the Exoplanet Archive's Vogt et al. 2002
  HIRES table for HD 68988 has its dates shifted by 1000 days, so those points receive a
  large jitter.
- **Limited coverage.** NEID is fetched only for stars with fits. KPF has no public L2 RVs
  for these targets, and HIRES raw frames have no public RV pipeline. See
  `RVdatabases/DATABASES.md`.

## Data sources and citations

Please cite the RV sources you use:

- **HIRES:** Teklu et al. 2025, A&A 702, A68
- **California Legacy Survey:** Rosenthal et al. 2021, ApJS 255, 8
- **HARPS RVBank:** Perdelwitz et al. 2024, A&A 683, A125; Trifonov et al. 2020, A&A 636, A74
- **SOPHIE:** Hébrard et al. 2016, A&A 588, A145; the SOPHIE archive acknowledgement above
- **NEID:** the NEID archive at NExScI
- **Published RV tables:** the NASA Exoplanet Archive, with each table's own reference in
  its `REFERENCE` header
- **Star names:** SIMBAD
