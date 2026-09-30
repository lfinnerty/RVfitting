# RV databases

The RV data under `RVdatabases/` are not committed to git, except the Teklu
table and the ExoArchive tables that were committed before this list was
written. Everything can be rebuilt with

```bash
python download_rv_databases.py
```

That command skips files that are already on disk. `vizier`, `exoarchive`,
`sophie`, `neid`, `espresso`, `harps_drs`, `elodie`, or `literature` can be passed to
fetch only one group. Readers are in `rv_io.py`, and
`plot_rvs.py --source` selects them by the name in the "Source" column.

| Source | Local path | Origin | Reference | Reader notes |
|---|---|---|---|---|
| `Teklu` | `tablea1_Teklu.dat`, `ReadMe_Teklu` | VizieR [J/A+A/702/A68](https://cdsarc.cds.unistra.fr/ftp/J/A+A/702/A68/) (`tablea1.dat.gz`, decompressed) | Teklu et al. 2025, A&A 702, A68 | Keck/HIRES, NZP-corrected. Values are m/s despite the ReadMe. |
| `ExoArchive` | `exoarchive/UID_*_RVC_*.tbl` | NASA Exoplanet Archive bulk RV download, listed in `exoarchive/wget_exoarchive_20260924.bat` | Per file (`REFERENCE` header) | Already barycentric; per-file median removed; analysis year from `REFERENCE`. |
| `CLS` | `CLS/table6.dat.gz`, `CLS/table2.dat`, `CLS/ReadMe` | VizieR [J/ApJS/255/8](https://cdsarc.cds.unistra.fr/ftp/J/ApJS/255/8/) | Rosenthal et al. 2021, ApJS 255, 8 | HIRES-k, HIRES-j, APF, and Lick offsets fitted separately. |
| `HARPS` | `HARPS_RVBank2/table4.dat.gz`, `table1.dat.gz`, `ReadMe` | VizieR [J/A+A/683/A125](https://cdsarc.cds.unistra.fr/ftp/J/A+A/683/A125/) (corrected version, 20 Apr 2026) | Perdelwitz et al. 2024, A&A 683, A125 | `RV_mlc_nzp`, SERVAL flag 0 only; offsets before and after the 2015 fibre upgrade. |
| `HARPS2020` | `HARPS/rvbank.dat`, `HARPS/Readme` | VizieR [J/A+A/636/A74](https://cdsarc.cds.unistra.fr/ftp/J/A+A/636/A74/) (`rvbank.dat.gz`, decompressed) | Trifonov et al. 2020, A&A 636, A74 | Kept because ~3% of its points (e.g. CoRoT hosts) are absent from v2; v2 wins duplicates. |
| `SOPHIE` | `SOPHIE/ccf_ra*.txt` (144 RA bins) | OHP SOPHIE archive, `sophiecc` CCF table ([atlas.obs-hp.fr/sophie](http://atlas.obs-hp.fr/sophie/)), fields `seq,objname,bjd,mask,ccf_offline,rv,err` | Pipeline RVs; see the archive's acknowledgement text | Most common mask per star; gross outliers removed; error floor added; offsets before and after SOPHIE+ (2011). **See `SOPHIE/DATA_POLICY.md` before sharing.** |
| `Hebrard` | `Hebrard2016/rvdata.dat`, `Readme` | VizieR [J/A+A/588/A145](https://cdsarc.cds.unistra.fr/ftp/J/A+A/588/A145/) (`table1.dat`) | Hébrard et al. 2016, A&A 588, A145 | Kept over the SOPHIE archive for the same exposures (newer analysis). |
| `NeveuVanMalle` | `Neveu-VanMalle2014/w94a_rv.dat`, `w94b_rv.dat`, `ReadMe` | VizieR [J/A+A/572/A49](https://cdsarc.cds.unistra.fr/ftp/J/A+A/572/A49/) | Neveu-VanMalle et al. 2014, A&A 572, A49 | Euler/CORALIE RVs of WASP-94 A and B (photon-noise errors); median removed. |
| `NEID` | `NEID/neid_l2.csv`, `NEID/queried_targets.json` | NEID archive `neidl2` table via TAP ([neid.ipac.caltech.edu](https://neid.ipac.caltech.edu/)); `python download_rv_databases.py neid` queries the stars with fits in `plots/` | NEID DRP (v1.5) CCF RVs | Barycentric `ccfrvmod`/`dvrms` (0.1 m/s precision in the table), HR mode, public rows only; offsets before and after the 2022 Contreras-fire shutdown. |
| `ESPRESSO` | `ESPRESSO/espresso_ccf.csv`, `ESPRESSO/ccf/*.fits`, `ESPRESSO/queried_targets.json` | ESO archive: ObsCore TAP search (5" box around the SIMBAD position), then each product's science-fibre CCF via datalink; `python download_rv_databases.py espresso` queries the stars with fits in `plots/` | ESPRESSO DRS (`QC CCF RV`) | Barycentric CCF RVs; most common mask per star; offsets per instrument mode and before/after the 2019 fibre-link change (ESPRESSO18/19). |
| `HARPSDRS` | `HARPS_DRS/harps_drs_ccf.csv`, `HARPS_DRS/ccf/*_ccf_*_A.fits`, `HARPS_DRS/queried_targets.json` | ESO archive HARPS products since 2022-01-01 (5" box, or 20" if the product's target name is one of the star's identifiers); the fibre-A CCF is extracted from each product's ~6 MB DRS 3.x tarball, and nights with long time series are thinned to 5 spectra. `python download_rv_databases.py harps_drs` queries the stars with fits in `plots/` | HARPS DRS 3.x (`DRS CCF RVC`, `DRS CCF NOISE`) | Drift-corrected barycentric CCF RVs; most common mask per star; own offset (the DRS zero point differs from RVBank's SERVAL RVs). |
| `ELODIE` | `ELODIE/elodie_ccf.csv`, `ELODIE/queried_targets.json` | OHP ELODIE archive (1994–2006): `e501` CCF table by 20" cone search, with each spectrum's `e500` header for its UT start and exposure time; `python download_rv_databases.py elodie` | ELODIE TACOS pipeline (`vfit`) | No RV errors in the archive: 10 m/s at S/N 100, scaled as 100/(S/N) below that; mid-exposure BJD computed from the header. Most common mask per star; gross outliers removed. |
| `Literature` | `literature/literature_rvs.csv`, `literature/<arXiv ID>/` | RV tables transcribed from arXiv sources, listed in `LITERATURE_TABLES` in `download_rv_databases.py`: Paredes et al. 2021 (CHIRON, HIP 86221) and Quinn et al. 2014 (TRES, HD 285507); `python download_rv_databases.py literature` | Per paper | One dataset (median removed) per table, labelled `Literature`. |
| `Synthetic` | `Synthetics/*_synthetic_RV.tbl` | Generated by `generate_synthetic_rvs.py` | — | Not real data; only loaded with `plot_rvs.py --synthetics`. |

## Derived and cache files

- `simbad_cache.json` stores the SIMBAD records (IDs, coordinates, systemic RV)
  that `rv_io` uses to match star names. It rebuilds itself on demand.
- `exoarchive_hosts.txt` is written by `list_rv_systems.py --file-lists`.
- `exoplanet_archive_pscomppars_20260929.csv` is the Exoplanet Archive table used by
  the hot-Jupiter scripts (query in the README). Its `pl_bmassj` holds Earth masses
  for some RV planets; `fit_rv_hot_jupiters.py` drops planets whose K contradicts
  the listed mass.
- `exoplanet_eu_shortperiod_20260930.csv` (transit status for `fit_rv_hot_jupiters.py`)
  is exoplanet.eu's `exoplanet.epn_core` table from the VO-Paris TAP service
  (`http://voparis-tap-planeto.obspm.fr/tap`): columns `target_name, alt_target_name,
  star_name, detection_type, period, tzero_tr, tzero_tr_error_min, radius,
  radius_error_min, inclination, ra, dec` for `period < 8.5`, fetched in six period bins
  (0, 1.5, 2.5, 3.3, 4.2, 5.5, 8.5 d) because the service truncates large replies.
- `twomass_kmag_rv_hot_jupiters_20260930.csv` holds the 2MASS K (VizieR II/246, nearest
  source within 5", or 20" for high proper-motion stars) of each RV hot Jupiter's host.
- `archive_search_20260930.json` and `vizier_rv_search_20260930.json` cache one-off
  archive and VizieR searches for extra RVs; they are not read by the pipeline.

## Removed

- `rv_data_fulton/` (Howard & Fulton 2016, 76 stars) was removed in the
  commit that introduced this list. All of its points are also in the
  ExoArchive Howard & Fulton 2016 tables, with identical values.

## Not included

- DACE: skipped for now. It needs one query per target and was slow and
  unreliable.
- KPF (KOA): as of 2026-09-29 only ~300 KPF L2 science files are public and none
  for the fitted targets; KOA's download API also rejected the L2 file paths, so
  there is no KPF reader yet. Raw KPF and HIRES frames exist, but no public
  pipeline turns HIRES iodine frames into RVs.
- HARPS-N and GIANO-B (TNG/IA2): frames are listed but flagged private (login required).
- NIRPS (ESO): a few nights for τ Boo only; no reader.
- Damasso et al. 2020 (CDS J/A+A/642/A133, `Damasso2020/`): HARPS-N RVs of V830 Tau,
  downloaded but unused, since its RVs are dominated by starspots.
