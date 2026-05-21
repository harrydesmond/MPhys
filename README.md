# TNG/NH RAR Interpretation Workflow

This branch contains the code used to reproduce the SPARC and simulation
versions of the fRAR figures, including the NewHorizon `gbar_tree` refresh for
galaxy 539.

The repository has two relevant layers:

- `RARinterpret/`: the modified fRAR analysis package and figure scripts.
- top-level `MPhys` scripts/data: simulation data assembly, tree-gravity
  helpers, diagnostics, and `Combined_TNG_NH_dataframe.csv`.

Generated plots, result pickles, Slurm output, backups, and smoke-test outputs
are not part of the git branch. They are products of the commands below.

## Environment

Production runs were made on Glamdring from:

```bash
cd /mnt/extraspace/hdesmond/RAR/Joshua/MPhys
```

The fRAR analysis scripts use Glamdring Python 3.11:

```bash
module load python/3.11.4
export LD_LIBRARY_PATH=/usr/local/shared/python/3.11.4/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret:/mnt/extraspace/hdesmond/RAR/Joshua/TaskmasterMPI:/mnt/extraspace/hdesmond/RAR/Joshua/vendor:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib-rar-$USER
```

The raw NewHorizon tree-gravity path used for the one-galaxy rerun uses
Glamdring Python 3.9.6 because that environment has the needed raw-data stack:

```bash
/usr/local/shared/python/3.9.6/bin/python3 run_pytree_one.py --gal 539 --sim NH
```

The committed Glamdring wrapper scripts set the important environment
variables themselves, including `PYTHONPATH`, `LD_LIBRARY_PATH`, the Matplotlib
config directory, and the MPI TCP interface pins.

## Combined Simulation Data

The figure scripts read the tracked combined file:

```text
Combined_TNG_NH_dataframe.csv
```

It contains TNG and NewHorizon radial-bin rows with both baryonic acceleration
definitions:

- `gbar_sph`: the original spherical/RAR pipeline baryonic acceleration.
- `gbar_tree`: the pytreegrav baryonic acceleration.

To rebuild it from the source CSV products on Glamdring:

```bash
cd /mnt/extraspace/hdesmond/RAR/Joshua/MPhys
python3 build_combined_dataframe.py --out Combined_TNG_NH_dataframe.csv
```

For the NH 539 refresh, first rerun the one-galaxy pytree calculation:

```bash
cd /mnt/extraspace/hdesmond/RAR/Joshua/MPhys
./job_pytree_539_glam.sh
PYTREE_DIR_NH_OVERRIDE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun_single/pytree_results \
  python3 build_combined_dataframe.py --out Combined_TNG_NH_dataframe.csv
```

Useful diagnostics:

```bash
/usr/local/shared/python/3.9.6/bin/python3 diagnose_pytree_targets.py --gal 539
/usr/local/shared/python/3.9.6/bin/python3 run_pytree_one.py --gal 539 --sim NH
```

## Figure Scripts

All figure commands below run from:

```bash
cd /mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
```

Main scripts:

- `run_fig2_pc.py`: partial correlations, corresponding to Fig. 2.
- `run_fig6_grid.py`: ExtraTrees feature-pair grid, corresponding to Fig. 6.
- `run_fig7_gencomb.py`: generic variable/angle scan, corresponding to Fig. 7.
- `run_tngnh_treeparam_grid.py`: simulation-specific ExtraTrees hyperparameter
  cache generation for `simhyper` runs.

Wrapper scripts:

- `job_sparc_fig2_glam.sh`, `job_sparc_fig6_glam.sh`,
  `job_sparc_fig7_glam.sh`: SPARC runs.
- `job_tngnh_fig2_glam.sh`, `job_tngnh_fig6_glam.sh`,
  `job_tngnh_fig7_glam.sh`: TNG/NH runs; first positional argument is `sph` or
  `tree`.
- `job_tngnh_hyperopt_glam.sh`: simulation-specific hyperparameter caches.

Outputs are written under `RARinterpret/plots/` and `RARinterpret/results/`.
Use `--png-only` for non-publication variants; without it, scripts save both
PNG and PDF.

## Baseline Figures

SPARC baseline:

```bash
./job_sparc_fig2_glam.sh
./job_sparc_fig6_glam.sh
./job_sparc_fig7_glam.sh
```

Simulation baseline with spherical baryonic acceleration:

```bash
./job_tngnh_fig2_glam.sh sph
./job_tngnh_fig6_glam.sh sph
./job_tngnh_fig7_glam.sh sph
```

Simulation baseline with tree baryonic acceleration:

```bash
./job_tngnh_fig2_glam.sh tree
./job_tngnh_fig6_glam.sh tree
./job_tngnh_fig7_glam.sh tree
```

The TNG/NH wrappers use the simulation feature set
`r,SB,MHI,Mstar,Reff,type` for Fig. 2 and the default TNG/NH feature set in
`TNGFrame` for Figs. 6 and 7. The default error model is observational
uncertainties for SPARC and a constant 10 percent relative model for TNG/NH.

## High-Precision Variants

High-precision variants use more resamples/splits/theta points and are labelled
with `--run-label hires`:

```bash
N_REPEAT=5000 ./job_sparc_fig2_glam.sh --run-label hires --png-only
N_SPLITS=2000 ./job_sparc_fig6_glam.sh --run-label hires --png-only
N_THETA=160 N_SPLITS=50 N_RESAMPLE=30 ./job_sparc_fig7_glam.sh --run-label hires --png-only

N_REPEAT=5000 ./job_tngnh_fig2_glam.sh tree --run-label hires --png-only
N_SPLITS=2000 ./job_tngnh_fig6_glam.sh tree --run-label hires --png-only
N_THETA=160 N_SPLITS=50 N_RESAMPLE=30 ./job_tngnh_fig7_glam.sh tree --run-label hires --png-only
```

Replace `tree` with `sph` for the spherical simulation version.

## Error-Model Variants

Simulation fixed-relative 5 percent and 20 percent variants:

```bash
./job_tngnh_fig2_glam.sh tree --run-label err05 --png-only --error-mode relative --relerr 0.05
./job_tngnh_fig6_glam.sh tree --run-label err05 --png-only --error-mode relative --relerr 0.05
./job_tngnh_fig7_glam.sh tree --run-label err05 --png-only --error-mode relative --relerr 0.05

./job_tngnh_fig2_glam.sh tree --run-label err20 --png-only --error-mode relative --relerr 0.20
./job_tngnh_fig6_glam.sh tree --run-label err20 --png-only --error-mode relative --relerr 0.20
./job_tngnh_fig7_glam.sh tree --run-label err20 --png-only --error-mode relative --relerr 0.20
```

Unweighted simulation variants:

```bash
./job_tngnh_fig2_glam.sh tree --run-label unweighted --png-only --error-mode unweighted
./job_tngnh_fig6_glam.sh tree --run-label unweighted --png-only --error-mode unweighted
./job_tngnh_fig7_glam.sh tree --run-label unweighted --png-only --error-mode unweighted
```

Replace `tree` with `sph` to make the corresponding spherical versions.

SPARC with the same fixed 10 percent scoring error model as the simulations:

```bash
./job_sparc_fig2_glam.sh --run-label sparc_err10 --png-only --error-mode relative --relerr 0.10
./job_sparc_fig6_glam.sh --run-label sparc_err10 --png-only --error-mode relative --relerr 0.10
./job_sparc_fig7_glam.sh --run-label sparc_err10 --png-only --error-mode relative --relerr 0.10
```

For SPARC Fig. 7, `sparc_err10` changes the scoring weights but keeps the
original SPARC observational uncertainties when constructing the pure-RAR and
SB-Jobs mock samples. To also use fixed 10 percent uncertainties in the mock
resampling, use:

```bash
./job_sparc_fig7_glam.sh \
  --run-label sparc_mockerr10 --png-only \
  --error-mode relative --relerr 0.10 \
  --mock-error-mode relative --mock-relerr 0.10
```

## SPARC Simulation-Style Fig. 7 Control

This is the SPARC-as-simulation control used to compare directly to TNG/NH
Fig. 7. It uses the simulation-compatible feature grid, fixed-relative 10
percent scoring by default, fixed features, pure-RAR mocks generated from the
fitted RARIF relation plus residual scatter, and the intrinsic fixed-feature
`Sigma_tot -> Jobs` control for `gbar`, `SB`, and `type`.

```bash
./job_sparc_fig7_glam.sh \
  --run-label sparc_simstyle --png-only \
  --sparc-sim-style
```

Output plot:

```text
../plots/gencomb_sharedET_sparc_simstyle.png
```

## Simulation-Specific Hyperparameter Variants

The default Fig. 6/7 simulation runs use the shared ExtraTrees hyperparameter
caches. The `simhyper` variants use caches optimized separately for each
simulation and baryonic-acceleration definition.

Generate the caches:

```bash
TRIALS=10000 NFOLDS=5 ./job_tngnh_hyperopt_glam.sh TNG sph
TRIALS=10000 NFOLDS=5 ./job_tngnh_hyperopt_glam.sh TNG tree
TRIALS=10000 NFOLDS=5 ./job_tngnh_hyperopt_glam.sh NH sph
TRIALS=10000 NFOLDS=5 ./job_tngnh_hyperopt_glam.sh NH tree
```

Use them in Fig. 6/7:

```bash
./job_tngnh_fig6_glam.sh tree \
  --run-label simhyper --png-only \
  --hyper-dir ../results/hyper_sim --hyper-scope sim

./job_tngnh_fig7_glam.sh tree \
  --run-label simhyper --png-only \
  --hyper-dir ../results/hyper_sim --hyper-scope sim
```

Replace `tree` with `sph` for the spherical variants.

## Batch Submission On Glamdring

The batch submitters create one wrapper per Slurm job and submit with
informative `addqueue -c` comments.

High-precision and error-model production batch:

```bash
cd /mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
QUEUE=cmb ./submit_new_production_glam.sh
```

Simulation-specific hyperparameter batch:

```bash
QUEUE=berg TRIALS=10000 NFOLDS=5 ./submit_tngnh_hyperopt_glam.sh
```

NH 539 tree refresh batch:

```bash
QUEUE=berg ./submit_tree_refresh_glam.sh all
QUEUE=berg ./submit_tree_refresh_glam.sh simhyper
```

`all` submits NH tree hyperopt plus baseline, hires, 5 percent, 20 percent, and
unweighted `gbar_tree` refreshes. `simhyper` submits the Fig. 6/7 tree refreshes
that use `../results/hyper_sim` with `--hyper-scope sim`.

## Important Options

- `--csv PATH`: switch from SPARC to TNG/NH mode and read the combined CSV.
- `--sim TNG|NH`: restrict a simulation run to one simulation; omitted means
  both TNG and NH are run and plotted.
- `--gbar-def sph|tree`: expose `gbar_sph` or `gbar_tree` as the fRAR-style
  `gbar` column.
- `--run-label LABEL`: suffix result and plot filenames to avoid overwriting.
- `--png-only`: save PNG only; omit for PNG plus PDF.
- `--error-mode default|observational|relative|unweighted`: choose the loss
  variance model. `default` means observational for SPARC and relative for
  TNG/NH.
- `--relerr X`: fixed relative error when `--error-mode relative`; the standard
  simulation value is `0.10`.
- `--mock-error-mode observational|relative`: SPARC Fig. 7 mock-resampling
  uncertainty model.
- `--mock-relerr X`: relative error for SPARC Fig. 7 mock resampling; defaults
  to `--relerr`.
- `--sparc-sim-style`: SPARC Fig. 7 control matching the simulation-style
  feature grid and mock construction.
- `--hyper-dir DIR` and `--hyper-scope shared|sim`: choose shared or
  simulation-specific ExtraTrees hyperparameter caches.
