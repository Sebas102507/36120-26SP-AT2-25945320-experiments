# Weather Intelligence — Experimentation Repository

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Data acquisition, exploratory analysis, feature engineering, model development and evaluation for
two machine learning services that forecast weather conditions in **Sydney, NSW**, using the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

| | |
|:--|:--|
| **Student name** | Juan Vargas T. |
| **Student ID** | 25945320 |
| **Subject** | 36120 Advanced Machine Learning Application — Assessment Task 2 |
| **Experiments repository** | https://github.com/Sebas102507/36120-26SP-AT2-25945320-experiments |
| **API repository** | https://github.com/Sebas102507/36120-26SP-AT2-25945320-api |
| **Live API** | https://three6120-26sp-at2-25945320-api.onrender.com (docs at `/docs`) |
| **Custom package** | [`nucleus-ml`](https://test.pypi.org/project/nucleus-ml/) on TestPyPI (module `nucleus_ml.weather`) |

---

## 1. Prediction targets

Both targets are derived from hourly observations at latitude −33.8688, longitude 151.2093, using
the formulas in the assignment brief. They are computed **before** any model is trained.

| Target | Type | Horizon | Selection metric |
|:--|:--|:--|:--|
| **Climate Comfort Index (CCI)**, 0–100 | Regression | +1, +2 and +3 days (one model per horizon) | Mean absolute error (MAE), averaged over horizons |
| **Weather Hazard Category (WHC)**: Low / Moderate / High / Extreme Risk | Multiclass classification | Exactly +7 days | Macro F1 |

Hourly values are aggregated to daily values before the formulas are applied: the daily mean for
temperature, humidity, wind speed and cloud cover, the daily total for precipitation and
snowfall, and the daily maximum for wind gusts.

## 2. Data and splits

| Period | Role |
|:--|:--|
| 2000-01-01 to 2024-12-31 | **Development (`train`)**: all model selection, via walk-forward cross-validation |
| 2025-01-01 to 2025-12-31 | **Test**: scored once, after the champion is chosen |
| 2026 onwards | **Production**: never used during the project, as required by the brief |

- **No separate validation split.** Model selection uses four expanding-window walk-forward
  folds inside the development split, validating on 2021, 2022, 2023 and 2024 in turn.
- **Embargo.** An embargo equal to the horizon (3 days for CCI, 7 for WHC) separates training
  from validation data in every fold and before the test year, so no training target falls
  inside a held-out period.
- **Data source.** Open-Meteo's default "Best Match" data combines ERA5/ERA5-Land reanalysis
  (used before 2017) with the 9 km ECMWF IFS model (used from 2017). The preprocessing notebook
  checks for a step change at 2017.

## 3. Model-development pipeline

| Stage | What happens | Where |
|:-:|:--|:--|
| 1 | Naive baselines (persistence, climatology / seasonal mode, majority class) | Experiment notebooks |
| 2 | Feature-family ablation, then Optuna hyperparameter tuning on all features, every trial scored by walk-forward CV | Experiment notebooks |
| 3 | Shortlist the best three tuned algorithms | Comparison notebooks |
| 4 | Feature-family selection per horizon with the settings fixed, then a short re-tune | Comparison notebooks |
| 5 | Champion: best mean CV score | Comparison notebooks |
| 6 | Final fit on all development data (2000–2024) | Comparison notebooks |
| 7 | One evaluation on the 2025 test year, champion against the baselines | Comparison notebooks |
| 8 | Promotion to `models/<task>/production/` with a drift reference, served by the API | Comparison notebooks |

Algorithms compared:

- **CCI:** dummy mean (null model), polynomial ridge regression, random forest, XGBoost and MLP.
- **WHC:** dummy majority class (null model), multinomial logistic regression, random forest,
  XGBoost and MLP.

All logic (target formulas, ingestion, preprocessing, feature engineering, the `Experiment`
workflow, comparison and promotion) lives in the custom package module `nucleus_ml.weather`, so
every notebook runs the same code.

## 4. Repository structure

Based on the Cookiecutter Data Science template.

```text
├── README.md                    <- This file
├── pyproject.toml               <- Dependencies (managed with uv); nucleus-ml comes from TestPyPI
├── uv.lock                      <- Exact dependency versions, for reproducibility
├── Makefile                     <- Convenience commands (make requirements, make test, make lint)
├── .github/workflows/
│   └── sync-production-models.yml  <- Copies production artefacts into the API repository
│
├── data/
│   ├── raw/                     <- sydney_hourly.csv, as downloaded from Open-Meteo
│   ├── interim/                 <- Daily aggregates and target inputs
│   └── processed/
│       ├── comfort_climate/     <- X_{train,test}.csv, y_{train,test}.csv, feature metadata
│       └── weather_hazard/      <- The same, plus aux_{train,test}.csv (WHI, error analysis only)
│
├── notebooks/
│   ├── ingestion/               <- 1. Download the hourly observations
│   ├── eda/                     <- 2. Exploratory data analysis
│   ├── preprocessing/           <- 3. Targets, features, leakage audit, train/test split
│   ├── comfort_climate/         <- 4. CCI experiment notebooks + 5. CCI comparison notebook
│   └── weather_hazard/          <- 4. WHC experiment notebooks + 5. WHC comparison notebook
│
├── models/
│   ├── preprocessing/           <- feature_pipeline.joblib: fitted pipeline, also used by the API
│   ├── comfort_climate/
│   │   ├── experiments/<name>/  <- One bundle per experiment (models, metadata, specification)
│   │   └── production/          <- Promoted CCI champion, copied to the API
│   └── weather_hazard/
│       ├── experiments/<name>/
│       └── production/          <- Promoted WHC champion, copied to the API
│
├── reports/
│   └── figures/                 <- Every figure produced by the notebooks, per task
│
├── 36120_26sp_at2_25945320_experiments/  <- Cookiecutter source module (helper scripts)
├── tests/                       <- Tests for the helper module
├── docs/                        <- Cookiecutter mkdocs skeleton
└── references/                  <- Reference material
```

Each bundle in `models/<task>/experiments/<name>/` contains:

| File | Contents |
|:--|:--|
| `<name>_h<horizon>.joblib` | Fitted model(s), one per horizon |
| `metadata.json` | Hyperparameters, features, importance and cross-validated scores |
| `manifest.json` | Model files and the feature column order |
| `spec.joblib` | Specification and tuned settings, used by the comparison notebook |

Each `production/` folder also contains `drift_reference.json` (the training distribution of every
feature used, for monitoring) and `promotion.json` (the audit record of how the champion was
chosen).

## 5. Setup

Requirements:

- Python 3.11 or newer (developed with Python 3.14);
- [uv](https://docs.astral.sh/uv/getting-started/installation/);
- internet access, for Open-Meteo and for installing `nucleus-ml` from TestPyPI.

```bash
git clone https://github.com/Sebas102507/36120-26SP-AT2-25945320-experiments.git
cd 36120-26SP-AT2-25945320-experiments
uv sync                     # or: make requirements
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

`uv sync` installs the exact versions in `uv.lock`. Only `nucleus-ml` is taken from TestPyPI;
every other package comes from PyPI (see `[tool.uv.sources]` in `pyproject.toml`).

To open the notebooks, either select the `.venv` interpreter as the kernel in VS Code, or start
JupyterLab from the environment:

```bash
uv run --with jupyterlab jupyter lab
```

## 6. Reproducing the results

Run the notebooks **top to bottom, in the order below**. Later notebooks only read artefacts that
earlier notebooks wrote. After setup (`uv sync` and `source .venv/bin/activate`), use the `.venv`
kernel.

CCI and WHC are independent after preprocessing: you can finish the whole CCI track (steps 4a then
5a) before starting WHC, or run the two experiment groups in parallel. Do **not** run a
comparison notebook until every experiment notebook for that task has finished.

```text
ingestion  →  eda (optional for models)  →  preprocessing
                                              ├─ CCI experiments (any order)  →  CCI comparison
                                              └─ WHC experiments (any order)  →  WHC comparison
```

### Step 1 — Ingestion

| Notebook | Reads | Writes |
|:--|:--|:--|
| `notebooks/ingestion/36120_26SP_AT2_25945320_data_ingestion.ipynb` | Open-Meteo API | `data/raw/sydney_hourly.csv` (2000–2025) |

Leave `REFRESH = False` to reuse the committed raw file. Set it to `True` only if you need a
fresh download. 2026 onwards is never downloaded: it is production data.

### Step 2 — Exploratory data analysis (optional for modelling)

| Notebook | Reads | Writes |
|:--|:--|:--|
| `notebooks/eda/36120_26SP_AT2_25945320_data_eda.ipynb` | `data/raw/sydney_hourly.csv` | figures under `reports/figures/` |

This notebook does not feed the models. Skip it if you only need to reproduce trained artefacts.

### Step 3 — Preprocessing (required before any experiment)

| Notebook | Reads | Writes |
|:--|:--|:--|
| `notebooks/preprocessing/36120_26SP_AT2_25945320_preprocessing.ipynb` | `data/raw/sydney_hourly.csv` | `data/interim/`, `data/processed/comfort_climate/`, `data/processed/weather_hazard/`, `models/preprocessing/feature_pipeline.joblib` |

This is the shared contract: daily aggregation, CCI and WHC targets, lagged/rolling features,
leakage audit, and the time split (development 2000–2024, test 2025). Every later notebook
assumes these files exist.

### Step 4a — CCI experiments (any order among themselves)

Each notebook trains one algorithm on `data/processed/comfort_climate/` and writes a bundle under
`models/comfort_climate/experiments/<name>/`. Run **all five** before the CCI comparison.

1. `notebooks/comfort_climate/36120_26SP_AT2_25945320_cci_dummy_mean_experiment.ipynb`
2. `notebooks/comfort_climate/36120_26SP_AT2_25945320_cci_polynomial_regression_experiment.ipynb`
3. `notebooks/comfort_climate/36120_26SP_AT2_25945320_cci_random_forest_experiment.ipynb`
4. `notebooks/comfort_climate/36120_26SP_AT2_25945320_cci_xgboost_experiment.ipynb`
5. `notebooks/comfort_climate/36120_26SP_AT2_25945320_cci_mlp_experiment.ipynb`

### Step 4b — WHC experiments (any order among themselves)

Same idea on `data/processed/weather_hazard/`, writing to
`models/weather_hazard/experiments/<name>/`. Run **all five** before the WHC comparison.

1. `notebooks/weather_hazard/36120_26SP_AT2_25945320_whc_dummy_experiment.ipynb`
2. `notebooks/weather_hazard/36120_26SP_AT2_25945320_whc_logistic_regression_experiment.ipynb`
3. `notebooks/weather_hazard/36120_26SP_AT2_25945320_whc_random_forest_experiment.ipynb`
4. `notebooks/weather_hazard/36120_26SP_AT2_25945320_whc_xgboost_experiment.ipynb`
5. `notebooks/weather_hazard/36120_26SP_AT2_25945320_whc_mlp_experiment.ipynb`

### Step 5a — CCI comparison and promotion

| Notebook | Reads | Writes |
|:--|:--|:--|
| `notebooks/comfort_climate/36120_26SP_AT2_25945320_cci_comparison.ipynb` | all CCI experiment bundles + processed CCI splits | `models/comfort_climate/production/` (`cci_model_*`) |

### Step 5b — WHC comparison and promotion

| Notebook | Reads | Writes |
|:--|:--|:--|
| `notebooks/weather_hazard/36120_26SP_AT2_25945320_whc_comparison.ipynb` | all WHC experiment bundles + processed WHC splits | `models/weather_hazard/production/` (`whc_model_*`) |

These two notebooks are the **only** place the 2025 test year is scored, and only after the
champion is chosen on development data. After both have run, the production folders are what the
API should serve (see section 7).

### Notes for a replicable rerun

- **Committed artefacts.** Raw data, processed tables and model bundles are in git, so you can
  start at step 4 (or 5) without repeating ingestion and preprocessing. If a comparison notebook
  reports a stale bundle, rerun that experiment notebook, then the comparison notebook.
- **Do not skip an experiment.** The comparison notebook ranks every bundle it finds. A missing
  dummy or MLP run changes the ranking and can change the promoted champion.
- **Seeds.** `RANDOM_STATE = 25945320` in `nucleus_ml.weather`. Leave it unchanged.
- **Runtime.** Every Optuna trial fits one model per walk-forward fold (four folds). For a
  quicker pass only, lower `N_TRIALS` in an experiment notebook, or `FEATURE_TRIALS` /
  `RETUNE_TRIALS` in a comparison notebook (for example 10 / 10 / 5). That will **not** match
  the submitted selections.
- **Progress.** `VERBOSE = 1` prints every step and trial; `0` prints results only.
- **Ingestion.** `REFRESH = False` reuses `data/raw/sydney_hourly.csv`.

## 7. Handing the models to the API

After step 5, the production artefacts are copied into the API repository in one of two ways:

- **Automatically:** a push to `main` that changes `models/preprocessing/` or a `production/`
  folder triggers `.github/workflows/sync-production-models.yml`, which commits the artefacts to
  the API repository's `models/` folder. This needs the repository secret `API_REPO_TOKEN`, a
  token with write access to the API repository.
- **Manually:** run `make artefacts` from the API repository (see its README).

## 8. Tests and code quality

```bash
make test     # pytest on tests/
make lint     # ruff format --check + ruff check
```

The analysis code itself (`nucleus_ml.weather`) is unit-tested in the package repository, and
the deployed API is tested end to end over HTTP in the API repository (`make test-api`).

## 9. Troubleshooting

| Symptom | Cause and fix |
|:--|:--|
| `uv sync` cannot find `nucleus-ml` | TestPyPI is unreachable or the version is not yet published. Check https://test.pypi.org/project/nucleus-ml/. |
| `FileNotFoundError` for `data/processed/...` | Run the preprocessing notebook (step 3) first. |
| A comparison notebook excludes a bundle as stale | The bundle was built on an older feature set. Rerun that experiment notebook. |
| `spec.joblib is missing` | The bundle was exported by an older package version. Rerun that experiment notebook. |
| Open-Meteo request fails or times out | Rerun the cell (requests are retried), or set `REFRESH = False` to use the saved CSV. |
| `ConvergenceWarning` from the MLP | Harmless: the trial is scored as usual; poorly converged settings simply score badly. |

## 10. Use of generative AI

Generative AI (Claude, Anthropic) was used to support coding, debugging, documentation and
report drafting. All generated content was reviewed, tested and validated by the student, who is
responsible for the submitted work.
