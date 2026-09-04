# Cursor agent instructions

Use this file as standing context for every session in this repository. It is UTS 36120 Advanced ML Assessment Task 2 (student 25945320): forecast Sydney weather targets from Open-Meteo observations.

Do not invent assignment formulas, weights, or evaluation rules. Quote them from the brief. If a decision is not in the brief, document it as a modelling choice and keep it reversible.

## Environment

- Work inside `.venv`. Activate with `source .venv/bin/activate`.
- Install deps from `pyproject.toml` (`pip install loguru mkdocs pandas pytest python-dotenv requests ruff tqdm typer`). `ipykernel` is already in the venv.
- Do not recreate the venv unless asked. It currently runs Python 3.14 even though `pyproject.toml` pins `~=3.10.0`.
- Do not `pip install -e .`. The package name starts with a digit (`36120_26sp_at2_25945320_experiments`), so an editable install fails with a SyntaxError. Import the local package by keeping the repo root on `sys.path`, or only use notebooks and scripts that do not need that import.

## Layout

Cookiecutter Data Science. Keep artefacts in the matching folder and do not overwrite raw files.

| Path | Role |
|---|---|
| `data/raw/` | Immutable Open-Meteo dumps. Never edit in place. |
| `data/interim/` | Daily aggregates, joined frames, intermediate joins. |
| `data/processed/` | Model-ready features and targets (no leakage). |
| `notebooks/` | Numbered exploration: `{n.n}-{initials}-{slug}.ipynb`. |
| `36120_26sp_at2_25945320_experiments/` | Reusable code promoted out of notebooks. |
| `reports/figures/` | Plots for the report. |
| `models/` | Trained artefacts only after a notebook has proven the recipe. |

Canonical ingestion notebook: `notebooks/ingestion/1.0-jsv-open-meteo-ingestion copy.ipynb`. Prefer renaming it to drop ` copy` over creating a third copy.

## What is already done

Hourly Sydney series for `2016-01-01` to `2025-12-31` are in `data/raw`. Leave 2026 onwards as unseen production data.

| File | Endpoint | Usable window | Notes |
|---|---|---|---|
| `data/raw/sydney_hourly_archive.csv` | `archive-api.open-meteo.com/v1/archive` | 2016-01-01 00:00 to 2025-12-31 23:00, 87,672 rows, 100% coverage | Historical Weather API / reanalysis. No `visibility`. Use for Climate Comfort Index (CCI). |
| `data/raw/sydney_hourly_forecast.csv` | `historical-forecast-api.open-meteo.com/v1/forecast` | 2016-12-31 11:00 to 2025-12-31 23:00; 78,901 rows with any data | Only source of `visibility` for Weather Hazard Category (WHC). Open-Meteo pads missing dates with nulls, so row counts are not coverage. |

Location used: Sydney `-33.8688, 151.2093`, timezone `Australia/Sydney`.

Do not re-download unless the user sets `REFRESH = True` or asks to change variables or dates. The ingestion notebook already has retries, column-completeness checks, DST irregularity warnings, and unit logging.

## Variables and units

Confirmed from the API (`snowfall` is centimetres, not millimetres):

- CCI inputs: `temperature_2m` (°C), `relative_humidity_2m` (%), `wind_speed_10m` (km/h), `cloud_cover` (%), `precipitation` (mm).
- WHC inputs: `precipitation` (mm), `wind_gusts_10m` (km/h), `visibility` (m), `snowfall` (cm), `temperature_2m` (°C).
- Candidate predictors only: `dew_point_2m`, `apparent_temperature`, `surface_pressure`, `cloud_cover_low`, `wind_direction_10m`, `weather_code`, `shortwave_radiation`.

`visibility` is forecast-only. Pending coordinator confirmation: either keep the forecast series or drop it and proxy visibility from humidity / dew point / low cloud. Do not silently drop visibility; the WHC formula weights it at 0.20.

## Findings the report must state

- **Visibility cap.** Observed visibility often sits at ~24,140 m and reaches 68,100 m. About 88% of non-null hours are above 20,000 m. The brief’s `VisibilityHazard = MIN((20000 - visibility) / 20000, 1)` therefore goes negative (min about -2.4) unless also clamped at 0. Choose literal vs `[0, 1]` and show the effect on the WHI distribution.
- **Snowfall.** Sydney max is 0.0 cm. `SnowHazard` is zero-variance but still carries a 0.10 weight.
- **DST.** Timestamps are naive local time. Check duplicate / skipped hours around October and April transitions before aggregating.
- **Forecast start.** WHC training cannot start before the first fully populated forecast timestamp (currently 2016-12-31 11:00, and `visibility` coverage is only 47.8% of the requested 2016–2025 grid).

## Leakage and target rules

These are hard constraints:

1. Aggregate hourly observations to daily before scoring. Brief constants (`RainScore` 20 mm, `RainHazard` 30 mm, gusts 100 km/h) are daily-scale. Computing the index hourly then averaging is not equivalent to computing it on daily aggregates — pick one, default to **daily aggregates**, and document it.
2. Build CCI and WHC from those aggregates, then shift targets by **+3 days (CCI)** and **+7 days (WHC)**.
3. Features may use day `t` and earlier only. No feature may use observations from the target day.
4. Split in time. Do not shuffle. Do not use 2026+ except as a final production hold-out if the brief requires it.

## Next work (do this unless the user asks otherwise)

1. New notebook `notebooks/features/2.0-jsv-daily-targets.ipynb` (or promote logic into `features.py` once stable).
2. Read the two raw CSVs; do not mutate them.
3. Daily aggregation with an explicit dictionary (e.g. sum precip/snowfall, max gusts, mean temp/humidity/wind/cloud, min visibility). Write to `data/interim/`.
4. Implement CCI and WHC exactly as the brief specifies, including the visibility clamp decision above.
5. Shift targets +3 / +7. Drop rows whose target falls outside the usable window.
6. Lagged and rolling features from `t` backward only. Write the model table to `data/processed/`.
7. Then modelling notebooks / `modeling/train.py`: time-based split, baseline first, then the models the brief asks for. Save figures under `reports/figures/`.

## Coding conventions

- Python 3, pandas, pathlib. Match the ingestion notebook style: dataclasses for config, typed signatures, short documented functions, `logging` or `loguru`.
- Ruff line length 99 for package code.
- Prefer a new numbered notebook over editing a finished one, unless the user is iterating on that notebook.
- Do not commit `.venv`, raw CSV dumps unless the user asks, or secrets.
- Keep the report-facing narrative in notebook markdown (units, windows, decisions). The written assignment will quote those cells.

## When stuck

Re-read the ingestion notebook findings cell and this file before changing data sources or target formulas. Ask before replacing `visibility` with a proxy or before refetching Open-Meteo.



## Repository Structure

Repository 1 - Experimentation Repository
This repository is used during data acquisition, exploratory data analysis, feature engineering, model development, experimentation, evaluation and reporting.

Expected Structure
project/

├── notebooks/
│
├── models/
│
├── reports/
│
├── pyproject.toml
│
└── README.md
Requirements
Private Github repository.
Use the Cookiecutter Data Science project structure.
Create subfolders for:
notebooks/comfort_climate
notebooks/weather_hazard
models/comfort_climate
models/weather_hazard
Store all notebooks in the appropriate notebook folders.
Store final model artefacts in the appropriate model folders.
Store experiment reports in the reports/ folder.
Provide setup instructions in the README.md.
Include student name and id in README.md.


## Target Variable Requirements
You are required to create two custom weather-based target variables using historical weather observations obtained from the Open-Meteo API (https://open-meteo.com/en/docs/historical-weather-api#json_return_objectLinks to an external site.). You can decide which information from this API can be used as features for your model. Be wary of any data leaks that can be fed to your models during training.

These target variables must be calculated before training machine learning models and will serve as the prediction targets for the project. You can use any data prior to 2026 as training, validation and test sets. All the data from 2026 onwards should be used as data in production (i.e. not accessible during the project).

Important
Target 1 (CCI) is a regression problem.
Target 2 (WHC) is a multiclass classification problem.
Students are responsible for creating both target variables before model development.
The target variables must be derived using the formulas defined below.
The machine learning models must predict future values of these derived targets rather than raw weather measurements.
All target variables must be generated using historical weather observations for the following location:

Parameter	Value
City	Sydney, NSW, Australia
Latitude	-33.8688
Longitude	151.2093
Students must retrieve historical weather observations corresponding to this location using the Open-Meteo Historical Weather API.

 

🟢 Target 1 – Climate Comfort Index (CCI)
Prediction Type: Regression

Prediction Horizon: Next 3 Days

Output Range: 0 – 100

The Climate Comfort Index (CCI) measures how suitable weather conditions are for outdoor human activities. Higher values indicate more comfortable conditions.

Required Weather Variables
temperature_2m
relative_humidity_2m
wind_speed_10m
cloud_cover
precipitation
Step 1 – Calculate Component Scores
Score	Formula
Temperature Score	TempScore = MAX(0, 1 - ABS(temperature_2m - 22) / 20)
Humidity Score	HumidityScore = MAX(0, 1 - ABS(relative_humidity_2m - 50) / 50)
Wind Score	WindScore = MAX(0, 1 - ABS(wind_speed_10m - 10) / 40)
Cloud Score	CloudScore = MAX(0, 1 - cloud_cover / 100)
Rain Score	RainScore = MAX(0, 1 - precipitation / 20)
Step 2 – Calculate Final Climate Comfort Index
CCI = 100 × (

  0.35 × TempScore +

  0.20 × HumidityScore +

  0.15 × WindScore +

  0.15 × CloudScore +

  0.15 × RainScore

)
Interpretation
CCI Range	Interpretation
80 - 100	Excellent outdoor conditions
60 - 79	Comfortable conditions
40 - 59	Moderate comfort
20 - 39	Poor comfort
0 - 19	Very poor comfort
 

The prediction granularity for this assignment is daily. Students should construct daily observations before deriving the target variables:
Variable	Daily Aggregation
Temperature	Daily Mean
Relative Humidity
Daily Mean
Wind Speed	Daily Mean
Cloud Cover
Daily Mean
Precipitation	Daily Total
 

 

🔴 Target 2 – Weather Hazard Category (WHC)
Prediction Type: Multiclass Classification

Prediction Horizon: Exactly +7 Days

Output Classes: Low Risk • Moderate Risk • High Risk • Extreme Risk

The Weather Hazard Category (WHC) measures the severity of weather conditions and their potential impact on people, infrastructure and transport operations.

Required Weather Variables
precipitation
wind_gusts_10m
cloud_cover
snowfall
temperature_2m
Step 1 – Calculate Hazard Scores
Hazard Score	Formula
Rain Hazard	RainHazard = MIN(precipitation / 30, 1)
Wind Hazard	WindHazard = MIN(wind_gusts_10m / 100, 1)
Cloud Hazard	
CloudHazard = MIN(CloudCover / 100, 1)
Snow Hazard	SnowHazard = MIN(snowfall / 15, 1)
Temperature Hazard	TempHazard = MIN(ABS(temperature_2m - 22) / 25, 1)
Step 2 – Calculate Weather Hazard Index
WHI = 100 × (

  0.30 × RainHazard +

  0.30 × WindHazard +

  0.20 × CloudHazard +

  0.10 × SnowHazard +

  0.10 × TempHazard

)
Step 3 – Convert WHI into Classification Labels
Condition	Weather Hazard Category	Class
WHI < 25	🟢 Low Risk	0
25 ≤ WHI < 50	🟡 Moderate Risk	1
50 ≤ WHI < 75	🟠 High Risk	2
WHI ≥ 75	🔴 Extreme Risk	3
 

The prediction granularity for this assignment is daily. Students should construct daily observations before deriving the target variables:
Variable	Daily Aggregation
Precipitation	Daily Total
Wind Gusts	Daily Maximum
Visibility	Daily Minimum
Snowfall	Daily Total
Temperature	Daily Mean
Machine Learning Target
Students must train their classification model to predict the Weather Hazard Category (WHC) labels rather than the continuous Weather Hazard Index (WHI).