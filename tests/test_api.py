"""Integration tests for the deployed Weather Intelligence API.

These tests call a running instance of the API over HTTP and check that every endpoint returns
the structure and values required by the assessment brief. By default they target the live
Render deployment; point them elsewhere with ``API_BASE_URL``::

    python -m pytest tests/test_api.py -v                                    # live service
    API_BASE_URL=http://localhost:8080 python -m pytest tests/test_api.py -v # local instance
    make test-api                                                            # live, must be up

If the service cannot be reached, every test here is skipped, so ``make test`` still passes
offline. Set ``API_TESTS_REQUIRED=1`` to turn an unreachable service into a failure (``make
test-api`` does this).

The free Render instance sleeps when idle, so the first request waits up to two minutes for it to
wake. Prediction responses are fetched once per date and shared between tests, to keep the
number of calls (each of which downloads weather history from Open-Meteo) small.
"""

from __future__ import annotations

from datetime import date, timedelta
import os
import time

import pytest
import requests

BASE_URL = os.environ.get(
    "API_BASE_URL", "https://three6120-26sp-at2-25945320-api.onrender.com"
).rstrip("/")
TESTS_REQUIRED = os.environ.get("API_TESTS_REQUIRED") == "1"

WAKE_UP_SECONDS = 120  # a sleeping free Render instance can take about a minute to start
REQUEST_TIMEOUT = 90  # a prediction downloads ~35 days of hourly history from Open-Meteo

REQUIRED_ENDPOINTS = [
    "/",
    "/health",
    "/predict/index/comfort_climate",
    "/predict/category/weather_hazard",
    "/model-metadata",
]
WHC_LABELS = {"Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"}
METADATA_FIELDS = {
    "target",
    "prediction_type",
    "algorithm",
    "forecast_horizon",
    "features",
    "feature_importance",
    "performance_metrics",
}
DRIFT_FIELDS = {
    "target",
    "experiment",
    "status",
    "rows_observed",
    "features_checked",
    "drifted_features",
}
DRIFT_STATUSES = ("ok", "drift detected", "insufficient data", "unavailable")

# Dates inside the supported range, in different seasons (summer and winter in Sydney).
SAMPLE_DATES = ["2025-01-01", "2024-07-15"]

pytestmark = pytest.mark.api


# ----------------------------------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api() -> requests.Session:
    """An HTTP session to a running API, or skip the module if none can be reached."""
    session = requests.Session()
    deadline = time.monotonic() + WAKE_UP_SECONDS
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            response = session.get(f"{BASE_URL}/health", timeout=30)
            if response.status_code == 200:
                return session
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as error:
            last_error = str(error)
        time.sleep(5)

    message = f"API at {BASE_URL} is unreachable ({last_error})."
    if TESTS_REQUIRED:
        pytest.fail(message)
    pytest.skip(message)


def get(api: requests.Session, path: str, **params) -> requests.Response:
    """GET ``path`` on the API under test."""
    return api.get(f"{BASE_URL}{path}", params=params or None, timeout=REQUEST_TIMEOUT)


def json_ok(response: requests.Response):
    """Assert a 200 JSON response and return its body, showing the body if it is not."""
    assert response.status_code == 200, (
        f"{response.request.method} {response.url} -> {response.status_code}: {response.text}"
    )
    assert response.headers["content-type"].startswith("application/json")
    return response.json()


def next_days(day: str, *offsets: int) -> list[str]:
    """ISO dates ``offset`` days after ``day``."""
    start = date.fromisoformat(day)
    return [(start + timedelta(days=offset)).isoformat() for offset in offsets]


@pytest.fixture(scope="module")
def overview(api):
    return json_ok(get(api, "/"))


@pytest.fixture(scope="module")
def metadata(api):
    return json_ok(get(api, "/model-metadata"))


@pytest.fixture(scope="module")
def cci_forecasts(api):
    """CCI responses for every sample date, fetched once."""
    return {day: json_ok(get(api, "/predict/index/comfort_climate", date=day))
            for day in SAMPLE_DATES}


@pytest.fixture(scope="module")
def whc_forecasts(api):
    """WHC responses for every sample date, fetched once."""
    return {day: json_ok(get(api, "/predict/category/weather_hazard", date=day))
            for day in SAMPLE_DATES}


# ----------------------------------------------------------------------------------------------
# GET / : project overview
# ----------------------------------------------------------------------------------------------


def test_root_has_the_required_fields(overview):
    for field in ("project", "description", "version", "github", "endpoints"):
        assert field in overview, f"missing field: {field}"
    for field in ("project", "description", "version", "github"):
        assert isinstance(overview[field], str) and overview[field].strip()


def test_root_links_to_the_github_repository(overview):
    assert overview["github"].startswith("https://github.com/")


def test_root_lists_every_required_endpoint(overview):
    missing = set(REQUIRED_ENDPOINTS) - set(overview["endpoints"])
    assert not missing, f"endpoints missing from /: {sorted(missing)}"


def test_root_describes_both_prediction_targets(overview):
    targets = overview["prediction_targets"]
    assert len(targets) == 2
    endpoints = " ".join(target["endpoint"] for target in targets)
    assert "/predict/index/comfort_climate" in endpoints
    assert "/predict/category/weather_hazard" in endpoints
    for target in targets:
        assert {"target", "endpoint", "prediction_type", "forecast_horizon"} <= set(target)


# ----------------------------------------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------------------------------------


def test_health_reports_healthy(api):
    body = json_ok(get(api, "/health"))
    assert body["status"] == "healthy"
    assert isinstance(body["message"], str) and body["message"].strip()


# ----------------------------------------------------------------------------------------------
# GET /predict/index/comfort_climate
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("day", SAMPLE_DATES)
def test_cci_response_has_the_required_structure(cci_forecasts, day):
    body = cci_forecasts[day]
    assert set(body) == {"input_date", "predictions"}
    assert body["input_date"] == day
    assert set(body["predictions"]) == {"comfort_climate"}


@pytest.mark.parametrize("day", SAMPLE_DATES)
def test_cci_predicts_the_next_three_days_in_order(cci_forecasts, day):
    forecast = cci_forecasts[day]["predictions"]["comfort_climate"]
    assert list(forecast) == next_days(day, 1, 2, 3)


@pytest.mark.parametrize("day", SAMPLE_DATES)
def test_cci_values_are_whole_numbers_between_0_and_100(cci_forecasts, day):
    for value in cci_forecasts[day]["predictions"]["comfort_climate"].values():
        assert isinstance(value, int) and not isinstance(value, bool)
        assert 0 <= value <= 100


def test_cci_is_deterministic(api, cci_forecasts):
    day = SAMPLE_DATES[0]
    again = json_ok(get(api, "/predict/index/comfort_climate", date=day))
    assert again == cci_forecasts[day]


# ----------------------------------------------------------------------------------------------
# GET /predict/category/weather_hazard
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("day", SAMPLE_DATES)
def test_whc_response_has_the_required_structure(whc_forecasts, day):
    body = whc_forecasts[day]
    assert set(body) == {"input_date", "predictions"}
    assert body["input_date"] == day
    assert set(body["predictions"]) == {"weather_hazard"}


@pytest.mark.parametrize("day", SAMPLE_DATES)
def test_whc_predicts_exactly_seven_days_ahead(whc_forecasts, day):
    forecast = whc_forecasts[day]["predictions"]["weather_hazard"]
    assert list(forecast) == next_days(day, 7)


@pytest.mark.parametrize("day", SAMPLE_DATES)
def test_whc_value_is_one_of_the_four_categories(whc_forecasts, day):
    (category,) = whc_forecasts[day]["predictions"]["weather_hazard"].values()
    assert category in WHC_LABELS


# ----------------------------------------------------------------------------------------------
# GET /model-metadata
# ----------------------------------------------------------------------------------------------


def test_metadata_describes_both_models(metadata):
    assert isinstance(metadata, list) and len(metadata) == 2
    targets = " ".join(str(entry["target"]).lower() for entry in metadata)
    assert "comfort" in targets and "hazard" in targets


def test_metadata_entries_have_every_required_field(metadata):
    for entry in metadata:
        missing = METADATA_FIELDS - set(entry)
        assert not missing, f"{entry.get('target')}: missing {sorted(missing)}"


def test_metadata_prediction_types_and_horizons(metadata):
    by_type = {entry["prediction_type"]: entry for entry in metadata}
    assert set(by_type) == {"regression", "classification"}
    for horizon in ("1", "2", "3"):
        assert horizon in by_type["regression"]["forecast_horizon"]
    assert "7" in by_type["classification"]["forecast_horizon"]


def test_metadata_features_importance_and_metrics_are_well_formed(metadata):
    for entry in metadata:
        assert isinstance(entry["algorithm"], str) and entry["algorithm"].strip()
        assert entry["features"], "feature list is empty"
        assert all(isinstance(name, str) for name in entry["features"])
        assert all(isinstance(value, (int, float))
                   for value in entry["feature_importance"].values())
        assert entry["performance_metrics"], "no performance metrics"
        assert all(isinstance(value, (int, float))
                   for value in entry["performance_metrics"].values())


def test_metadata_reports_each_tasks_selection_metric(metadata):
    by_type = {entry["prediction_type"]: entry for entry in metadata}
    assert "mae" in by_type["regression"]["performance_metrics"]
    assert "macro_f1" in by_type["classification"]["performance_metrics"]


# ----------------------------------------------------------------------------------------------
# GET /monitoring/drift (extra endpoint)
# ----------------------------------------------------------------------------------------------


def test_drift_reports_each_model(api, cci_forecasts):
    # cci_forecasts ensures at least a few rows have been served before the check.
    reports = json_ok(get(api, "/monitoring/drift"))
    assert isinstance(reports, list) and len(reports) == 2
    for report in reports:
        assert DRIFT_FIELDS <= set(report)
        assert report["status"].startswith(DRIFT_STATUSES), report["status"]
        assert isinstance(report["rows_observed"], int) and report["rows_observed"] >= 0
        assert isinstance(report["drifted_features"], list)


# ----------------------------------------------------------------------------------------------
# Invalid requests: clear errors with the right status codes
# ----------------------------------------------------------------------------------------------

PREDICTION_PATHS = ["/predict/index/comfort_climate", "/predict/category/weather_hazard"]


@pytest.mark.parametrize("path", PREDICTION_PATHS)
@pytest.mark.parametrize("bad_date", ["01-01-2025", "2025/01/01", "2025-02-30", "tomorrow"])
def test_malformed_dates_return_400(api, path, bad_date):
    response = get(api, path, date=bad_date)
    assert response.status_code == 400, response.text
    assert "YYYY-MM-DD" in response.json()["detail"]


@pytest.mark.parametrize("path", PREDICTION_PATHS)
@pytest.mark.parametrize("out_of_range", ["1990-01-01", "2099-01-01"])
def test_dates_outside_the_supported_range_return_400(api, path, out_of_range):
    response = get(api, path, date=out_of_range)
    assert response.status_code == 400, response.text
    assert "range" in response.json()["detail"]


@pytest.mark.parametrize("path", PREDICTION_PATHS)
def test_missing_date_returns_422(api, path):
    response = get(api, path)
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_unknown_path_returns_404(api):
    assert get(api, "/predict/unknown").status_code == 404


@pytest.mark.parametrize("path", PREDICTION_PATHS)
def test_wrong_method_returns_405(api, path):
    response = api.post(f"{BASE_URL}{path}", params={"date": SAMPLE_DATES[0]},
                        timeout=REQUEST_TIMEOUT)
    assert response.status_code == 405


# ----------------------------------------------------------------------------------------------
# Self-documentation
# ----------------------------------------------------------------------------------------------


def test_openapi_schema_documents_every_required_endpoint(api):
    schema = json_ok(get(api, "/openapi.json"))
    missing = set(REQUIRED_ENDPOINTS) - set(schema["paths"])
    assert not missing, f"endpoints missing from /openapi.json: {sorted(missing)}"


def test_interactive_docs_are_served(api):
    response = get(api, "/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
