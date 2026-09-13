"""Shared experiment machinery for the 36120 AT2 weather intelligence project.

Every experiment notebook — five for the Climate Comfort Index regression and five for the
Weather Hazard Category classification — follows the same ten-section structure and calls the
functions below to do it. Keeping the shared work here rather than in each notebook means a
change to the metric definition, the baseline construction or the artefact layout happens once
and applies everywhere, and it keeps each notebook to the part that is genuinely specific to its
algorithm: the estimator, the search space and the interpretation.

Student 25945320.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

__all__ = [
    "BalancedXGBClassifier",
    "project_root",
    "load_task",
    "Task",
    "regression_metrics",
    "classification_metrics",
    "climatology_profile",
    "regression_baselines",
    "OrdinalLabels",
    "hazard_class_from_whi",
    "per_class_report",
    "build_metadata",
    "classification_baselines",
    "cumulative_feature_sets",
    "run_ablation",
    "optimise",
    "study_frame",
    "plot_ablation",
    "plot_optuna",
    "plot_regression_validation",
    "plot_classification_validation",
    "plot_importance",
    "export_experiment",
]

# --------------------------------------------------------------------------------------------
# Task configuration
# --------------------------------------------------------------------------------------------

TASK_CONFIG = {
    "comfort_climate": {
        "kind": "regression",
        "metadata_file": "cci_feature_metadata.json",
        "target_name": "Climate Comfort Index",
        "persistence_column": "cci",
        "production_stem": "cci_model",
    },
    "weather_hazard": {
        "kind": "classification",
        "metadata_file": "whc_feature_metadata.json",
        "target_name": "Weather Hazard Category",
        "persistence_column": "whi",
        "production_stem": "whc_model",
    },
}

# The order in which engineered families are added during the ablation. Families with the most
# prior justification come first, so each step asks "does this add anything beyond what is
# already present?" rather than crediting a family for signal an earlier one would have supplied.
GROUP_ORDER = [
    "same_day_observations",
    "same_day_indices",
    "same_day_scores",
    "lags",
    "rolling_statistics",
    "rolling_precipitation",
    "rolling_gusts",
    "deltas",
    "seasonal",
]


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost multiclass classifier that optionally applies balanced sample weights.

    This class is module-scoped so exported Joblib artefacts can be loaded from comparison and
    deployment processes, rather than depending on a notebook's transient ``__main__`` module.
    """

    def __init__(self, balance_classes: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.balance_classes = balance_classes

    def fit(self, X, y, **kwargs):  # noqa: N803 - scikit-learn estimator signature
        if self.balance_classes and "sample_weight" not in kwargs:
            kwargs["sample_weight"] = compute_sample_weight("balanced", y)
        return super().fit(X, y, **kwargs)


def project_root(start: Path | None = None) -> Path:
    """Return the project directory containing ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project root from the current directory.")


@dataclass
class Task:
    """Everything an experiment notebook needs to know about the task it is running.

    Resolves the directory layout, loads the preprocessing metadata and exposes the feature and
    target column contracts, so no notebook hard-codes a path or a column list.
    """

    name: str
    kind: str
    root: Path
    data_dir: Path
    model_dir: Path
    figure_dir: Path
    metadata: dict
    feature_columns: list[str]
    feature_groups: dict[str, list[str]]
    target_columns: list[str]
    horizons: list[int]
    target_name: str
    persistence_column: str
    class_labels: dict[int, str] | None = None
    auxiliary_columns: list[str] = field(default_factory=list)
    production_stem: str = "model"

    @property
    def is_classification(self) -> bool:
        return self.kind == "classification"

    def family_of(self, column: str) -> str:
        """Return the engineered family a feature column belongs to."""
        for family, columns in self.feature_groups.items():
            if column in columns:
                return family
        return "unknown"

    def ordered_groups(self) -> list[str]:
        """Feature families in ablation order, with any unlisted family appended."""
        ordered = [group for group in GROUP_ORDER if group in self.feature_groups]
        return ordered + [g for g in self.feature_groups if g not in ordered]

    def save_figure(self, fig, name: str) -> Path:
        """Write a figure into this task's report figure directory."""
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        path = self.figure_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def load_xy(self, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load one split's feature and target files and confirm they align by date."""
        def read(prefix: str) -> pd.DataFrame:
            path = self.data_dir / f"{prefix}_{split}.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found. Run the preprocessing notebook first."
                )
            return (
                pd.read_csv(path, parse_dates=["feature_date"])
                .set_index("feature_date")
                .sort_index()
            )

        features, targets = read("X"), read("y")
        if not features.index.equals(targets.index):
            raise AssertionError(f"{split}: X and y indices do not match.")
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"{split}: X is missing {len(missing)} column(s), e.g. {missing[:5]}")
        return features[self.feature_columns], targets

    def load_aux(self, split: str) -> pd.DataFrame:
        """Load the auxiliary file, used for hazard error analysis only."""
        path = self.data_dir / f"aux_{split}.csv"
        return (
            pd.read_csv(path, parse_dates=["feature_date"])
            .set_index("feature_date")
            .sort_index()
        )


def load_task(name: str, root: Path | None = None) -> Task:
    """Build a :class:`Task` for ``comfort_climate`` or ``weather_hazard``."""
    if name not in TASK_CONFIG:
        raise ValueError(f"Unknown task {name!r}; expected one of {sorted(TASK_CONFIG)}")
    config = TASK_CONFIG[name]
    root = root or project_root()
    data_dir = root / "data" / "processed" / name
    metadata = json.loads((data_dir / config["metadata_file"]).read_text())

    class_labels = None
    if "class_labels" in metadata:
        class_labels = {int(k): v for k, v in metadata["class_labels"].items()}

    task = Task(
        name=name,
        kind=config["kind"],
        root=root,
        data_dir=data_dir,
        model_dir=root / "models" / name / "experiments",
        figure_dir=root / "reports" / "figures" / name,
        metadata=metadata,
        feature_columns=list(metadata["feature_columns"]),
        feature_groups={k: list(v) for k, v in metadata["feature_groups"].items()},
        target_columns=list(metadata["target_columns"]),
        horizons=[int(h) for h in metadata["horizons_days"]],
        target_name=config["target_name"],
        persistence_column=config["persistence_column"],
        class_labels=class_labels,
        auxiliary_columns=list(metadata.get("auxiliary_columns", [])),
        production_stem=config["production_stem"],
    )
    task.model_dir.mkdir(parents=True, exist_ok=True)
    task.figure_dir.mkdir(parents=True, exist_ok=True)
    return task


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R2 and prediction spread, all on the 0-100 CCI scale.

    ``pred_std`` is reported alongside the error metrics because it is the direct test for a
    model that has retreated to predicting the mean: a spread far below the observed spread means
    the model is declining to commit, which can produce a respectable MAE while being useless for
    identifying the uncomfortable days a user actually cares about.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pred_std": float(np.std(y_pred)),
    }


def classification_metrics(y_true, y_pred, labels: Sequence[int]) -> dict[str, float]:
    """Accuracy, balanced accuracy, macro and weighted F1, and mean ordinal error.

    ``macro_f1`` is the headline: it averages F1 across classes without weighting by frequency,
    so a model that ignores the rare hazard classes cannot hide behind the majority class.
    ``mean_ordinal_error`` exploits the fact that the classes are ordered — predicting Low when
    the truth is Extreme is a worse error than predicting Moderate, and a plain classification
    metric cannot see that difference.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro",
                                       zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels),
                                          average="weighted", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, labels=list(labels),
                                               average="macro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, labels=list(labels),
                                                     average="macro", zero_division=0)),
            "mean_ordinal_error": float(np.mean(np.abs(y_true - y_pred))),
        }


def per_class_report(y_true, y_pred, labels: Sequence[int],
                     class_labels: dict[int, str]) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support, including classes with no support."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = pd.DataFrame({
            "category": [class_labels.get(c, str(c)) for c in labels],
            "precision": precision_score(y_true, y_pred, labels=list(labels), average=None,
                                         zero_division=0),
            "recall": recall_score(y_true, y_pred, labels=list(labels), average=None,
                                   zero_division=0),
            "f1": f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0),
            "support": [int((y_true == c).sum()) for c in labels],
            "predicted": [int((y_pred == c).sum()) for c in labels],
        }, index=pd.Index(labels, name="class"))
    return frame


# --------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------

def climatology_profile(series: pd.Series, window: int = 15) -> pd.Series:
    """Mean of ``series`` by day of year, smoothed with a wrapped centred window.

    Centring is acceptable here because the smoothing is applied to a day-of-year profile
    derived solely from training data, not to the time series itself. No future observation
    enters any prediction.
    """
    profile = series.groupby(series.index.dayofyear).mean().reindex(range(1, 367))
    profile = profile.interpolate(limit_direction="both")
    wrapped = pd.concat([profile, profile, profile])
    smoothed = wrapped.rolling(window, center=True, min_periods=1).mean()
    return smoothed.iloc[len(profile):2 * len(profile)].set_axis(profile.index)


def regression_baselines(task: Task, X: dict, y: dict) -> dict[str, dict[str, np.ndarray]]:
    """Persistence and climatology forecasts for every horizon, on the validation split.

    Persistence carries today's CCI forward unchanged; it uses the most recent observation and
    nothing else. Climatology predicts the training-set seasonal average; it uses the calendar
    and nothing else. The two are complementary, so beating both means the model has combined
    them, which is the minimum bar for a useful forecast.
    """
    profiles = {
        target: climatology_profile(y["train"][target].set_axis(X["train"].index))
        for target in task.target_columns
    }
    baselines = {}
    for target in task.target_columns:
        index = X["val"].index
        baselines[target] = {
            "persistence": X["val"][task.persistence_column].to_numpy(),
            "climatology": profiles[target].reindex(index.dayofyear).to_numpy(),
        }
    return baselines


class OrdinalLabels:
    """Map the hazard classes observed in training onto contiguous codes and back.

    XGBoost requires class codes ``0..K-1`` and raises if any are missing, which would happen if
    a hazard class never occurred in the training period. Encoding every classifier the same way
    keeps the notebooks identical in structure and means a rerun on a different archive window
    cannot fail for one algorithm and not another.
    """

    def __init__(self, labels) -> None:
        self.classes_ = sorted(int(v) for v in pd.unique(np.asarray(labels)))
        self._forward = {label: code for code, label in enumerate(self.classes_)}
        self._backward = {code: label for label, code in self._forward.items()}

    @property
    def contiguous(self) -> bool:
        """True when the observed classes already are ``0..K-1``."""
        return self.classes_ == list(range(len(self.classes_)))

    def encode(self, labels) -> np.ndarray:
        values = np.asarray(labels, dtype=int)
        unknown = set(np.unique(values)) - set(self._forward)
        if unknown:
            raise ValueError(f"Labels {sorted(unknown)} were never seen during training.")
        return np.array([self._forward[int(v)] for v in values], dtype=int)

    def decode(self, codes) -> np.ndarray:
        return np.array([self._backward[int(c)] for c in np.asarray(codes, dtype=int)], dtype=int)


def hazard_class_from_whi(whi) -> np.ndarray:
    """Apply the brief's WHI thresholds to produce hazard class labels 0-3."""
    return np.digitize(np.asarray(whi, dtype=float), bins=[25, 50, 75], right=False)


def classification_baselines(task: Task, X: dict, y: dict) -> dict[str, np.ndarray]:
    """Majority-class, persistence and seasonal-mode forecasts on the validation split.

    Persistence carries the hazard class observed on day *t* forward to *t*+7. The class on *t*
    is not itself a feature, but it follows directly from the ``whi`` feature by applying the
    brief's thresholds, so the rule needs no extra inputs.
    """
    target = task.target_columns[0]
    train_labels = y["train"][target]
    majority = int(train_labels.mode().iloc[0])

    monthly_mode = train_labels.set_axis(X["train"].index).groupby(
        X["train"].index.month
    ).agg(lambda s: int(s.mode().iloc[0]))

    index = X["val"].index
    return {
        "majority_class": np.full(len(index), majority, dtype=int),
        "persistence": hazard_class_from_whi(X["val"][task.persistence_column]),
        "seasonal_mode": index.month.map(monthly_mode).to_numpy(dtype=int),
    }


# --------------------------------------------------------------------------------------------
# Feature-group ablation
# --------------------------------------------------------------------------------------------

def cumulative_feature_sets(task: Task, seed_label: str | None = None) -> dict[str, list[str]]:
    """Return cumulative feature sets, one per engineered family, in ablation order.

    The sequence starts from persistence alone — a single column, the index observed today — and
    adds one family at a time, so the contribution of each block of engineering is measured
    rather than assumed.
    """
    seed_label = seed_label or f"persistence only ({task.persistence_column})"
    configs = {seed_label: [task.persistence_column]}
    cumulative: list[str] = []
    for group in task.ordered_groups():
        cumulative = cumulative + task.feature_groups[group]
        configs[f"+ {group}"] = list(dict.fromkeys(cumulative))
    return configs


def run_ablation(configs: dict[str, list[str]],
                 score: Callable[[list[str], str], dict[str, float]],
                 targets: Sequence[str]) -> pd.DataFrame:
    """Fit and score one model per (feature set, target) pair.

    Parameters
    ----------
    configs
        Mapping of label to feature column list, from :func:`cumulative_feature_sets`.
    score
        Callable taking ``(columns, target)`` and returning a metric dictionary.
    targets
        Target columns to evaluate.
    """
    rows = []
    for label, columns in configs.items():
        for target in targets:
            rows.append({"config": label, "n_features": len(columns), "target": target,
                         **score(columns, target)})
    return pd.DataFrame(rows)


def plot_ablation(task: Task, ablation: pd.DataFrame, metric: str, targets: Sequence[str],
                  name: str, higher_is_better: bool, reference: dict | None = None):
    """Plot cumulative ablation performance and the increment each family contributed."""
    table = ablation.pivot(index="config", columns="target", values=metric)
    order = list(dict.fromkeys(ablation["config"]))
    table = table.reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    table[list(targets)].plot(marker="o", ax=axes[0])
    for label, value in (reference or {}).items():
        axes[0].axhline(value, ls="--", lw=0.9, color="grey")
        axes[0].text(0, value, f" {label}", fontsize=6, color="grey", va="bottom")
    direction = "higher is better" if higher_is_better else "lower is better"
    axes[0].set(title=f"Validation {metric} as families are added ({direction})",
                ylabel=metric, xlabel="")
    axes[0].tick_params(axis="x", labelrotation=35, labelsize=7)
    axes[0].legend(fontsize=8)

    increment = table[list(targets)].diff()
    if not higher_is_better:
        increment = -increment
    increment.plot.bar(ax=axes[1], width=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title=f"Improvement in {metric} contributed by each family",
                ylabel=f"{metric} improvement", xlabel="")
    axes[1].tick_params(axis="x", labelrotation=35, labelsize=7)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    task.save_figure(fig, name)
    return fig, table


# --------------------------------------------------------------------------------------------
# Optuna
# --------------------------------------------------------------------------------------------

def optimise(objective: Callable, n_trials: int, direction: str, seed: int,
             study_name: str, show_progress: bool = False):
    """Run an Optuna study with a reproducible TPE sampler and a median pruner.

    A fixed-seed ``TPESampler`` makes the search reproducible, which matters because the
    selected configuration is a reported result. ``MedianPruner`` is attached for the notebooks
    that report intermediate values; trials that are clearly worse than the running median are
    stopped early rather than fitted to completion.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress,
                   catch=(ValueError, np.linalg.LinAlgError))
    return study


def study_frame(study) -> pd.DataFrame:
    """Return completed trials as a tidy frame, best first."""
    import optuna

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append({"trial": trial.number, "value": trial.value, **trial.params})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No Optuna trial completed successfully.")
    ascending = study.direction.name == "MINIMIZE"
    return frame.sort_values("value", ascending=ascending).reset_index(drop=True)


def plot_optuna(task: Task, studies: dict[str, Any], metric_name: str, name: str):
    """Plot the optimisation history and parameter importances for each study.

    The history shows whether the search converged or was still improving when it stopped, which
    is the evidence for whether more trials would be worth running. Parameter importances show
    which hyperparameters the objective was actually sensitive to; a flat importance profile
    means the search was mostly sampling noise, which is itself worth reporting.
    """
    import optuna

    n = len(studies)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7), squeeze=False)
    for column, (label, study) in enumerate(studies.items()):
        values = [t.value for t in study.trials if t.value is not None]
        minimising = study.direction.name == "MINIMIZE"
        running = np.minimum.accumulate(values) if minimising else np.maximum.accumulate(values)
        axes[0, column].plot(values, "o", ms=3, alpha=0.55, label="trial")
        axes[0, column].plot(running, lw=1.4, color="tab:red", label="best so far")
        axes[0, column].set(title=f"{label}: search history", xlabel="Trial",
                            ylabel=metric_name)
        axes[0, column].legend(fontsize=7)

        try:
            importance = optuna.importance.get_param_importances(study)
            series = pd.Series(importance).sort_values()
            series.plot.barh(ax=axes[1, column], color="tab:blue")
            axes[1, column].set(title=f"{label}: hyperparameter importance", xlabel="Importance")
        except Exception as error:  # too few trials, or a single distinct parameter value
            axes[1, column].text(0.5, 0.5, f"Importance unavailable\n({type(error).__name__})",
                                 ha="center", va="center", fontsize=8)
            axes[1, column].set_axis_off()
        axes[1, column].tick_params(labelsize=7)
    fig.suptitle(f"Optuna search, objective = {metric_name}", fontsize=12)
    fig.tight_layout()
    task.save_figure(fig, name)
    return fig


# --------------------------------------------------------------------------------------------
# Evaluation plots
# --------------------------------------------------------------------------------------------

def plot_regression_validation(task: Task, index: pd.Index, observed: dict, predicted: dict,
                               model_label: str, name: str):
    """Predicted-versus-observed scatter and a validation time series, one column per horizon."""
    targets = list(observed)
    fig, axes = plt.subplots(2, len(targets), figsize=(4.6 * len(targets), 7.5), squeeze=False)
    for column, target in enumerate(targets):
        truth = np.asarray(observed[target], dtype=float)
        prediction = np.asarray(predicted[target], dtype=float)
        scatter_axis = axes[0, column]
        scatter_axis.scatter(truth, prediction, s=10, alpha=0.5)
        limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
        scatter_axis.plot(limits, limits, color="black", lw=1)
        scatter_axis.set(
            title=f"{target}: predicted vs observed\n"
                  f"spread {prediction.std():.1f} vs {truth.std():.1f}",
            xlabel="Observed", ylabel="Predicted")
        series_axis = axes[1, column]
        series_axis.plot(index, truth, lw=1.1, label="Observed")
        series_axis.plot(index, prediction, lw=1.1, label="Predicted")
        series_axis.set(title=f"{target} over validation", xlabel="Feature date", ylabel="CCI")
        series_axis.tick_params(axis="x", labelrotation=30, labelsize=7)
        series_axis.legend(fontsize=7)
    fig.suptitle(f"{model_label}: validation predictions by horizon", fontsize=12)
    fig.tight_layout()
    task.save_figure(fig, name)
    return fig


def plot_classification_validation(task: Task, index: pd.Index, observed, predicted,
                                   labels: Sequence[int], model_label: str, name: str,
                                   whi: pd.Series | None = None):
    """Confusion matrix, per-class recall and a class timeline over the validation period."""
    observed = np.asarray(observed, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    names = [task.class_labels.get(c, str(c)) for c in labels]

    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1, 1.1])

    matrix_axis = fig.add_subplot(grid[0, 0])
    matrix = confusion_matrix(observed, predicted, labels=list(labels))
    image = matrix_axis.imshow(matrix, cmap="Blues")
    matrix_axis.set(xticks=range(len(labels)), yticks=range(len(labels)),
                    xlabel="Predicted", ylabel="Observed", title="Confusion matrix (counts)")
    matrix_axis.set_xticklabels(names, rotation=30, fontsize=7)
    matrix_axis.set_yticklabels(names, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix_axis.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                             color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=matrix_axis, shrink=0.8)

    recall_axis = fig.add_subplot(grid[0, 1])
    report = per_class_report(observed, predicted, labels, task.class_labels)
    report[["precision", "recall", "f1"]].plot.bar(ax=recall_axis, width=0.8)
    recall_axis.set(title="Per-class precision, recall and F1", ylim=(0, 1.05), xlabel="")
    recall_axis.set_xticklabels(names, rotation=30, fontsize=7)
    recall_axis.legend(fontsize=7)

    timeline_axis = fig.add_subplot(grid[1, :])
    timeline_axis.step(index, observed, where="mid", lw=1.2, label="Observed class")
    timeline_axis.step(index, predicted, where="mid", lw=1.0, alpha=0.8, label="Predicted class")
    if whi is not None:
        twin = timeline_axis.twinx()
        twin.plot(index, whi, color="grey", lw=0.6, alpha=0.6, label="WHI at t+7")
        for boundary in (25, 50, 75):
            twin.axhline(boundary, color="grey", lw=0.5, ls=":")
        twin.set_ylabel("WHI", fontsize=8)
        twin.tick_params(labelsize=7)
    timeline_axis.set(title="Hazard class over the validation period", xlabel="Feature date",
                      yticks=list(labels))
    timeline_axis.set_yticklabels(names, fontsize=7)
    timeline_axis.tick_params(axis="x", labelrotation=30, labelsize=7)
    timeline_axis.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"{model_label}: validation classification", fontsize=12)
    fig.tight_layout()
    task.save_figure(fig, name)
    return fig, report


def plot_importance(task: Task, importances: pd.DataFrame, name: str, title: str,
                    top_n: int = 20):
    """Top individual features and totals by engineered family.

    Many of these features are strongly correlated — a variable, its four lags and its four
    rolling means all carry nearly the same signal — and tree ensembles split that shared
    importance among the group. An individually unimportant-looking feature may therefore be part
    of a jointly essential block. Importance ranks features; it does not prove any single one is
    dispensable. The ablation is the more reliable guide to what can be removed.
    """
    columns = [c for c in importances.columns if c != "family"]
    ranked = importances.assign(mean=importances[columns].mean(axis=1))
    top = ranked.sort_values("mean", ascending=False).head(top_n)
    family_totals = (
        ranked.assign(family=[task.family_of(c) for c in ranked.index])
        .groupby("family")[columns].sum()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    top[columns].iloc[::-1].plot.barh(ax=axes[0], width=0.8)
    axes[0].set(title=f"Top {top_n} features by mean importance", xlabel="Importance")
    axes[0].tick_params(labelsize=7)
    axes[0].legend(fontsize=8)
    family_totals.plot.bar(ax=axes[1], width=0.8)
    axes[1].set(title="Total importance by family", ylabel="Summed importance", xlabel="")
    axes[1].tick_params(axis="x", labelrotation=35, labelsize=7)
    axes[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    task.save_figure(fig, name)
    return fig, family_totals


# --------------------------------------------------------------------------------------------
# Artefact export
# --------------------------------------------------------------------------------------------

def _horizon_key(key):
    """Map a target column name onto its integer horizon, leaving other keys untouched.

    The notebooks key their fitted models by target column (``cci_t1``, ``whc_t7``) because that
    is what every other table in the notebook is indexed by. The deployment helper keys them by
    horizon. Converting here keeps both conventions intact rather than forcing one of them to
    bend, and it is the single place that mapping is defined.
    """
    if isinstance(key, int):
        return key
    text = str(key)
    stem = text.rsplit("_t", 1)[-1]
    return int(stem) if "_t" in text and stem.isdigit() else key


def _write_local_bundle(output_dir: Path, experiment: str, models: dict, task: Task,
                        metadata: dict, model_family: str) -> Path:
    """Write a self-contained bundle: one model file per horizon, plus a manifest."""
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for key, model in models.items():
        filename = f"{experiment}_h{key}.joblib" if isinstance(key, int) \
            else f"{experiment}_{key}.joblib"
        joblib.dump(model, output_dir / filename)
        model_files[str(key)] = filename

    (output_dir / "features.json").write_text(json.dumps(task.feature_columns, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (output_dir / "manifest.json").write_text(json.dumps({
        "experiment": experiment,
        "task": task.name,
        "prediction_type": task.kind,
        "model_family": model_family,
        "model_files": model_files,
        "feature_columns": task.feature_columns,
        "feature_column_order_is_significant": True,
        "history_days": int(task.metadata.get("warmup_days", 30)),
    }, indent=2, default=str))
    return output_dir


def export_experiment(task: Task, experiment: str, models: dict, metadata: dict,
                      model_family: str) -> Path:
    """Write the fitted models and their metadata as an API-consumable bundle.

    Regression experiments delegate to ``modeling.deployment.export_tabular_bundle`` so the
    deployed CCI layout stays owned by one function. That helper validates that it has been given
    exactly horizons 1, 2 and 3, so models are re-keyed from target column to horizon first.

    Classification experiments are written locally. ``export_tabular_bundle`` is specific to the
    three-horizon CCI service by its own contract, and forcing a +7 day single-class model through
    it would either fail or produce a manifest that misdescribes what was saved. The local bundle
    carries the same feature-order contract the API depends on; adding a matching
    ``export_classification_bundle`` to ``deployment.py`` is the tidier long-term fix.
    """
    output_dir = task.model_dir / experiment
    keyed = {_horizon_key(key): model for key, model in models.items()}

    if task.kind == "regression":
        try:
            from .deployment import export_tabular_bundle

            return export_tabular_bundle(
                output_dir=output_dir,
                experiment=experiment,
                models=keyed,
                feature_columns=task.feature_columns,
                metadata=metadata,
                model_family=model_family,
            )
        except (ImportError, TypeError, ValueError) as error:
            warnings.warn(
                f"export_tabular_bundle could not accept this experiment ({error}); "
                "writing the bundle locally instead.",
                RuntimeWarning,
            )

    return _write_local_bundle(output_dir, experiment, keyed, task, metadata, model_family)


def build_metadata(task: Task, experiment: str, algorithm: str, model_entries: list[dict],
                   best_params: dict, extra: dict | None = None) -> dict:
    """Assemble the metadata document consumed by the ``/model-metadata`` endpoint."""
    document = {
        "experiment": experiment,
        "student_id": "25945320",
        "target": task.target_name,
        "algorithm": algorithm,
        "prediction_type": task.kind,
        "feature_columns": task.feature_columns,
        "feature_column_order_is_significant": True,
        "best_hyperparameters": best_params,
        "split_boundaries": task.metadata["split_boundaries"],
        "embargo_days": task.metadata["embargo_days"],
        "evaluated_on": ["train", "val"],
        "held_out": ["test"],
        "models": model_entries,
    }
    if task.class_labels:
        document["class_labels"] = {str(k): v for k, v in task.class_labels.items()}
    document.update(extra or {})
    return document


# --------------------------------------------------------------------------------------------
# Experiment discovery, comparison and promotion
# --------------------------------------------------------------------------------------------

MODEL_SUFFIXES = (".joblib", ".pkl", ".pickle")


def _key_from_filename(stem: str, fallback):
    """Recover the horizon a saved model file belongs to from its filename.

    Model files are written by two different code paths — the project's own deployment helper for
    the comfort service and the local writer for the hazard service — so the naming is not fully
    under this module's control. Rather than hard-coding one convention, the trailing integer is
    read from whichever of the common patterns is present (``_h3``, ``_t3``, ``_3``).
    """
    import re

    for pattern in (r"_h(\d+)$", r"_t(\d+)$", r"horizon[_-]?(\d+)$", r"_(\d+)$"):
        found = re.search(pattern, stem)
        if found:
            return int(found.group(1))
    return fallback


def discover_bundles(task: Task) -> dict[str, dict]:
    """Find every exported experiment bundle for this task.

    Returns
    -------
    dict
        Mapping of experiment name to ``{"dir", "metadata", "model_files"}``. Bundles containing
        no loadable model file are still listed, with an empty ``model_files``, so a partially
        written experiment is visible rather than silently skipped.
    """
    root = task.model_dir
    if not root.is_dir():
        raise FileNotFoundError(f"No experiments directory at {root}. Run the experiment "
                                "notebooks first.")

    bundles = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        metadata = {}
        for name in ("manifest.json", "metadata.json"):
            path = directory / name
            if path.is_file():
                try:
                    metadata = {**metadata, **json.loads(path.read_text())}
                except json.JSONDecodeError:
                    warnings.warn(f"{path} is not valid JSON; ignoring it.", RuntimeWarning)

        model_files = {}
        candidates = [p for p in sorted(directory.rglob("*")) if p.suffix in MODEL_SUFFIXES]
        for path in candidates:
            key = _key_from_filename(path.stem, fallback=None)
            if key is None and len(candidates) == 1 and len(task.horizons) == 1:
                key = task.horizons[0]
            if key is None:
                key = path.stem
            model_files[key] = path

        bundles[directory.name] = {
            "dir": directory,
            "metadata": metadata,
            "model_files": model_files,
        }
    return bundles


def load_bundle_models(bundle: dict, expected_keys: Sequence) -> dict:
    """Load the fitted estimators from one bundle, keyed by horizon.

    Raises a message that names the directory and lists what was actually found when the bundle
    does not carry a model for every expected horizon, because a silent partial load would
    produce a comparison table that looks complete and is not.
    """
    import __main__
    import joblib

    # Bundles written before BalancedXGBClassifier was promoted from the notebook recorded it
    # under ``__main__``. Keep those existing artefacts readable; newly exported bundles refer to
    # this module directly.
    if not hasattr(__main__, "BalancedXGBClassifier"):
        __main__.BalancedXGBClassifier = BalancedXGBClassifier

    files = bundle["model_files"]
    missing = [key for key in expected_keys if key not in files]
    if missing:
        listing = sorted(p.name for p in bundle["dir"].rglob("*") if p.is_file())
        raise FileNotFoundError(
            f"{bundle['dir'].name}: no model file found for horizon(s) {missing}. "
            f"Files present: {listing}. If the deployment helper uses a different naming "
            "convention, extend _key_from_filename in modeling/experiment.py."
        )
    return {key: joblib.load(files[key]) for key in expected_keys}


_PRODUCTION_SKIP = {".gitkeep", ".DS_Store"}


def _sidecar_kind(name: str) -> str | None:
    """Map a bundle sidecar onto the stable production filename kind."""
    lowered = name.lower()
    if "promotion" in lowered:
        return "promotion"
    if "manifest" in lowered:
        return "manifest"
    if "metadata" in lowered:
        return "metadata"
    if "features" in lowered:
        return "features"
    if "input_scaler" in lowered:
        return "input_scaler"
    if "target_scaler" in lowered:
        return "target_scaler"
    return None


def _production_filename(stem: str, path: Path, horizon: object | None) -> str:
    """Stable production name so the API path does not change when the winner does."""
    suffixes = "".join(path.suffixes) or path.suffix
    kind = _sidecar_kind(path.name)
    if kind:
        return f"{stem}_{kind}{path.suffix}"
    if path.suffix in MODEL_SUFFIXES or path.suffix == ".pt":
        if isinstance(horizon, int):
            return f"{stem}_h{horizon}{suffixes}"
        return f"{stem}{suffixes}"
    return f"{stem}_{path.name}"


def _rewrite_json_filenames(value: Any, rename_map: dict[str, str]) -> Any:
    """Replace copied filenames inside manifests and metadata."""
    if isinstance(value, str):
        return rename_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_json_filenames(item, rename_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_json_filenames(item, rename_map) for key, item in value.items()}
    return value


def promote_bundle(task: Task, bundle: dict, record: dict) -> Path:
    """Copy a selected experiment bundle into production under a stable filename.

    Experiment directories keep their algorithm-specific names. Production always uses
    ``cci_model_*`` or ``whc_model_*``, so the API can load the same paths after every
    promotion. The production directory is replaced rather than merged, so exactly one
    model is live per task.
    """
    import shutil

    stem = task.production_stem
    destination = task.root / "models" / task.name / "production"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    allowed_suffixes = MODEL_SUFFIXES + (".pt", ".json")
    horizon_by_name = {path.name: key for key, path in bundle["model_files"].items()}
    rename_map: dict[str, str] = {}
    for path in sorted(p for p in bundle["dir"].rglob("*") if p.is_file()):
        if path.name in _PRODUCTION_SKIP or path.suffix not in allowed_suffixes:
            continue
        new_name = _production_filename(stem, path, horizon_by_name.get(path.name))
        if new_name in rename_map.values():
            raise ValueError(
                f"Production filename {new_name!r} collided while promoting "
                f"{bundle['dir'].name}."
            )
        shutil.copy2(path, destination / new_name)
        rename_map[path.name] = new_name

    for path in destination.glob("*.json"):
        try:
            content = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        path.write_text(json.dumps(_rewrite_json_filenames(content, rename_map), indent=2,
                                   default=str))

    record = {
        **record,
        "production_stem": stem,
        "renamed_files": rename_map,
    }
    promotion_name = f"{stem}_promotion.json"
    (destination / promotion_name).write_text(json.dumps(record, indent=2, default=str))
    return destination
