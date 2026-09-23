import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "sensitivity",
    "balanced_accuracy",
]


def calculate_binary_metrics(y_true, y_prob, threshold):
    """Calculate threshold-dependent binary metrics from saved predictions."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    specificity = tn / max(tn + fp, 1)
    sensitivity = float(recall)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "sensitivity": sensitivity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def threshold_grid(start=0.05, stop=0.95, step=0.01):
    """Return a rounded inclusive threshold grid to avoid floating-point drift."""
    return np.round(np.arange(start, stop + step / 2, step), 2)


def analyze_arrays(y_true, y_prob, thresholds=None):
    """Evaluate one saved prediction set over a threshold grid."""
    thresholds = threshold_grid() if thresholds is None else np.asarray(thresholds)
    rows = []
    for threshold in thresholds:
        row = {"threshold": float(threshold)}
        row.update(calculate_binary_metrics(y_true, y_prob, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _best_row(sweep, metric):
    """Select the first threshold attaining the maximum metric value."""
    return sweep.loc[sweep[metric].idxmax()]


def summarize_round(round_number, sweep):
    """Return best-F1, best-balanced-accuracy, and threshold=0.25 summaries."""
    threshold_row = sweep.iloc[(sweep["threshold"] - 0.25).abs().argmin()]
    f1_row = _best_row(sweep, "f1")
    balanced_row = _best_row(sweep, "balanced_accuracy")

    def as_metrics(row):
        return {"threshold": float(row["threshold"]), **{key: float(row[key]) for key in METRIC_COLUMNS}}

    return {
        "round": int(round_number),
        "best_f1": as_metrics(f1_row),
        "best_balanced_accuracy": as_metrics(balanced_row),
        "threshold_0.25": as_metrics(threshold_row),
    }


def load_prediction_files(predictions_dir):
    """Load saved round prediction arrays in round-number order."""
    paths = glob.glob(os.path.join(predictions_dir, "round_*_global_predictions.npz"))
    if not paths:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    def round_number(path):
        return int(os.path.basename(path).split("_")[1])

    return [(round_number(path), path) for path in sorted(paths, key=round_number)]


def validate_prediction_file(path, expected_samples=None):
    """Validate one saved prediction archive without loading a model."""
    with np.load(path) as data:
        required = {"y_true", "y_prob"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Prediction file is missing keys: {sorted(missing)}")
        y_true = np.asarray(data["y_true"])
        y_prob = np.asarray(data["y_prob"])

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob have different sample counts")
    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError("y_prob contains values outside [0, 1]")
    if expected_samples is not None and y_true.shape[0] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {y_true.shape[0]}")
    return y_true, y_prob


def run_analysis(predictions_dir, output_dir):
    """Analyze saved prediction files and write sweeps, summaries, and a final plot."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []
    summaries = []
    prediction_files = load_prediction_files(predictions_dir)

    for round_number, path in prediction_files:
        y_true, y_prob = validate_prediction_file(path)
        sweep = analyze_arrays(y_true, y_prob)
        sweep.insert(0, "round", round_number)
        all_rows.append(sweep)
        summaries.append(summarize_round(round_number, sweep))

    complete_sweep = pd.concat(all_rows, ignore_index=True)
    complete_sweep.to_csv(os.path.join(output_dir, "threshold_sweep.csv"), index=False)

    summary_rows = []
    for summary in summaries:
        row = {"round": summary["round"]}
        for label in ("best_f1", "best_balanced_accuracy", "threshold_0.25"):
            for key, value in summary[label].items():
                row[f"{label}_{key}"] = value
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "threshold_summary.csv"), index=False)

    final_round = max(round_number for round_number, _ in prediction_files)
    final_sweep = complete_sweep[complete_sweep["round"] == final_round]
    plt.figure(figsize=(9, 6))
    for metric in ("f1", "sensitivity", "specificity", "balanced_accuracy"):
        plt.plot(final_sweep["threshold"], final_sweep[metric], label=metric.replace("_", " ").title())
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title(f"Final Global Threshold Sensitivity (Round {final_round})")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_threshold_sensitivity.png"), dpi=150)
    plt.close()

    return summaries, complete_sweep


def main():
    parser = argparse.ArgumentParser(description="Analyze saved FL prediction probabilities offline")
    parser.add_argument("--predictions-dir", default=os.path.join("outputs", "current_run", "predictions"))
    parser.add_argument("--output-dir", default=os.path.join("outputs", "current_run", "threshold_analysis"))
    args = parser.parse_args()
    summaries, _ = run_analysis(args.predictions_dir, args.output_dir)
    for summary in summaries:
        print(
            f"Round {summary['round']}: "
            f"best F1 threshold={summary['best_f1']['threshold']:.2f}, "
            f"best balanced-accuracy threshold={summary['best_balanced_accuracy']['threshold']:.2f}, "
            f"threshold 0.25 F1={summary['threshold_0.25']['f1']:.4f}, "
            f"balanced accuracy={summary['threshold_0.25']['balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()