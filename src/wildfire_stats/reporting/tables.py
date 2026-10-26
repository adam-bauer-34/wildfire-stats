"""Tables of fit comparisons and land-cover classification accuracy."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from wildfire_stats.config import resolve

# The 8 GFA / MODIS biome categories used to validate MODIS against GFA (notebook 3)
VALIDATION_CLASSES = [
    "Deciduous Broadleaf forest",
    "Evergreen Broadleaf forest",
    "Mixed forest",
    "Savannas",
    "Woody savannas",
    "Evergreen Needleleaf forest",
    "Open shrublands",
    "Grasslands"
]


def format_pm(val, se):
    """'x.xxx ± y.yyy' (3 decimals), the format used in the summary tables."""
    return f"{val:.3f} ± {se:.3f}"


def parse_val_pm_se(val):
    """From 'x.xxx ± y.yyy' or float -> float(x.xxx)."""
    if isinstance(val, str) and "±" in val:
        return float(val.split("±")[0].strip())
    try:
        return float(val)
    except Exception:
        return np.nan


def build_comparison_table(results_dict, counts_dict, drop_classes=None,
                           threshold=None, min_n=None, max_relerr=None):
    """Construct a summary DataFrame comparing best-fit distributions across datasets.

    Args:
        results_dict (dict): Nested dict {dataset → class → results} with likelihood matrices and params.
        counts_dict (dict): Nested dict {dataset → class → sample counts}.
        drop_classes (list[str], optional): MODIS classes to exclude from final table. Defaults to None.
        threshold (float, optional): Max log-likelihood ratio for alternative fits. Defaults to ``llhr_cutoff``.
        min_n (int, optional): Minimum sample count to include a class. Defaults to ``min_fires``.
        max_relerr (float, optional): Maximum allowed relative parameter error. Defaults to ``max_rel_error``.

    Returns:
        pandas.DataFrame: Table summarizing best and competitive fits across datasets by MODIS class.
    """
    max_relerr = resolve(max_relerr, "max_rel_error")
    min_n = resolve(min_n, "min_fires")
    threshold = resolve(threshold, "llhr_cutoff")
    rows_by_class = {}

    # Build class-level summaries for each dataset
    for dataset, classes in results_dict.items():
        counts = counts_dict.get(dataset, {})
        for modis_class, res in classes.items():
            n = counts.get(modis_class, np.nan)
            if pd.isna(n) or n < min_n:
                continue

            best_fit = res.get("best_fit", None)
            llmat = res.get("likelihood_matrix", None)
            params = res.get("params", None)
            if best_fit is None or llmat is None or params is None:
                continue

            # Check if a given distribution has valid parameter estimates
            def valid_fit(dist):
                if dist not in params.index:
                    return False
                if pd.notna(params.loc[dist, "reduces_to"]):
                    return False
                for p, se in [("p1", "p1_se"), ("p2", "p2_se")]:
                    if p in params.columns and se in params.columns:
                        val, err = params.loc[dist, p], params.loc[dist, se]
                        if pd.notna(val) and pd.notna(err) and val != 0:
                            if abs(err / val) > max_relerr:
                                return False
                return True

            # Identify candidate fits within the likelihood threshold
            col = [c for c in llmat.columns if best_fit in c]
            if not col:
                continue
            deltas = llmat[col[0]].copy()

            fits = []
            if valid_fit(best_fit):
                fits.append((best_fit, "", ""))

            for dist, val in deltas.items():
                if dist == best_fit:
                    continue
                if val <= threshold and valid_fit(dist):
                    fits.append((dist, round(val, 3) if not np.isinf(val) else np.inf, ""))

            if not fits:
                continue

            rows_by_class.setdefault(modis_class, {}).setdefault(dataset, {
                "n": n, "fits": fits
            })

    # Expand nested dict into a table
    final_rows = []
    all_classes = sorted(rows_by_class.keys())
    for modis in all_classes:
        mtbs = rows_by_class[modis].get("MTBS", {"n": "", "fits": []})
        gfa = rows_by_class[modis].get("GFA", {"n": "", "fits": []})
        idaho = rows_by_class[modis].get("Idaho", {"n": "", "fits": []})

        nrows = max(len(mtbs["fits"]), len(gfa["fits"]), len(idaho["fits"]))
        for i in range(nrows):
            row = {"modis_class": modis}
            for dataset, d in [("MTBS", mtbs), ("GFA", gfa), ("Idaho", idaho)]:
                row[f"{dataset}_n"] = d["n"] if i == 0 else ""
                if i < len(d["fits"]):
                    dist, val, reduces = d["fits"][i]
                    row[f"{dataset}_fit"] = dist
                    row[f"{dataset}_loglik"] = val
                    row[f"{dataset}_reduces_to"] = reduces
                else:
                    row[f"{dataset}_fit"] = ""
                    row[f"{dataset}_loglik"] = ""
                    row[f"{dataset}_reduces_to"] = ""
            final_rows.append(row)

    df = pd.DataFrame(final_rows)

    # Drop unwanted classes
    if drop_classes:
        df = df[~df["modis_class"].isin(drop_classes)]

    # Keep MODIS class name only on first row per group
    def keep_first(series):
        return [series.iloc[0]] + [""] * (len(series) - 1)
    df["modis_class"] = df.groupby("modis_class")["modis_class"].transform(keep_first)

    return df.reset_index(drop=True)


def classification_metrics(df, ref_col="gfa_class", pred_col="modis_class", kind="sensitivity", valid_classes=None):
    """Per-class agreement between a reference and a predicted classification.

    Only rows where both columns are in ``valid_classes`` (default
    ``VALIDATION_CLASSES``) are used.

    Args:
        df: Table with both classifications (e.g. ``build_gfa_modis_validation`` output).
        ref_col: Reference / ground-truth column.
        pred_col: Predicted column.
        kind: ``"sensitivity"`` -> land_cover, n_samples, sensitivity, specificity
            (was ``evaluate_modis_against_gfa``); ``"reliability"`` -> land_cover,
            n_predicted_samples, precision, recall, specificity, f1_score (was
            ``evaluate_modis_reliability``, called with ref_col="modis_class",
            pred_col="gfa_class").

    Returns:
        pd.DataFrame sorted by land_cover (empty if no rows survive the filter).
    """
    if kind not in ("sensitivity", "reliability"):
        raise ValueError(f"kind must be 'sensitivity' or 'reliability', got {kind!r}")
    valid_classes = VALIDATION_CLASSES if valid_classes is None else valid_classes

    # Keep only samples within these classes
    df = df[df[ref_col].isin(valid_classes) & df[pred_col].isin(valid_classes)].copy()
    if df.empty:
        print("No valid samples remain after filtering by the 8 target classes.")
        return pd.DataFrame()

    df[ref_col] = df[ref_col].astype(str).str.strip()
    df[pred_col] = df[pred_col].astype(str).str.strip()
    all_classes = sorted(set(valid_classes) & set(df[ref_col].unique()) & set(df[pred_col].unique()))
    cm = confusion_matrix(df[ref_col], df[pred_col], labels=all_classes)

    results = []
    for i, cls in enumerate(all_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        n_samples = cm[i, :].sum()

        if kind == "sensitivity":
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            results.append({
                "land_cover": cls,
                "n_samples": n_samples,
                "sensitivity": sensitivity,
                "specificity": specificity
            })
        else:
            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
            results.append({
                "land_cover": cls,
                "n_predicted_samples": n_samples,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1
            })

    summary = pd.DataFrame(results).sort_values("land_cover").reset_index(drop=True)

    print(f"Evaluated {len(df)} samples across {len(all_classes)} biome classes.")
    return summary


def confusion_to_labels(confusion: pd.DataFrame):
    """Expand a crosstab (rows = true, columns = predicted) back into label lists."""
    true_labels = []
    pred_labels = []
    for true_class in confusion.index:
        for pred_class in confusion.columns:
            count = confusion.loc[true_class, pred_class]
            true_labels.extend([true_class] * count)
            pred_labels.extend([pred_class] * count)
    return true_labels, pred_labels


def print_overall_metrics(confusion: pd.DataFrame) -> None:
    """Print accuracy, weighted F1, and the sklearn classification report for a crosstab."""
    true_labels, pred_labels = confusion_to_labels(confusion)
    print("=== Summary Statistics ===")
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("Weighted F1:", f1_score(true_labels, pred_labels, average="weighted"))
    print(classification_report(true_labels, pred_labels))


def period_metrics(true_labels, pred_labels) -> dict:
    """Accuracy and weighted precision / recall / F1 (zero_division=0) plus the text report."""
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision_weighted": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "report": classification_report(true_labels, pred_labels, zero_division=0)
    }


def print_period_metrics(metrics: dict) -> None:
    """Print the per-period metrics returned by ``plotting.landcover.plot_confusion_by_period``."""
    for label, m in metrics.items():
        print(f"\n=== {label} ===")
        if m:
            print(f"Accuracy: {m['accuracy']:.3f}")
            print(f"Weighted Precision: {m['precision_weighted']:.3f}")
            print(f"Weighted Recall: {m['recall_weighted']:.3f}")
            print(f"Weighted F1: {m['f1_weighted']:.3f}")
        else:
            print("No data available for this period.")
