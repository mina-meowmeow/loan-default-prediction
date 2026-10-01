"""Statistical outlier flagging for bank tables. See note.txt for the design notes.

Usage:
    python outlier_detector.py <data.csv> <config.yaml> [--out flagged.csv] [--report report.txt]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml


@dataclass
class ColumnReport:
    name: str
    method: str
    n_flagged: int = 0
    skew: float | None = None
    reason: str | None = None


@dataclass
class DetectionResult:
    df: pd.DataFrame
    column_reports: list[ColumnReport] = field(default_factory=list)
    domain_violation_counts: dict[str, int] = field(default_factory=dict)
    consistency_violation_counts: dict[str, int] = field(default_factory=dict)


def _modified_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        mad_fallback = (series - median).abs().mean()
        if mad_fallback == 0:
            return pd.Series(0.0, index=series.index)
        return 0.6745 * (series - median) / mad_fallback
    return 0.6745 * (series - median) / mad


def _standard_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _check_balance_exceeds_limit(df: pd.DataFrame, slack: float = 1.02) -> pd.Series:
    return df["current_balance"] > df["total_credit_limit"] * slack


def _check_monthly_income_mismatch(df: pd.DataFrame, tolerance: float = 0.15) -> pd.Series:
    return (df["monthly_income"] * 12 - df["annual_income"]).abs() > tolerance * df["annual_income"]


def _check_nonpositive_installment(df: pd.DataFrame) -> pd.Series:
    return df["installment"] <= 0


CONSISTENCY_CHECKS = {
    "balance_exceeds_limit": _check_balance_exceeds_limit,
    "monthly_income_income_mismatch": _check_monthly_income_mismatch,
    "installment_loan_term_mismatch": _check_nonpositive_installment,
}


def detect(df: pd.DataFrame, config: dict) -> DetectionResult:
    df = df.copy()
    exclude = set(config.get("exclude_columns", []))
    min_unique = config.get("min_unique_for_numeric", 8)
    skew_threshold = config.get("skew_threshold", 1.0)
    z_thresh = config.get("zscore_threshold", 3.5)
    rz_thresh = config.get("robust_zscore_threshold", 3.5)

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    reports: list[ColumnReport] = []
    flag_cols: list[str] = []

    for col in numeric_cols:
        series = df[col]
        n_valid = series.notna().sum()
        n_unique = series.nunique(dropna=True)

        if n_valid < 20:
            reports.append(ColumnReport(col, "skipped_low_variance", reason=f"only {n_valid} non-null values"))
            continue
        if n_unique < min_unique:
            reports.append(ColumnReport(col, "skipped_low_cardinality", reason=f"{n_unique} distinct values"))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skew = series.skew()

        flag_col = f"_flag_{col}"
        z_col = f"_z_{col}"

        if abs(skew) > skew_threshold:
            z = _modified_zscore(series)
            flagged = z.abs() > rz_thresh
            method = "robust_z"
        else:
            z = _standard_zscore(series)
            flagged = z.abs() > z_thresh
            method = "standard_z"

        df[z_col] = z
        df[flag_col] = flagged.fillna(False)
        flag_cols.append(flag_col)

        reports.append(ColumnReport(col, method, n_flagged=int(flagged.sum()), skew=round(float(skew), 3)))

    domain_counts: dict[str, int] = {}
    for col, rule in config.get("domain_rules", {}).items():
        if col not in df.columns:
            continue
        viol = pd.Series(False, index=df.index)
        if "min" in rule:
            viol |= df[col] < rule["min"]
        if "max" in rule:
            viol |= df[col] > rule["max"]
        if "values" in rule:
            viol |= ~df[col].isin(rule["values"])
        viol = viol.fillna(False)
        dcol = f"_domain_violation_{col}"
        df[dcol] = viol
        flag_cols.append(dcol)
        domain_counts[col] = int(viol.sum())

    consistency_counts: dict[str, int] = {}
    for rule in config.get("consistency_rules", []):
        name, params = rule["name"], rule.get("params", {})
        check_fn = CONSISTENCY_CHECKS.get(name)
        if check_fn is None:
            print(f"warn: unknown consistency rule '{name}', skipping", file=sys.stderr)
            continue
        try:
            result = check_fn(df, **params).fillna(False)
        except Exception as e:
            print(f"warn: consistency rule '{name}' failed: {e}", file=sys.stderr)
            continue
        ccol = f"_consistency_{name}"
        df[ccol] = result
        flag_cols.append(ccol)
        consistency_counts[name] = int(result.sum())

    df["anomaly_flag_count"] = df[flag_cols].sum(axis=1) if flag_cols else 0
    df["is_anomaly"] = df["anomaly_flag_count"] > 0

    return DetectionResult(df, reports, domain_counts, consistency_counts)


def write_report(result: DetectionResult, path: str, total_rows: int) -> None:
    lines = [
        f"rows analyzed: {total_rows}",
        f"rows flagged (>=1 anomaly signal): {int(result.df['is_anomaly'].sum())}",
        "",
        "-- per-column z-score results --",
    ]
    for r in result.column_reports:
        if r.method in ("standard_z", "robust_z"):
            lines.append(f"{r.name:28s} method={r.method:11s} skew={r.skew:>7} flagged={r.n_flagged}")
        else:
            lines.append(f"{r.name:28s} {r.method} ({r.reason})")
    lines += ["", "-- domain rule violations --"]
    for col, n in result.domain_violation_counts.items():
        lines.append(f"{col:28s} violations={n}")
    lines += ["", "-- cross-field consistency violations --"]
    for name, n in result.consistency_violation_counts.items():
        lines.append(f"{name:28s} violations={n}")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag anomalous rows in a bank table.")
    parser.add_argument("csv")
    parser.add_argument("config")
    parser.add_argument("--out", default="flagged_output.csv")
    parser.add_argument("--report", default="anomaly_report.txt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(args.csv)
    result = detect(df, config)
    result.df.to_csv(args.out, index=False)
    write_report(result, args.report, len(df))


if __name__ == "__main__":
    main()
