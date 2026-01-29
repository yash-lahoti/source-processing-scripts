#!/usr/bin/env python3
"""
Cohort ID and site analysis: compare patient_uid vs pat_mrn, unique counts by site_id.
Uses config.yaml for paths and column names. Produces EDA-style breakdown so numbers can be confirmed.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import argparse
from typing import Optional, Tuple

from patient_aggregator.config_loader import load_config, get_output_directory


def find_data_directory(project_root: Path) -> Optional[Path]:
    """Find directory containing cohort (and other data) files."""
    for name in ["sample_data", "data", "input_data"]:
        d = project_root / name
        if d.exists() and d.is_dir():
            for ext in [".csv", ".xlsx", ".xls"]:
                if list(d.glob(f"cohort{ext}")):
                    return d
    # Fallback: project root
    for ext in [".csv", ".xlsx", ".xls"]:
        if (project_root / f"cohort{ext}").exists():
            return project_root
    return None


def load_cohort_file(data_dir: Path, cohort_base: str) -> pd.DataFrame:
    """Load cohort from cohort.xlsx, cohort.csv, or cohort.xls."""
    for ext in [".xlsx", ".csv", ".xls"]:
        path = data_dir / f"{cohort_base}{ext}"
        if path.exists():
            if ext == ".csv":
                try:
                    return pd.read_csv(path, encoding="utf-8")
                except UnicodeDecodeError:
                    return pd.read_csv(path, encoding="latin-1")
            return pd.read_excel(path)
    raise FileNotFoundError(f"Cohort file not found: {data_dir / cohort_base}[.xlsx|.csv|.xls]")


def run_cohort_analysis(
    config_path: Optional[Path] = None,
    cohort_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load cohort, compare patient_uid vs pat_mrn, and summarize unique patients by site_id.
    Returns (cohort DataFrame, summary dict).
    """
    if config_path is None:
        config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    patient_id_col = config["patient_id_column"]
    out_dir = output_dir or get_output_directory(config, project_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cohort analysis config
    cohort_cfg = config.get("cohort_analysis", {})
    cohort_base = cohort_cfg.get("cohort_file", "cohort")
    site_col = cohort_cfg.get("site_id_column", "site_id")
    mrn_col = cohort_cfg.get("mrn_column", "pat_mrn")

    # Resolve cohort path
    if cohort_path is not None:
        cohort_path = Path(cohort_path)
        if not cohort_path.exists():
            raise FileNotFoundError(f"Cohort file not found: {cohort_path}")
        data_dir_cohort = cohort_path.parent
        if cohort_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(cohort_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(cohort_path, encoding="latin-1")
        else:
            df = pd.read_excel(cohort_path)
    else:
        data_dir_resolved = data_dir or find_data_directory(project_root)
        if data_dir_resolved is None:
            raise FileNotFoundError(
                "Cohort not found. Put cohort.csv or cohort.xlsx in sample_data/, data/, or pass --cohort path."
            )
        df = load_cohort_file(data_dir_resolved, cohort_base)

    # Validate columns
    missing = [c for c in [patient_id_col, mrn_col, site_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cohort missing columns: {missing}. Found: {list(df.columns)}. "
            f"Set cohort_analysis.site_id_column / mrn_column in config if names differ."
        )

    # Drop full-row duplicates for id comparison
    df_clean = df.drop_duplicates()
    n_rows = len(df)
    n_rows_clean = len(df_clean)

    # Unique counts
    n_uid = df[patient_id_col].nunique()
    n_mrn = df[mrn_col].nunique()
    n_uid_clean = df_clean[patient_id_col].nunique()
    n_mrn_clean = df_clean[mrn_col].nunique()

    # 1:1 checks
    uid_per_mrn = df.groupby(mrn_col)[patient_id_col].nunique()
    mrn_per_uid = df.groupby(patient_id_col)[mrn_col].nunique()
    mrn_with_multiple_uid = (uid_per_mrn > 1).sum()
    uid_with_multiple_mrn = (mrn_per_uid > 1).sum()
    mrn_missing = df[mrn_col].isna().sum()
    mrn_empty = (df[mrn_col].astype(str).str.strip() == "").sum()
    mrn_placeholder = (df[mrn_col].astype(str).str.strip().isin(["0", "NA", "nan", ""])).sum()

    # By site
    unique_patients_by_site_uid = df.groupby(site_col)[patient_id_col].nunique()
    unique_patients_by_site_mrn = df.groupby(site_col)[mrn_col].nunique()
    total_rows_by_site = df.groupby(site_col).size()

    summary = {
        "n_rows": n_rows,
        "n_rows_deduped": n_rows_clean,
        "n_unique_patient_uid": n_uid,
        "n_unique_pat_mrn": n_mrn,
        "n_unique_patient_uid_deduped": n_uid_clean,
        "n_unique_pat_mrn_deduped": n_mrn_clean,
        "mrn_with_multiple_uid": int(mrn_with_multiple_uid),
        "uid_with_multiple_mrn": int(uid_with_multiple_mrn),
        "mrn_missing": int(mrn_missing),
        "mrn_empty_or_placeholder": int(mrn_empty + mrn_placeholder),
        "ids_match": n_uid == n_mrn and mrn_with_multiple_uid == 0 and uid_with_multiple_mrn == 0,
        "by_site_uid": unique_patients_by_site_uid,
        "by_site_mrn": unique_patients_by_site_mrn,
        "total_rows_by_site": total_rows_by_site,
    }

    # Build report text
    lines = [
        "=" * 60,
        "Cohort patient_uid vs pat_mrn and site_id analysis",
        "=" * 60,
        "",
        "Config:",
        f"  patient_id_column: {patient_id_col}",
        f"  mrn_column: {mrn_col}",
        f"  site_id_column: {site_col}",
        "",
        "--- Row counts ---",
        f"  Total rows:           {n_rows}",
        f"  Rows (after dedup):   {n_rows_clean}",
        "",
        "--- Unique identifier counts ---",
        f"  Unique {patient_id_col}:  {n_uid}",
        f"  Unique {mrn_col}:         {n_mrn}",
        f"  Match (n_uid == n_mrn):   {n_uid == n_mrn}",
        "",
        "--- 1:1 consistency ---",
        f"  MRN with >1 patient_uid:  {mrn_with_multiple_uid}",
        f"  patient_uid with >1 MRN:  {uid_with_multiple_mrn}",
        f"  Missing MRN:              {mrn_missing}",
        f"  Empty/placeholder MRN:    {mrn_empty + mrn_placeholder}",
        f"  IDs consistent (1:1):     {summary['ids_match']}",
        "",
        "--- Unique patients by site_id (EDA breakdown) ---",
        "",
    ]

    by_site_df = pd.DataFrame({
        "site_id": unique_patients_by_site_uid.index,
        "unique_patient_uid": unique_patients_by_site_uid.values,
        "unique_pat_mrn": unique_patients_by_site_mrn.values,
        "total_rows": total_rows_by_site.values,
    })
    by_site_df["uid_eq_mrn"] = by_site_df["unique_patient_uid"] == by_site_df["unique_pat_mrn"]
    by_site_df = by_site_df.sort_values("site_id")

    lines.append(by_site_df.to_string(index=False))
    lines.append("")
    lines.append("--- Totals ---")
    lines.append(f"  Sum(unique patient_uid per site): {by_site_df['unique_patient_uid'].sum()}")
    lines.append(f"  Sum(unique pat_mrn per site):     {by_site_df['unique_pat_mrn'].sum()}")
    lines.append(f"  (If each patient is at one site, these equal global unique counts above.)")
    lines.append("")
    report_text = "\n".join(lines)

    # Save report and CSV
    report_file = out_dir / "cohort_id_site_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    by_site_file = out_dir / "cohort_unique_patients_by_site.csv"
    by_site_df.to_csv(by_site_file, index=False)

    print(report_text)
    print(f"Report saved: {report_file}")
    print(f"By-site CSV saved: {by_site_file}")

    # Optional bar chart: unique patients per site
    if save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(by_site_df))
            w = 0.35
            ax.bar([i - w / 2 for i in x], by_site_df["unique_patient_uid"], w, label=f"unique {patient_id_col}", color="steelblue")
            ax.bar([i + w / 2 for i in x], by_site_df["unique_pat_mrn"], w, label=f"unique {mrn_col}", color="coral", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(by_site_df["site_id"].astype(str))
            ax.set_xlabel("site_id")
            ax.set_ylabel("Unique count")
            ax.set_title("Unique patients per site (patient_uid vs pat_mrn)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plot_file = out_dir / "cohort_unique_patients_by_site.png"
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"Plot saved: {plot_file}")
        except Exception as e:
            print(f"Could not save plot: {e}")

    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Cohort analysis: compare patient_uid vs pat_mrn, unique patients by site_id (uses config.yaml)."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--cohort", type=Path, default=None, help="Path to cohort file (overrides config/data dir)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory (default: auto-detect)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: from config)")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving bar chart")
    args = parser.parse_args()

    run_cohort_analysis(
        config_path=args.config,
        cohort_path=args.cohort,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
