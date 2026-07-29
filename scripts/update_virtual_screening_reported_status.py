from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "Database.csv"
VIRTUAL_DIR = ROOT / "data" / "VirtualScreening"

METHODS = [
    ("ChyTryPep", "Chymotrypsin + Trypsin + Pepsin (pH=1.3)"),
    ("ThePap", "Thermolysin + Papain"),
    ("PepPan", "Pepsin + Pancreatic"),
    ("ProteinaseK", "Proteinase K"),
    ("Thermolysin", "Thermolysin"),
    ("Papain", "Papain"),
    ("Pepsin", "Pepsin (pH=1.3)"),
    ("Chymotrypsin", "Chymotrypsin"),
    ("Trypsin", "Trypsin"),
]

TASKS = {
    "bitter": {
        "label": "Bitter",
        "root": VIRTUAL_DIR / "bitter" / "virtual_digest",
        "probability_prefix": "bitter_probability",
        "download_prefix": "bitter",
        "method_summary": "hydrolysis_method_bitter_summary.csv",
        "reported_summary": "hydrolysis_method_bitter_reported_summary.csv",
        "source_count_summary": "hydrolysis_source_count_bitter_summary.csv",
        "high_conf_dir": "high_confidence_bitter",
        "high_conf_files": [
            "high_confidence_bitter_gte_0.93.csv.gz",
            "high_confidence_bitter_gte_0.93_minlen8.csv.gz",
        ],
        "high_conf_summary": "high_confidence_bitter_gte_0.93_summary.json",
        "high_conf_by_method": "high_confidence_bitter_gte_0.93_by_method.csv",
    },
    "umami": {
        "label": "Umami",
        "root": VIRTUAL_DIR / "umami" / "virtual_digest",
        "probability_prefix": "umami_probability",
        "download_prefix": "umami",
        "method_summary": "hydrolysis_method_umami_summary.csv",
        "reported_summary": "hydrolysis_method_umami_reported_summary.csv",
        "source_count_summary": "hydrolysis_source_count_umami_summary.csv",
        "high_conf_dir": "high_confidence_umami",
        "high_conf_files": [
            "high_confidence_umami_gte_0.999.csv.gz",
            "high_confidence_umami_gte_0.999_minlen8.csv.gz",
        ],
        "high_conf_summary": "high_confidence_umami_gte_0.999_summary.json",
        "high_conf_minlen_summary": "high_confidence_umami_gte_0.999_minlen8_summary.json",
        "high_conf_by_method": "high_confidence_umami_gte_0.999_by_method.csv",
        "high_conf_minlen_by_method": "high_confidence_umami_gte_0.999_minlen8_by_method.csv",
    },
}

UNIFIED_MANIFEST_COLUMNS = [
    "Task",
    "Dataset",
    "Library",
    "Hydrolysis_Method_ID",
    "Hydrolysis_Method",
    "Enzyme(s)",
    "Probability_Tier",
    "Probability group",
    "Count",
    "Unique sequences",
    "Mean_Probability",
    "Median_Probability",
    "Mean_Length",
    "Download_File",
    "Pool file",
    "Filtered file",
    "N/C terminal analysis",
    "MEME output",
    "Status",
]

BASE_DATASET = "Virtual hydrolysate fragments from 21,249 proteins across 60 species"
BASE_LIBRARY = "Virtual hydrolysate screening"
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 6, "mtime": 1}


def normalize_sequence_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def split_taste_labels(value: Any) -> set[str]:
    return {part.strip().lower() for part in str(value).split(";") if part.strip()}


def load_reported_sequences(label: str) -> set[str]:
    database = pd.read_csv(DATABASE_PATH, dtype=str, keep_default_na=False).fillna("")
    missing = {"Sequence", "Taste"}.difference(database.columns)
    if missing:
        raise ValueError(f"{DATABASE_PATH} is missing columns: {sorted(missing)}")
    mask = database["Taste"].map(lambda value: label.lower() in split_taste_labels(value))
    return set(normalize_sequence_series(database.loc[mask, "Sequence"]))


def to_csv(path: Path, df: pd.DataFrame, compression: Any = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig", compression=compression)
    tmp_path.replace(path)


def read_probability_pool(task_key: str, method_id: str) -> pd.DataFrame:
    cfg = TASKS[task_key]
    download_dir = cfg["root"] / "download_files" / "by_hydrolysis_method"
    prefix = cfg["probability_prefix"]
    files = [
        download_dir / f"{prefix}_{method_id}_lt_0.50.csv.gz",
        download_dir / f"{prefix}_{method_id}_gte_0.50.csv.gz",
    ]
    parts = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_csv(path, keep_default_na=False))
    pool = pd.concat(parts, ignore_index=True)
    pool["sequence"] = normalize_sequence_series(pool["sequence"])
    if "Hydrolysis_Source_Count" not in pool.columns and "hydrolysis_count" in pool.columns:
        pool["Hydrolysis_Source_Count"] = pd.to_numeric(pool["hydrolysis_count"], errors="coerce").fillna(1).astype(int)
    return pool


def reported_manifest_row(task_label: str, method_id: str, method_name: str, status: str, subset: pd.DataFrame, path: Path) -> dict[str, Any]:
    relative = path.relative_to(VIRTUAL_DIR).as_posix()
    return {
        "Task": task_label,
        "Dataset": BASE_DATASET,
        "Library": BASE_LIBRARY,
        "Hydrolysis_Method_ID": method_id,
        "Hydrolysis_Method": method_name,
        "Enzyme(s)": method_name,
        "Probability_Tier": status,
        "Probability group": status,
        "Count": int(len(subset)),
        "Unique sequences": int(subset["sequence"].nunique()) if "sequence" in subset.columns else int(len(subset)),
        "Mean_Probability": float(subset["Final_Prob"].mean()) if len(subset) and "Final_Prob" in subset.columns else pd.NA,
        "Median_Probability": float(subset["Final_Prob"].median()) if len(subset) and "Final_Prob" in subset.columns else pd.NA,
        "Mean_Length": float(subset["length"].mean()) if len(subset) and "length" in subset.columns else pd.NA,
        "Download_File": relative,
        "Pool file": relative,
        "Filtered file": relative,
        "N/C terminal analysis": f"{task_key_placeholder(task_label)}/virtual_digest/tables/hydrolysis_method_{task_key_placeholder(task_label)}_summary.csv",
        "MEME output": "",
        "Status": "Ready",
    }


def task_key_placeholder(task_label: str) -> str:
    return task_label.lower()


def export_reported_files_for_task(task_key: str, reported_sequences: set[str]) -> pd.DataFrame:
    cfg = TASKS[task_key]
    task_label = cfg["label"]
    download_dir = cfg["root"] / "download_files" / "by_hydrolysis_method"
    tables_dir = cfg["root"] / "tables"
    manifest_rows: list[dict[str, Any]] = []
    reported_rows: list[dict[str, Any]] = []
    unique_parts: list[pd.DataFrame] = []

    for method_id, method_name in METHODS:
        pool = read_probability_pool(task_key, method_id)
        pool["Hydrolysis_Method_ID"] = method_id
        pool["Hydrolysis_Method"] = method_name
        pool["Reported_Status"] = pool["sequence"].isin(reported_sequences).map({True: "Reported", False: "No reported"})
        if "Final_Prob" in pool.columns:
            pool = pool.sort_values(["Final_Prob", "sequence"], ascending=[False, True])
        else:
            pool = pool.sort_values("sequence")

        source_sequences = int(len(pool))
        reported_count = int(pool["Reported_Status"].eq("Reported").sum())
        no_reported_count = source_sequences - reported_count
        reported_rows.append(
            {
                "method_id": method_id,
                "method_name": method_name,
                "source_sequences": source_sequences,
                "Reported": reported_count,
                "No_reported": no_reported_count,
                "Reported_Rate": reported_count / source_sequences if source_sequences else 0.0,
            }
        )

        for status, file_status in [("Reported", "reported"), ("No reported", "no_reported")]:
            subset = pool[pool["Reported_Status"].eq(status)].copy()
            out_path = download_dir / f"{cfg['download_prefix']}_{file_status}_{method_id}.csv.gz"
            if "Reported_Status" in subset.columns:
                columns = ["sequence", "length", "Hydrolysis_Method_ID", "Hydrolysis_Method", "Reported_Status"]
                columns += [col for col in subset.columns if col not in columns and col not in {"sequence_norm"}]
            else:
                columns = list(subset.columns)
            to_csv(out_path, subset[columns], compression=GZIP_COMPRESSION)
            manifest_rows.append(reported_manifest_row(task_label, method_id, method_name, status, subset, out_path))

        unique_cols = ["sequence", "length", "Final_Prob", "Reported_Status"]
        if "Hydrolysis_Source_Count" in pool.columns:
            unique_cols.append("Hydrolysis_Source_Count")
        unique_parts.append(pool[unique_cols].drop_duplicates("sequence"))
        print(f"{task_label:6s} {method_id:14s} Reported={reported_count:5d} No reported={no_reported_count:7d}")

    reported_df = pd.DataFrame(reported_rows)
    to_csv(tables_dir / cfg["reported_summary"], reported_df)

    reported_manifest = pd.DataFrame(manifest_rows, columns=UNIFIED_MANIFEST_COLUMNS)
    to_csv(tables_dir / "reported_download_manifest.csv", reported_manifest)

    update_method_summary(task_key, reported_df)
    update_source_count_summary(task_key, pd.concat(unique_parts, ignore_index=True).drop_duplicates("sequence"))
    return reported_manifest


def update_method_summary(task_key: str, reported_df: pd.DataFrame) -> None:
    cfg = TASKS[task_key]
    summary_path = cfg["root"] / "tables" / cfg["method_summary"]
    summary = pd.read_csv(summary_path, keep_default_na=False)
    keep = summary.drop(columns=[col for col in ["Reported", "No_reported", "Reported_Rate"] if col in summary.columns])
    merged = keep.merge(
        reported_df[["method_id", "Reported", "No_reported", "Reported_Rate"]],
        on="method_id",
        how="left",
    )
    source_idx = merged.columns.get_loc("source_sequences")
    for col in ["Reported", "No_reported", "Reported_Rate"]:
        values = merged.pop(col)
        merged.insert(source_idx + 1, col, values)
        source_idx += 1
    to_csv(summary_path, merged)


def update_source_count_summary(task_key: str, unique_df: pd.DataFrame) -> None:
    cfg = TASKS[task_key]
    out_path = cfg["root"] / "tables" / cfg["source_count_summary"]
    df = unique_df.copy()
    if "Hydrolysis_Source_Count" not in df.columns:
        return
    df["Hydrolysis_Source_Count"] = pd.to_numeric(df["Hydrolysis_Source_Count"], errors="coerce").fillna(1).astype(int)
    df["Final_Prob"] = pd.to_numeric(df["Final_Prob"], errors="coerce")
    rows = []
    for hydrolysis_count, group in df.groupby("Hydrolysis_Source_Count", sort=True):
        reported = int(group["Reported_Status"].eq("Reported").sum())
        n = int(len(group))
        row = {
            "hydrolysis_count": int(hydrolysis_count),
            "N": n,
            "Reported": reported,
            "No_reported": n - reported,
            "Final_lt_0_50": int((group["Final_Prob"] < 0.50).sum()),
            "Final_gte_0_50": int((group["Final_Prob"] >= 0.50).sum()),
            "Final_gte_0_85": int((group["Final_Prob"] >= 0.85).sum()),
            "Final_gte_0_90": int((group["Final_Prob"] >= 0.90).sum()),
            "Final_gte_0_95": int((group["Final_Prob"] >= 0.95).sum()),
            "Final_gte_0_99": int((group["Final_Prob"] >= 0.99).sum()),
            "Final_gte_0_999": int((group["Final_Prob"] >= 0.999).sum()),
            "Mean_Final_Prob": float(group["Final_Prob"].mean()),
            "Median_Final_Prob": float(group["Final_Prob"].median()),
            "Mean_Length": float(pd.to_numeric(group["length"], errors="coerce").mean()),
        }
        row["Final_Positive_Rate_0_50"] = row["Final_gte_0_50"] / n if n else 0.0
        rows.append(row)
    to_csv(out_path, pd.DataFrame(rows))


def update_high_confidence_files(task_key: str, reported_sequences: set[str]) -> None:
    cfg = TASKS[task_key]
    high_conf_dir = cfg["root"] / cfg["high_conf_dir"]
    task_label = cfg["label"]

    for file_name in cfg["high_conf_files"]:
        path = high_conf_dir / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path, keep_default_na=False)
        df["sequence"] = normalize_sequence_series(df["sequence"])
        df["Reported_Status"] = df["sequence"].isin(reported_sequences).map({True: "Reported", False: "No reported"})
        if "sequence_norm" in df.columns:
            df["sequence_norm"] = df["sequence"]
        to_csv(path, df, compression=GZIP_COMPRESSION)
        print(f"{task_label:6s} updated high-confidence status: {file_name}")

    main_file = high_conf_dir / cfg["high_conf_files"][0]
    if main_file.exists():
        main_df = pd.read_csv(main_file, keep_default_na=False)
        update_summary_json(high_conf_dir / cfg["high_conf_summary"], main_df)
        update_by_method_csv(high_conf_dir / cfg["high_conf_by_method"], main_df)

    minlen_file_name = cfg["high_conf_files"][1] if len(cfg["high_conf_files"]) > 1 else None
    minlen_file = high_conf_dir / minlen_file_name if minlen_file_name else None
    if minlen_file and minlen_file.exists():
        minlen_df = pd.read_csv(minlen_file, keep_default_na=False)
        minlen_summary_name = cfg.get("high_conf_minlen_summary")
        if minlen_summary_name:
            update_summary_json(high_conf_dir / str(minlen_summary_name), minlen_df)
        minlen_by_method_name = cfg.get("high_conf_minlen_by_method")
        if minlen_by_method_name:
            update_by_method_csv(high_conf_dir / str(minlen_by_method_name), minlen_df)


def update_summary_json(path: Path, df: pd.DataFrame) -> None:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data["high_confidence_rows"] = int(len(df))
    data["reported_rows"] = int(df["Reported_Status"].eq("Reported").sum()) if "Reported_Status" in df.columns else 0
    data["no_reported_rows"] = int(df["Reported_Status"].eq("No reported").sum()) if "Reported_Status" in df.columns else int(len(df))
    if "Final_Prob" in df.columns:
        data["mean_final_prob"] = float(pd.to_numeric(df["Final_Prob"], errors="coerce").mean())
        data["median_final_prob"] = float(pd.to_numeric(df["Final_Prob"], errors="coerce").median())
    if "length" in df.columns:
        data["mean_length"] = float(pd.to_numeric(df["length"], errors="coerce").mean())
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update_by_method_csv(path: Path, df: pd.DataFrame) -> None:
    if not path.exists():
        return
    current = pd.read_csv(path, keep_default_na=False)
    rows = []
    for method_id, method_name in METHODS:
        method_col = f"from_{method_id}"
        if method_col not in df.columns:
            continue
        flags = df[method_col].map(lambda value: bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"})
        subset = df[flags].copy()
        count_col = "high_confidence_rows" if "high_confidence_rows" in current.columns else "high_confidence_count"
        row = {
            "method_id": method_id,
            "method_name": method_name,
            count_col: int(len(subset)),
            "reported": int(subset["Reported_Status"].eq("Reported").sum()) if "Reported_Status" in subset.columns else 0,
            "no_reported": int(subset["Reported_Status"].eq("No reported").sum()) if "Reported_Status" in subset.columns else int(len(subset)),
            "mean_final_prob": float(pd.to_numeric(subset["Final_Prob"], errors="coerce").mean()) if len(subset) else pd.NA,
            "median_final_prob": float(pd.to_numeric(subset["Final_Prob"], errors="coerce").median()) if len(subset) else pd.NA,
            "mean_length": float(pd.to_numeric(subset["length"], errors="coerce").mean()) if len(subset) else pd.NA,
        }
        if "median_final_prob" not in current.columns:
            row.pop("median_final_prob")
        rows.append(row)
    to_csv(path, pd.DataFrame(rows))


def rebuild_combined_manifest(reported_manifests: list[pd.DataFrame]) -> None:
    manifest_path = VIRTUAL_DIR / "virtual_screening_manifest.csv"
    existing = pd.read_csv(manifest_path, keep_default_na=False)
    probability_rows = existing[
        ~existing["Probability group"].astype(str).isin(["Reported", "No reported"])
    ].copy()
    combined = pd.concat([*reported_manifests, probability_rows], ignore_index=True, sort=False)

    method_order = {method_id: idx for idx, (method_id, _) in enumerate(METHODS)}
    tier_order = {
        "Reported": 0,
        "No reported": 1,
        "< 0.50": 2,
        ">= 0.50": 3,
        ">= 0.85": 4,
        ">= 0.90": 5,
        ">= 0.95": 6,
    }
    task_order = {"Bitter": 0, "Umami": 1}
    combined["_task_order"] = combined["Task"].map(task_order).fillna(99)
    combined["_method_order"] = combined["Hydrolysis_Method_ID"].map(method_order).fillna(99)
    combined["_tier_order"] = combined["Probability group"].map(tier_order).fillna(99)
    combined = combined.sort_values(["_task_order", "_method_order", "_tier_order"]).drop(columns=["_task_order", "_method_order", "_tier_order"])
    combined = combined[UNIFIED_MANIFEST_COLUMNS]
    to_csv(manifest_path, combined)

    bitter_manifest = combined[combined["Task"].eq("Bitter")].copy()
    to_csv(VIRTUAL_DIR / "bitter" / "virtual_digest" / "tables" / "virtual_screening_manifest_bitter.csv", bitter_manifest)


def main() -> None:
    reported_manifests = []
    for task_key, cfg in TASKS.items():
        reported = load_reported_sequences(cfg["label"])
        print(f"{cfg['label']} reported database sequences: {len(reported):,}")
        reported_manifests.append(export_reported_files_for_task(task_key, reported))
        update_high_confidence_files(task_key, reported)
    rebuild_combined_manifest(reported_manifests)
    print("Updated virtual screening reported/no-reported assets.")


if __name__ == "__main__":
    main()
