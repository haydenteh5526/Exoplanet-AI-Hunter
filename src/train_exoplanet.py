#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust exoplanet training script for NASA tables (Kepler/K2/TESS).

- Handles NASA CSVs that start with '# ...' comment lines + a later "rowid,..." header.
- Builds a unified binary label: label_3 = PLANET vs NOT
  * Kepler: koi_disposition == CONFIRMED -> PLANET
  * K2:     disposition in {CONFIRMED, CP} -> PLANET
  * TESS:   tfopwg_disp in {CP, CONFIRMED, CONFIRMED PLANET} -> PLANET
- Merges all three datasets, keeps identifiers for name mapping, drops non-numeric features,
  deduplicates (by best-available object ID), trains a baseline (impute+scale+logreg),
  and saves artifacts:
    outdir/
      model.joblib
      features.json
      metrics.json
      aggregated.parquet   # with IDs, source, label, and the numeric features used
"""

import argparse, json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

from typing import List

def resolve_paths(paths: List[str]) -> List[Path]:
    """
    Resolve input paths relative to:
    - current working dir,
    - repo root (two levels up from this file),
    - data/raw/<filename> under repo root (if only a filename was given)
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # src/ -> repo root
    resolved = []
    for p in paths:
        pth = Path(p)
        if pth.exists():
            resolved.append(pth)
            continue
        # try relative to repo root
        cand = (repo_root / p).resolve()
        if cand.exists():
            resolved.append(cand); continue
        # if only a filename was given, try data/raw/<filename>
        if pth.name and (repo_root / "data" / "raw" / pth.name).exists():
            resolved.append((repo_root / "data" / "raw" / pth.name).resolve()); continue
        raise FileNotFoundError(f"Could not find data file: {p}")
    return resolved


# Columns we consider as "identifiers" to keep for name mapping (not used as features)
ID_CANDIDATES = [
    "kepoi_name","kepler_name","kepid",           # Kepler
    "pl_name","epic_id","hostname",               # K2
    "toi","tid","tic_id","ctoi_alias"             # TESS
]

# ---------- Robust NASA reader ----------

def read_nasa_table(path: Path) -> pd.DataFrame:
    """
    NASA Archive files often start with comment lines ('# ...') describing the schema,
    and only later contain the actual CSV header (starts with 'rowid,').
    This function finds that header line and reads the table from there.
    """
    path = Path(path)
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.lower().startswith("rowid,"):
                header_row = i
                break
        else:
            header_row = 0  # fallback

    df = pd.read_csv(path, skiprows=header_row)
    df["__source__"] = path.stem  # provenance
    return df

# ---------- Label builders per mission ----------

def add_label_kepler(df: pd.DataFrame) -> pd.DataFrame: ## problem with this is it adds a new column to the data csv to say "Exoplanet" or "Not": to narrow it down to 2 options
    cols = {c.lower(): c for c in df.columns} ##could be avoided if we just made the panda pd read csv ignore case
    if "koi_disposition" not in cols:
        raise ValueError("Kepler file missing 'koi_disposition'")
    col = cols["koi_disposition"]
    lab = df[col].astype(str).str.upper()
    df["label_3"] = np.where(lab.str.contains("CONFIRMED"), "PLANET", "NOT")
    return df

def add_label_k2(df: pd.DataFrame) -> pd.DataFrame:
    # K2 tables vary; most have 'disposition'
    disp_col = None
    for key in ["disposition", "k2_disposition"]:
        for c in df.columns:
            if c.lower() == key:
                disp_col = c
                break
        if disp_col: break
    if disp_col is None:
        # any column containing 'disposition'
        for c in df.columns:
            if "disposition" in c.lower():
                disp_col = c
                break
    if disp_col is None:
        raise ValueError("K2 file missing a disposition column")

    lab = df[disp_col].astype(str).str.upper().str.strip()
    df["label_3"] = np.where((lab == "CONFIRMED") | (lab == "CP") | (lab.str.contains("CONFIRMED")), "PLANET", "NOT")
    return df

def add_label_tess(df: pd.DataFrame) -> pd.DataFrame:
    # TOI tables typically have 'tfopwg_disp' (values: CP, FP, PC)
    disp_col = None
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "tfopwg_disp" or ("tfopwg" in lc and "disp" in lc):
            disp_col = c
            break
    if disp_col is None:
        raise ValueError("TESS file missing 'tfopwg_disp'")

    lab = df[disp_col].astype(str).str.upper().str.strip()
    df["label_3"] = np.where(lab.isin(["CP", "CONFIRMED", "CONFIRMED PLANET"]), "PLANET", "NOT")
    return df

# ---------- Feature building & dedupe ----------

def detect_id_cols(df: pd.DataFrame):
    return [c for c in ID_CANDIDATES if c in df.columns]

def build_feature_matrix(df: pd.DataFrame, id_cols):
    # Drop label + id + source columns; keep numeric only; drop 'rowid'
    X = df.drop(columns=["label_3"] + id_cols + ["__source__"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    if "rowid" in X.columns:
        X = X.drop(columns=["rowid"])
    # Remove constant columns
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]
    return X

def compute_metrics(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(acc),
        "precision_pos": float(prec),
        "recall_pos": float(rec),
        "f1_pos": float(f1),
    }
    if len(set(y_true)) == 2:
        try: out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception: out["roc_auc"] = None
        try: out["pr_auc_pos"] = float(average_precision_score(y_true, y_prob))
        except Exception: out["pr_auc_pos"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc_pos"] = None
    return out

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Train exoplanet classifier on Kepler+K2+TESS NASA tables.")
    ap.add_argument("--data", nargs=3, required=True,
                    help="Paths to the three files: cumulative.csv (Kepler), k2_pandas.csv (K2), TOI.csv (TESS)") ## should automatically read, the user isnt going to input this
    ap.add_argument("--outdir", required=True, help="Directory to write artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    kep_path, k2_path, tess_path = resolve_paths(args.data)

    # 1) Read tables (robust header detection)
    kepler = read_nasa_table(kep_path)
    k2     = read_nasa_table(k2_path)
    tess   = read_nasa_table(tess_path)

    # 2) Add label_3 per mission
    kepler["__source__"] = "kepler"; kepler = add_label_kepler(kepler)
    k2["__source__"]     = "k2";     k2     = add_label_k2(k2)
    tess["__source__"]   = "tess";   tess   = add_label_tess(tess)

    # 3) Merge
    all_df = pd.concat([kepler, k2, tess], ignore_index=True)
    all_df = all_df.copy()  # defragment to avoid PerformanceWarning on later inserts

    # 4) Keep track of IDs and object id
    id_cols = detect_id_cols(all_df)

    # Build a single object ID per row without .apply (avoids fragmentation)
    if id_cols:
        vals = all_df[id_cols].to_numpy()
        mask = ~pd.isna(vals)
        first_any = mask.any(axis=1)
        first_idx = mask.argmax(axis=1)
        obj_ids = np.where(first_any, vals[np.arange(len(vals)), first_idx], None)
        all_df["__obj_id__"] = obj_ids
    else:
        all_df["__obj_id__"] = None

    # 5) Deduplicate (prefer by source+object_id+label)
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["__source__", "__obj_id__", "label_3"], keep="last")
    after = len(all_df)
    if after < before:
        print(f"[info] Removed {before - after} duplicate rows")

    # 6) Build feature matrix + target
    X = build_feature_matrix(all_df, id_cols + ["__obj_id__"])
    y = (all_df["label_3"] == "PLANET").astype(int).values

    # 7) Train/val split & model
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"))
    ])
    pipe.fit(Xtr, ytr)
    yprob = pipe.predict_proba(Xva)[:, 1]
    metrics = compute_metrics(yva, yprob)
    metrics.update({
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_features": int(X.shape[1] ),
        "class_balance_overall": float(y.mean())
    })
    print(json.dumps(metrics, indent=2))

    # 8) Save artifacts
    joblib.dump(pipe, outdir / "model.joblib")
    (outdir / "features.json").write_text(json.dumps(list(X.columns), indent=2))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- PATCH 1: ensure string dtypes for Arrow ---
    # Coerce ID-like columns and __source__/__obj_id__ to pandas nullable string
    for c in (id_cols + ["__obj_id__", "__source__"]):
        if c in all_df.columns:
            all_df[c] = all_df[c].astype("string")

    # Save the aggregated dataset with IDs + source + label + features used
    cols_keep = id_cols + ["__source__", "label_3", "__obj_id__"] + list(X.columns)
    all_df.loc[:, cols_keep].to_parquet(outdir / "aggregated.parquet", index=False)
    print(f"[ok] Artifacts saved under: {outdir}")

if __name__ == "__main__":
    main()
