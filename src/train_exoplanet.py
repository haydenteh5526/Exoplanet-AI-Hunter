#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exoplanet classifier (XGBoost only) for NASA-style tables (Kepler/K2/TESS).

- Reads NASA CSVs that often start with header comments (# ...) and the real
  header later (usually a line beginning with 'rowid,').
- Creates a unified binary label: label_3 = PLANET vs NOT
  * Kepler: koi_disposition == CONFIRMED -> PLANET
  * K2:     disposition in {CONFIRMED, CP} -> PLANET
  * TESS:   tfopwg_disp in {CP, CONFIRMED, CONFIRMED PLANET} -> PLANET
- Merges the three datasets, keeps identifiers for name mapping, drops non-numeric features,
  deduplicates (by mission/source + object id + label), trains XGBoost with
  RandomizedSearchCV + final early-stopping refit, and saves artifacts:
    outdir/
      model.joblib         # trained sklearn Pipeline (drop-NaN-cols -> imputer -> XGBClassifier)
      features.json        # numeric feature list (order matters)
      metrics.json         # validation metrics + best params
      aggregated.parquet   # cleaned, deduped dataset with IDs/source/label/features used
"""

import argparse, json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib

from xgboost.callback import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

from xgboost import XGBClassifier
from scipy.stats import loguniform, randint

# ----------------------------- utility transformers -----------------------------

class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely NaN for the current fit fold."""
    def __init__(self):
        self.keep_cols_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            mask = X.notna().any(axis=0)
            self.keep_cols_ = X.columns[mask].tolist()
        else:
            # If array, keep columns that have any non-nan
            mask = ~np.all(np.isnan(X), axis=0)
            self.keep_cols_ = np.arange(X.shape[1])[mask].tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.keep_cols_]
        else:
            return X[:, self.keep_cols_]

# ----------------------------- path helpers -----------------------------

def resolve_paths(paths: List[str]) -> List[Path]:
    """Resolve input paths relative to repo root or data/raw."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # src/ -> repo root
    resolved = []
    for p in paths:
        pth = Path(p)
        if pth.exists():
            resolved.append(pth); continue
        cand = (repo_root / p).resolve()
        if cand.exists():
            resolved.append(cand); continue
        if pth.name and (repo_root / "data" / "raw" / pth.name).exists():
            resolved.append((repo_root / "data" / "raw" / pth.name).resolve()); continue
        raise FileNotFoundError(f"Could not find data file: {p}")
    return resolved

# ----------------------------- ID columns ------------------------------

ID_CANDIDATES = [
    "kepoi_name","kepler_name","kepid",      # Kepler
    "pl_name","epic_id","hostname",          # K2
    
    "toi","tid","tic_id","ctoi_alias"        # TESS
]

def detect_id_cols(df: pd.DataFrame):
    return [c for c in ID_CANDIDATES if c in df.columns]

# ----------------------------- IO & labels -----------------------------

def read_nasa_table(path: Path) -> pd.DataFrame:
    """Find the real CSV header (line starting with 'rowid,') and read from there."""
    path = Path(path)
    with open(path, "r", errors="ignore") as f:
        header_row = 0
        for i, line in enumerate(f):
            if line.lower().startswith("rowid,"):
                header_row = i; break
    df = pd.read_csv(path, skiprows=header_row)
    df["__source__"] = path.stem
    return df

def add_label_kepler(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    col = cols.get("koi_disposition")
    if col is None:
        raise ValueError("Kepler file missing 'koi_disposition'")
    lab = df[col].astype(str).str.upper()
    df["label_3"] = np.where(lab.str.contains("CONFIRMED"), "PLANET", "NOT")
    return df

def add_label_k2(df: pd.DataFrame) -> pd.DataFrame:
    disp_col = None
    for key in ["disposition", "k2_disposition"]:
        for c in df.columns:
            if c.lower() == key:
                disp_col = c; break
        if disp_col: break
    if disp_col is None:
        for c in df.columns:
            if "disposition" in c.lower():
                disp_col = c; break
    if disp_col is None:
        raise ValueError("K2 file missing a disposition column")
    lab = df[disp_col].astype(str).str.upper().str.strip()
    df["label_3"] = np.where((lab == "CONFIRMED") | (lab == "CP") | (lab.str.contains("CONFIRMED")),
                             "PLANET", "NOT")
    return df

def add_label_tess(df: pd.DataFrame) -> pd.DataFrame:
    disp_col = None
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "tfopwg_disp" or ("tfopwg" in lc and "disp" in lc):
            disp_col = c; break
    if disp_col is None:
        raise ValueError("TESS file missing 'tfopwg_disp'")
    lab = df[disp_col].astype(str).str.upper().str.strip()
    df["label_3"] = np.where(lab.isin(["CP", "CONFIRMED", "CONFIRMED PLANET"]),
                             "PLANET", "NOT")
    return df

# ----------------------------- features & metrics ----------------------

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add simple engineered features (safe if inputs absent)."""
    X = X.copy()
    if 'transit_duration' in X.columns and 'orbital_period' in X.columns:
        X['transit_duration_ratio'] = X['transit_duration'] / X['orbital_period']
    if 'stellar_mass' in X.columns and 'stellar_radius' in X.columns:
        X['stellar_density'] = X['stellar_mass'] / (X['stellar_radius']**3)
    return X

def get_feature_importance(model):
    if hasattr(model, 'named_steps'):
        xgb = model.named_steps['clf']
        importances = xgb.feature_importances_
        features = model.named_steps['imputer'].feature_names_in_
        return pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    return None

def build_feature_matrix(df: pd.DataFrame, id_cols):
    X = df.drop(columns=["label_3"] + id_cols + ["__source__"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    if "rowid" in X.columns:
        X = X.drop(columns=["rowid"])
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
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc_pos"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None; out["pr_auc_pos"] = None
    return out

# ----------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost exoplanet classifier")
    ap.add_argument("--data", nargs=3, required=True,
                    help="Paths to: cumulative.csv (Kepler), k2_pandas.csv (K2), TOI.csv (TESS)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_iter", type=int, default=80, help="RandomizedSearchCV iterations")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Read data
    kep_path, k2_path, tess_path = resolve_paths(args.data)
    kepler = add_label_kepler(read_nasa_table(kep_path))
    k2     = add_label_k2(read_nasa_table(k2_path))
    tess   = add_label_tess(read_nasa_table(tess_path))
    all_df = pd.concat([kepler, k2, tess], ignore_index=True).copy()

    # 2) Build IDs + dedupe
    id_cols = detect_id_cols(all_df)
    if id_cols:
        vals = all_df[id_cols].to_numpy()
        mask = ~pd.isna(vals)
        obj_ids = np.where(mask.any(axis=1), vals[np.arange(len(vals)), mask.argmax(axis=1)], None)
        all_df["__obj_id__"] = obj_ids
    else:
        all_df["__obj_id__"] = None

    before = len(all_df)
    all_df = all_df.drop_duplicates(subset=["__source__", "__obj_id__", "label_3"], keep="last")
    after  = len(all_df)
    if after < before:
        print(f"[info] Removed {before - after} duplicate rows")

    # 3) Features
    X = engineer_features(build_feature_matrix(all_df, id_cols + ["__obj_id__"]))
    y = (all_df["label_3"] == "PLANET").astype(int).values
    print(f"[info] Initial feature count: {X.shape[1]}")

    # 4) Split (hold-out for early stopping)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # 5) Class imbalance
    pos, neg = (ytr == 1).sum(), (ytr == 0).sum()
    spw = float(neg / max(1, pos))
    tree_method = "gpu_hist" if args.gpu else "hist"
    print(f"[info] pos={pos}, neg={neg}, scale_pos_weight={spw:.2f}, tree_method={tree_method}")

    # 6) RandomizedSearchCV (NO early stopping here)
    xgb_pipe = Pipeline([
        ("dropnan", DropAllNaNColumns()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method=tree_method,
            n_estimators=2000,
            n_jobs=-1,
            random_state=args.seed
        ))
    ])

    param_dist = {
        "clf__max_depth": randint(3, 9),
        "clf__learning_rate": loguniform(1e-3, 3e-1),
        "clf__subsample": loguniform(0.5, 1.0),
        "clf__colsample_bytree": loguniform(0.5, 1.0),
        "clf__min_child_weight": loguniform(0.2, 10),
        "clf__gamma": loguniform(1e-4, 10),
        "clf__reg_alpha": loguniform(1e-4, 10),
        "clf__reg_lambda": loguniform(1e-4, 10),
        "clf__scale_pos_weight": [spw],
    }

    search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="average_precision",
        n_jobs=-1,
        cv=3,
        verbose=1,
        random_state=args.seed
    )
    # IMPORTANT: do NOT pass eval_set/early_stopping here
    search.fit(Xtr, ytr)
    best_params = search.best_params_
    print("[info] Best params:", best_params)

    # 7) Final refit with early stopping on (Xva, yva)
    best_pipe = Pipeline([
        ("dropnan", DropAllNaNColumns()),
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method=tree_method,
            n_estimators=2000,
            n_jobs=-1,
            random_state=args.seed,
            **{k.replace("clf__", ""): v for k, v in best_params.items()}
        ))
    ])
    best_pipe.fit(
        Xtr, ytr,
        clf__eval_set=[(Xva, yva)],
        clf__callbacks=[EarlyStopping(rounds=50, save_best=True, maximize=True)],
        clf__verbose=False
    )


    # 8) Metrics
    yprob = best_pipe.predict_proba(Xva)[:, 1]
    metrics = compute_metrics(yva, yprob)
    metrics.update({
        "best_params": best_params,
        "cv_best_score": float(search.best_score_),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "n_features": int(X.shape[1]),
        "class_balance_overall": float(y.mean())
    })

    importance_df = get_feature_importance(best_pipe)
    if importance_df is not None:
        metrics["feature_importance"] = importance_df.head(20).to_dict(orient="records")
        print("\nTop features:\n", importance_df.head(10).to_string(index=False))

    print("\nModel metrics:\n", json.dumps(metrics, indent=2))

    # 9) Save artifacts
    joblib.dump(best_pipe, outdir / "model.joblib")
    (outdir / "features.json").write_text(json.dumps(list(X.columns), indent=2))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save merged dataset for traceability
    for c in (id_cols + ["__obj_id__", "__source__"]):
        if c in all_df.columns:
            all_df[c] = all_df[c].astype("string")
    cols_keep = id_cols + ["__source__", "label_3", "__obj_id__"] + list(X.columns)
    all_df.loc[:, cols_keep].to_parquet(outdir / "aggregated.parquet", index=False)
    print(f"[ok] Artifacts saved under: {outdir}")

if __name__ == "__main__":
    main()
