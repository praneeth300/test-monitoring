import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


BASELINE_DATASET_REPO = os.getenv("BASELINE_DATASET_REPO", "praneeth232/medical-insurance-cost-prediction")
LOGS_DATASET_REPO = os.getenv("LOGS_DATASET_REPO", "praneeth232/medical-insurance-logs")
REPORT_DIR = os.getenv("REPORT_DIR", "monitoring_reports")


def _download_csv(repo_id: str, filename: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_baseline() -> pd.DataFrame:
    Xtrain = _download_csv(BASELINE_DATASET_REPO, "Xtrain.csv")
    ytrain = _download_csv(BASELINE_DATASET_REPO, "ytrain.csv")
    if not Xtrain.empty and not ytrain.empty:
        Xtrain = Xtrain.copy()
        Xtrain["charges"] = ytrain.values.squeeze()
    return Xtrain


def load_logs() -> pd.DataFrame:
    return _download_csv(LOGS_DATASET_REPO, "logs.csv")


def ks_tests(new_df: pd.DataFrame, baseline_df: pd.DataFrame, numeric_cols):
    rows = []
    for col in numeric_cols:
        if col in new_df.columns and col in baseline_df.columns:
            try:
                stat, p = stats.ks_2samp(baseline_df[col].dropna(), new_df[col].dropna())
                rows.append({"feature": col, "test": "KS", "stat": float(stat), "p_value": float(p), "drift": bool(p < 0.05)})
            except Exception:
                rows.append({"feature": col, "test": "KS", "stat": np.nan, "p_value": np.nan, "drift": False})
    return pd.DataFrame(rows)


def chi2_tests(new_df: pd.DataFrame, baseline_df: pd.DataFrame, cat_cols):
    rows = []
    for col in cat_cols:
        if col in new_df.columns and col in baseline_df.columns:
            try:
                base_counts = baseline_df[col].value_counts()
                new_counts = new_df[col].value_counts()
                idx = base_counts.index.union(new_counts.index)
                base = base_counts.reindex(idx, fill_value=0).values
                new = new_counts.reindex(idx, fill_value=0).values
                chi2, p = stats.chisquare(f_obs=new, f_exp=(base + 1))
                rows.append({"feature": col, "test": "Chi2", "stat": float(chi2), "p_value": float(p), "drift": bool(p < 0.05)})
            except Exception:
                rows.append({"feature": col, "test": "Chi2", "stat": np.nan, "p_value": np.nan, "drift": False})
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    y_true = df["ground_truth"].values
    y_pred = df["predicted_charges"].values
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def concept_drift_auc(baseline: pd.DataFrame, new: pd.DataFrame):
    try:
        cols = [c for c in ["age","bmi","children","sex","smoker","region","charges"] if c in baseline.columns and c in new.columns]
        if len(cols) < 3:
            return np.nan
        b = baseline[cols].copy(); b["domain"] = 0
        n = new[cols].copy(); n["domain"] = 1
        data = pd.concat([b, n], ignore_index=True)
        y = data.pop("domain").values
        num_cols = [c for c in ["age","bmi","children","charges"] if c in data.columns]
        cat_cols = [c for c in ["sex","smoker","region"] if c in data.columns]
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        clf = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=1000))
        ])
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:,1]
        return float(roc_auc_score(y_test, proba))
    except Exception:
        return np.nan


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    baseline = load_baseline()
    logs = load_logs()

    report = {"generated_at": datetime.utcnow().isoformat()}

    if baseline.empty:
        report["error"] = "Baseline unavailable"
    if logs.empty:
        report["warning"] = "No logs available"

    numeric_cols = ["age","bmi","children"]
    categorical_cols = ["sex","smoker","region"]

    if not baseline.empty and not logs.empty:
        # Data drift
        ks_df = ks_tests(logs, baseline, numeric_cols)
        chi_df = chi2_tests(logs, baseline, categorical_cols)
        drift_df = pd.concat([ks_df, chi_df], ignore_index=True)
        drift_df.to_csv(os.path.join(REPORT_DIR, "data_drift.csv"), index=False)
        report["data_drift_alert"] = bool(drift_df["drift"].any())

        # Model drift (requires ground truth)
        gt_df = logs.dropna(subset=["ground_truth"]) if "ground_truth" in logs.columns else pd.DataFrame()
        if not gt_df.empty:
            metrics_overall = compute_metrics(gt_df)
            report["metrics"] = metrics_overall
            # Daily trend
            trend = gt_df.copy()
            trend["date"] = pd.to_datetime(trend["timestamp"]).dt.date
            trend_metrics = trend.groupby("date").apply(lambda d: pd.Series(compute_metrics(d)))
            trend_metrics.to_csv(os.path.join(REPORT_DIR, "model_metrics_trend.csv"))
            # Alert if last RMSE > 1.2x previous mean
            if len(trend_metrics) >= 1 and not np.isnan(trend_metrics["RMSE"].iloc[-1]):
                recent = trend_metrics["RMSE"].iloc[-1]
                baseline_rmse = trend_metrics["RMSE"].iloc[:-1].mean() if len(trend_metrics) > 1 else recent
                report["model_drift_alert"] = bool(baseline_rmse and recent > 1.2 * baseline_rmse)
        else:
            report["metrics"] = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
            report["model_drift_alert"] = False

        # Concept drift
        new_df = logs.copy()
        if "ground_truth" in new_df.columns and "predicted_charges" in new_df.columns:
            new_df["charges"] = new_df["ground_truth"].fillna(new_df["predicted_charges"]) if not new_df.empty else np.nan
        auc = concept_drift_auc(baseline, new_df)
        report["concept_drift_auc"] = auc
        report["concept_drift_alert"] = bool(not np.isnan(auc) and auc > 0.7)

        # Fairness
        fairness = {}
        if not gt_df.empty:
            gt_df = gt_df.copy()
            gt_df["abs_error"] = (gt_df["ground_truth"] - gt_df["predicted_charges"]).abs()
            for col in ["sex", "smoker"]:
                if col in gt_df.columns:
                    grp = gt_df.groupby(col)["abs_error"].mean()
                    fairness[col] = {k: float(v) for k, v in grp.to_dict().items()}
                    fairness[f"{col}_alert"] = bool(grp.max() > 1.2 * grp.min()) if len(grp) > 1 else False
        report["fairness"] = fairness

    # Save summary
    with open(os.path.join(REPORT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print concise status for CI logs
    print(json.dumps(report))


if __name__ == "__main__":
    main()


