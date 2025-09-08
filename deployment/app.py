import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError
import joblib
import os
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="praneeth232/medical_insurance_model",
    filename="best_medical_insurance_model_v1.joblib"
)
model = joblib.load(model_path)

# Hugging Face datasets for baseline and logs
BASELINE_DATASET_REPO = "praneeth232/medical-insurance-cost-prediction"
LOGS_DATASET_REPO = "praneeth232/medical-insurance-logs"

api = HfApi(token=os.getenv("HF_TOKEN"))

def _ensure_logs_repo():
    try:
        api.repo_info(repo_id=LOGS_DATASET_REPO, repo_type="dataset")
    except RepositoryNotFoundError:
        create_repo(repo_id=LOGS_DATASET_REPO, repo_type="dataset", private=False)

def _download_csv(repo_id: str, filename: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_baseline() -> pd.DataFrame:
    # Xtrain from training dataset repo
    Xtrain = _download_csv(BASELINE_DATASET_REPO, "Xtrain.csv")
    ytrain = _download_csv(BASELINE_DATASET_REPO, "ytrain.csv")
    if not Xtrain.empty and not ytrain.empty:
        Xtrain = Xtrain.copy()
        Xtrain["charges"] = ytrain.values.squeeze()
    return Xtrain

def load_logs() -> pd.DataFrame:
    # consolidated logs file
    return _download_csv(LOGS_DATASET_REPO, "logs.csv")

def append_log(record: dict):
    _ensure_logs_repo()
    logs_df = load_logs()
    new_row = pd.DataFrame([record])
    if logs_df.empty:
        combined = new_row
    else:
        combined = pd.concat([logs_df, new_row], ignore_index=True)
    # Save to temp and upload as logs.csv (overwrite)
    tmp_path = "logs.csv"
    combined.to_csv(tmp_path, index=False)
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="logs.csv",
        repo_id=LOGS_DATASET_REPO,
        repo_type="dataset",
    )

def ks_tests(new_df: pd.DataFrame, baseline_df: pd.DataFrame, numeric_cols):
    results = []
    for col in numeric_cols:
        if col in new_df.columns and col in baseline_df.columns:
            try:
                stat, p = stats.ks_2samp(baseline_df[col].dropna(), new_df[col].dropna())
                results.append({"feature": col, "test": "KS", "stat": stat, "p_value": p, "drift": p < 0.05})
            except Exception:
                results.append({"feature": col, "test": "KS", "stat": np.nan, "p_value": np.nan, "drift": False})
    return pd.DataFrame(results)

def chi2_tests(new_df: pd.DataFrame, baseline_df: pd.DataFrame, cat_cols):
    results = []
    for col in cat_cols:
        if col in new_df.columns and col in baseline_df.columns:
            try:
                base_counts = baseline_df[col].value_counts()
                new_counts = new_df[col].value_counts()
                # align categories
                all_idx = base_counts.index.union(new_counts.index)
                base = base_counts.reindex(all_idx, fill_value=0).values
                new = new_counts.reindex(all_idx, fill_value=0).values
                chi2, p = stats.chisquare(f_obs=new, f_exp=(base + 1))  # add-1 smoothing
                results.append({"feature": col, "test": "Chi2", "stat": chi2, "p_value": p, "drift": p < 0.05})
            except Exception:
                results.append({"feature": col, "test": "Chi2", "stat": np.nan, "p_value": np.nan, "drift": False})
    return pd.DataFrame(results)

def compute_metrics(df: pd.DataFrame):
    df = df.dropna(subset=["ground_truth"]) if "ground_truth" in df.columns else pd.DataFrame()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    y_true = df["ground_truth"].values
    y_pred = df["predicted_charges"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def concept_drift_auc(baseline: pd.DataFrame, new: pd.DataFrame):
    try:
        common_cols = [c for c in ["age","bmi","children","sex","smoker","region","charges"] if c in baseline.columns and c in new.columns]
        if len(common_cols) < 3:
            return np.nan
        b = baseline[common_cols].copy(); b["domain"] = 0
        n = new[common_cols].copy(); n["domain"] = 1
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
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        from sklearn.metrics import roc_auc_score
        proba = clf.predict_proba(X_test)[:,1]
        return roc_auc_score(y_test, proba)
    except Exception:
        return np.nan

# Streamlit UI for Insurance Charges Prediction
st.title("Insurance Charges Prediction App")
st.write("""
This application predicts the **medical insurance charges** based on personal and lifestyle details.
Please enter the required information below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}])

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Performance Metrics", "Feature Distributions", "Concept Drift Analysis", "Fairness Analysis"])

numeric_cols = ['age','bmi','children']
categorical_cols = ['sex','smoker','region']

if page == "Predict":
    st.header("Make a Prediction")
    if st.button("Predict Charges"):
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction Result:")
        st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")

        # Optional ground truth entry
        with st.expander("Add ground truth (optional)"):
            gt = st.number_input("Actual charges (if known)", min_value=0.0, value=0.0, step=100.0)

        record = {
            **input_data.iloc[0].to_dict(),
            "predicted_charges": float(prediction),
            "ground_truth": float(gt) if gt and gt > 0 else np.nan,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            append_log(record)
            st.info("Logged prediction to Hugging Face dataset.")
        except Exception as e:
            st.warning(f"Failed to log prediction: {e}")

elif page == "Feature Distributions":
    st.header("Data Drift Monitoring")
    baseline = load_baseline()
    logs = load_logs()
    if baseline.empty or logs.empty:
        st.warning("Baseline or logs unavailable.")
    else:
        ks_df = ks_tests(logs, baseline, numeric_cols)
        chi_df = chi2_tests(logs, baseline, categorical_cols)
        st.subheader("Statistical Tests")
        st.dataframe(pd.concat([ks_df, chi_df], ignore_index=True))
        st.subheader("Feature Distributions (baseline vs. new)")
        for col in numeric_cols + categorical_cols:
            if col in logs.columns and col in baseline.columns:
                st.write(f"Distribution for {col}")
                st.line_chart(pd.DataFrame({
                    "baseline": baseline[col].value_counts(normalize=True, bins=20 if col in numeric_cols else None, sort=False),
                    "new": logs[col].value_counts(normalize=True, bins=20 if col in numeric_cols else None, sort=False)
                }).fillna(0))

elif page == "Performance Metrics":
    st.header("Model Drift Monitoring")
    logs = load_logs()
    if logs.empty or "ground_truth" not in logs.columns:
        st.warning("Not enough logged data with ground truth.")
    else:
        metrics = compute_metrics(logs.dropna(subset=["ground_truth"]))
        st.metric("MAE", f"{metrics['MAE']:.2f}" if not np.isnan(metrics['MAE']) else "NA")
        st.metric("RMSE", f"{metrics['RMSE']:.2f}" if not np.isnan(metrics['RMSE']) else "NA")
        st.metric("R²", f"{metrics['R2']:.3f}" if not np.isnan(metrics['R2']) else "NA")
        # Trend plot by day
        logs_ts = logs.copy()
        logs_ts["date"] = pd.to_datetime(logs_ts["timestamp"]).dt.date
        logs_ts = logs_ts.dropna(subset=["ground_truth"]).groupby("date").apply(lambda d: compute_metrics(d)).apply(pd.Series)
        if not logs_ts.empty:
            st.line_chart(logs_ts[["MAE","RMSE","R2"]])
        # Simple alert: compare last RMSE to mean
        if not logs_ts.empty and not np.isnan(logs_ts["RMSE"].iloc[-1]):
            recent = logs_ts["RMSE"].iloc[-1]
            baseline_rmse = logs_ts["RMSE"].iloc[:-1].mean() if len(logs_ts) > 1 else recent
            if baseline_rmse and recent > 1.2 * baseline_rmse:
                st.error("Alert: RMSE increased by >20% compared to baseline.")

elif page == "Concept Drift Analysis":
    st.header("Concept Drift Analysis")
    baseline = load_baseline()
    logs = load_logs()
    if baseline.empty or logs.empty or "predicted_charges" not in logs.columns:
        st.warning("Insufficient data for concept drift.")
    else:
        # For concept drift, include target as predicted/ground truth when available
        new_df = logs.copy()
        new_df["charges"] = new_df["ground_truth"].fillna(new_df["predicted_charges"]) if "ground_truth" in new_df.columns else new_df["predicted_charges"]
        auc = concept_drift_auc(baseline, new_df)
        st.write(f"Domain classifier ROC-AUC: {auc if not np.isnan(auc) else 'NA'}")
        if not np.isnan(auc) and auc > 0.7:
            st.error("Concept drift detected (AUC > 0.7)")

elif page == "Fairness Analysis":
    st.header("Fairness Monitoring")
    logs = load_logs()
    if logs.empty or "ground_truth" not in logs.columns:
        st.warning("Not enough logged data with ground truth.")
    else:
        df = logs.dropna(subset=["ground_truth"]).copy()
        df["abs_error"] = (df["ground_truth"] - df["predicted_charges"]).abs()
        for group_col in ["sex","smoker"]:
            if group_col in df.columns:
                st.subheader(f"Group: {group_col}")
                grp = df.groupby(group_col)["abs_error"].mean()
                st.bar_chart(grp)
                if grp.max() > 1.2 * grp.min():
                    st.error(f"Fairness alert: {group_col} disparity > 20%")
