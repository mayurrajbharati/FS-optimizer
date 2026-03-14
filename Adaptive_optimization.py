# ==========================================================
# Filesystem Performance Model Comparison
# ==========================================================

import pandas as pd
import numpy as np
import joblib


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# ==========================================================
# 1 Load Dataset
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv("final_dataset.csv")

print("Dataset shape:", df.shape)


# ==========================================================
# 2 Feature Engineering
# ==========================================================

print("\nPerforming feature engineering...")

df["read_ahead_ratio"] = df["read_ahead_sizes"] / df["block_size"]

df["commit_density"] = df["commit_intervals"] / df["block_size"]

pattern_map = {
    "randread": "random",
    "randwrite": "random",
    "randrw": "random",
    "read": "sequential",
    "write": "sequential",
    "rw": "sequential"
}

df["workload_pattern"] = df["workload_type"].map(pattern_map)

scheduler_class_map = {
    "bfq": "fairness",
    "kyber": "latency",
    "none": "latency"
}

df["scheduler_class"] = df["io_schedulers"].map(scheduler_class_map)


# ==========================================================
# 3 Targets
# ==========================================================

target_cols = ["mean_latency", "bandwidth", "iops"]

X = df.drop(columns=target_cols)

y = df[target_cols]


# ==========================================================
# 4 Feature Types
# ==========================================================

categorical_cols = [
    "workload_type",
    "journal_modes",
    "io_schedulers",
    "barriers",
    "noatime_options",
    "io_engines",
    "block_size",
    "read_ahead_sizes",
    "commit_intervals",
    "workload_pattern",
    "scheduler_class"
]

numeric_cols = [
    "read_ahead_ratio",
    "commit_density"
]


# ==========================================================
# 5 Preprocessing
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


# ==========================================================
# 6 Train Test Split
# ==========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# 7 Define Models
# ==========================================================

models = {

    "LightGBM": MultiOutputRegressor(
        LGBMRegressor(random_state=42)
    ),

    "XGBoost": MultiOutputRegressor(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42
        )
    ),

    "RandomForest": MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    ),

    "ExtraTrees": MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=200,
            random_state=42
        )
    )

}


# ==========================================================
# 8 Train Models
# ==========================================================

results_overall = {}

latency_scores = {}
bandwidth_scores = {}
iops_scores = {}

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{name} Overall R2:", r2)
    print(f"{name} RMSE:", rmse)

    results_overall[name] = r2

    latency_scores[name] = r2_score(y_test["mean_latency"], preds[:,0])
    bandwidth_scores[name] = r2_score(y_test["bandwidth"], preds[:,1])
    iops_scores[name] = r2_score(y_test["iops"], preds[:,2])

    trained_models[name] = pipeline


# ==========================================================
# 9 Save Best Model
# ==========================================================

best_model_name = max(results_overall, key=results_overall.get)

print("\nBest Model:", best_model_name)

best_model = trained_models[best_model_name]

joblib.dump(best_model, "best_filesystem_model.pkl")

print("Best model saved as best_filesystem_model.pkl")


# ==========================================================
# 10 Leaderboard Graph
# ==========================================================

print("\nGenerating model leaderboard graph...")

results_df = pd.DataFrame({
    "Model": list(results_overall.keys()),
    "R2 Score": list(results_overall.values())
})

results_df = results_df.sort_values(by="R2 Score", ascending=False)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,5))

ax = sns.barplot(
    data=results_df,
    x="R2 Score",
    y="Model",
    palette="viridis"
)

for i, v in enumerate(results_df["R2 Score"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center")

plt.title("Filesystem Performance Model Leaderboard", fontsize=14, weight="bold")

plt.xlabel("R² Score")

plt.ylabel("Model")

plt.xlim(0.7,1)

plt.tight_layout()

plt.savefig("model_leaderboard.png", dpi=300)

plt.show()


# ==========================================================
# 11 Metric-wise Graphs
# ==========================================================

metrics = {
    "Latency R²": latency_scores,
    "Bandwidth R²": bandwidth_scores,
    "IOPS R²": iops_scores
}

for metric_name, metric_scores in metrics.items():

    metric_df = pd.DataFrame({
        "Model": list(metric_scores.keys()),
        "Score": list(metric_scores.values())
    })

    metric_df = metric_df.sort_values(by="Score", ascending=False)

    plt.figure(figsize=(8,5))

    ax = sns.barplot(
        data=metric_df,
        x="Score",
        y="Model",
        palette="mako"
    )

    for i, v in enumerate(metric_df["Score"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")

    plt.title(metric_name + " Comparison", fontsize=13)

    plt.xlabel("R² Score")

    plt.ylabel("Model")

    plt.xlim(0.7,1)

    plt.tight_layout()

    filename = metric_name.replace(" ", "_").lower() + ".png"

    plt.savefig(filename, dpi=300)

    plt.show()