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
# 10 Leaderboard Graph (Improved)
# ==========================================================

print("\nGenerating model leaderboard graph...")

results_df = pd.DataFrame({
    "Model": list(results_overall.keys()),
    "R2 Score": list(results_overall.values())
})

results_df = results_df.sort_values(by="R2 Score", ascending=True)

plt.figure(figsize=(10,6))

sns.set_style("whitegrid")

colors = sns.color_palette("viridis", len(results_df))

bars = plt.barh(
    results_df["Model"],
    results_df["R2 Score"],
    color=colors,
    edgecolor="black"
)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va='center',
        fontsize=11
    )

plt.title(
    "Filesystem Performance Model Leaderboard",
    fontsize=18,
    weight="bold"
)

plt.xlabel("R² Score", fontsize=13)
plt.ylabel("Model", fontsize=13)

# Dynamic axis range
plt.xlim(min(results_df["R2 Score"]) - 0.05, 1)

plt.grid(axis="x", linestyle="--", alpha=0.6)

plt.tight_layout()

plt.savefig("model_leaderboard.png", dpi=400)

plt.show()