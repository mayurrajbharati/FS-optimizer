# ==========================================================
# Adaptive Filesystem ML Training Pipeline (FINAL UPDATED)
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from scipy.stats import f_oneway
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

df["io_intensity"] = df["read_ahead_sizes"] * df["commit_intervals"]
df["block_commit_ratio"] = df["block_size"] / (df["commit_intervals"] + 1)

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
# 3 ANOVA Feature Selection
# ==========================================================

print("\nRunning ANOVA feature selection...")

candidate_features = [
    "read_ahead_ratio",
    "commit_density",
    "io_intensity",
    "block_commit_ratio"
]

anova_results = []

for feature in candidate_features:
    groups = [
        df[df["workload_pattern"] == cat][feature]
        for cat in df["workload_pattern"].unique()
    ]

    f_stat, p_val = f_oneway(*groups)
    anova_results.append((feature, f_stat, p_val))

    print(f"{feature} -> F={f_stat:.4f}, p={p_val:.6f}")

anova_results = sorted(anova_results, key=lambda x: x[2])
top_features = [x[0] for x in anova_results[:2]]

print("\nTop Selected Features:", top_features)


# ==========================================================
# 4 Targets
# ==========================================================

target_cols = ["mean_latency", "bandwidth", "iops"]
X = df.drop(columns=target_cols)
y = df[target_cols]


# ==========================================================
# 5 Feature Types
# ==========================================================

categorical_cols = [
    "workload_type", "journal_modes", "io_schedulers",
    "barriers", "noatime_options", "io_engines",
    "block_size", "read_ahead_sizes", "commit_intervals",
    "workload_pattern", "scheduler_class"
]

numeric_cols = top_features


# ==========================================================
# 6 Preprocessing
# ==========================================================

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])


# ==========================================================
# 7 Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================================
# 8 Hyperparameter Tuning (MULTIPLE MODELS)
# ==========================================================

print("\nRunning Hyperparameter Optimization...")

tuning_configs = {
    "LightGBM": (
        LGBMRegressor(random_state=42),
        {
            "model__estimator__n_estimators": [100, 200],
            "model__estimator__learning_rate": [0.05, 0.1],
            "model__estimator__max_depth": [-1, 5]
        }
    ),

    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "model__estimator__n_estimators": [100, 200],
            "model__estimator__max_depth": [None, 10],
            "model__estimator__min_samples_split": [2, 5]
        }
    ),

    "XGBoost": (
        XGBRegressor(objective="reg:squarederror", random_state=42),
        {
            "model__estimator__n_estimators": [100, 200],
            "model__estimator__learning_rate": [0.05, 0.1],
            "model__estimator__max_depth": [3, 6]
        }
    )
}

best_models = {}

for name, (model, param_grid) in tuning_configs.items():

    print(f"\nTuning {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", MultiOutputRegressor(model))
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=5,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_
    print(f"{name} Best Params:", search.best_params_)


# ==========================================================
# 9 Additional Models
# ==========================================================

et = ExtraTreesRegressor(n_estimators=200, random_state=42)
ridge = Ridge(alpha=1.0)
knn = KNeighborsRegressor(n_neighbors=5)

poly_pipeline =Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(degree=2)),
    ("model", MultiOutputRegressor(LinearRegression()))
])


# ==========================================================
# 10 Weighted Voting
# ==========================================================

print("\nComputing model weights...")

cv_models = {
    "rf": RandomForestRegressor(random_state=42),
    "lgbm": LGBMRegressor(random_state=42),
    "ridge": ridge,
    "knn": knn
}

cv_scores = {}

for name, model in cv_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", MultiOutputRegressor(model))
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="r2")
    cv_scores[name] = np.mean(scores)

min_score = min(cv_scores.values())

weights = {k: (v - min_score + 1e-5) for k, v in cv_scores.items()}
total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}


voting = VotingRegressor(
    estimators=[
        ("rf", RandomForestRegressor(random_state=42)),
        ("lgbm", LGBMRegressor(random_state=42)),
        ("ridge", ridge),
        ("knn", knn)
    ],
    weights=[
        weights["rf"],
        weights["lgbm"],
        weights["ridge"],
        weights["knn"]
    ]
)


# ==========================================================
# 11 Model Training
# ==========================================================

models = {
    "RandomForest_Tuned": best_models["RandomForest"],
    "XGBoost_Tuned": best_models["XGBoost"],
    "LightGBM_Tuned": best_models["LightGBM"],
    "ExtraTrees": et,
    "Ridge": ridge,
    "KNN": knn,
    "Polynomial": poly_pipeline,
    "WeightedVoting": voting
}

results = {}
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    if isinstance(model, Pipeline):
        pipeline = model
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", MultiOutputRegressor(model))
        ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{name} R2: {r2:.4f}")
    print(f"{name} RMSE: {rmse:.2f}")

    results[name] = r2
    trained_models[name] = pipeline


# ==========================================================
# 12 Model Comparison Plot
# ==========================================================

results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "R2": list(results.values())
}).sort_values(by="R2")

plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="R2", y="Model")
plt.title("Model Comparison (R² Score)", weight="bold")
plt.tight_layout()
plt.savefig("model_leaderboard.png", dpi=400)
plt.show()


# ==========================================================
# 13 Save Best Model
# ==========================================================

best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

joblib.dump(best_model, "best_model.pkl")

print("Model saved as best_model.pkl")