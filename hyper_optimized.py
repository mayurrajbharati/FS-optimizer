# ==========================================
# Filesystem Performance Prediction Model
# ==========================================

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

from lightgbm import LGBMRegressor


# ==========================================
# 1 Load Dataset
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv("final_dataset.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2 Feature Engineering
# ==========================================

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


# ==========================================
# 3 Define Targets
# ==========================================

target_cols = ["mean_latency", "bandwidth", "iops"]

X = df.drop(columns=target_cols)

y = df[target_cols]


# ==========================================
# 4 Identify Feature Types
# ==========================================

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


# ==========================================
# 5 Preprocessing Pipeline
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


# ==========================================
# 6 Train/Test Split
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 7 Model Pipeline
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", MultiOutputRegressor(
        LGBMRegressor(random_state=42)
    ))
])


# ==========================================
# 8 Hyperparameter Search Space
# ==========================================

param_grid = {

    "model__estimator__n_estimators": [200, 300, 400, 500],

    "model__estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],

    "model__estimator__max_depth": [-1, 6, 8, 10],

    "model__estimator__num_leaves": [31, 50, 70],

    "model__estimator__subsample": [0.7, 0.8, 0.9],

    "model__estimator__colsample_bytree": [0.7, 0.8, 0.9]
}


# ==========================================
# 9 Hyperparameter Optimization
# ==========================================

print("\nRunning hyperparameter optimization...")

search = RandomizedSearchCV(

    pipeline,

    param_distributions=param_grid,

    n_iter=30,

    cv=5,

    scoring="r2",

    verbose=2,

    n_jobs=-1,

    random_state=42
)

search.fit(X_train, y_train)


# ==========================================
# 10 Best Model
# ==========================================

best_model = search.best_estimator_

print("\nBest Parameters Found:\n")

print(search.best_params_)


# ==========================================
# 11 Model Evaluation
# ==========================================

print("\nEvaluating model...")

preds = best_model.predict(X_test)

r2 = r2_score(y_test, preds)

rmse = np.sqrt(mean_squared_error(y_test, preds))


print("\nFinal Model Performance")

print("R2 Score:", r2)

print("RMSE:", rmse)


# ==========================================
# 12 Save Model
# ==========================================

joblib.dump(best_model, "filesystem_performance_model.pkl")

print("\nModel saved as filesystem_performance_model.pkl")


# ==========================================
# 13 Feature Importance
# ==========================================

try:

    model = best_model.named_steps["model"].estimators_[0]

    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False)

    print("\nTop Important Features:")

    print(importance_df.head(15))

except:

    print("\nFeature importance not available")