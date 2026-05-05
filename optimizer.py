# ==========================================
# Adaptive Filesystem Configuration Optimizer (Bayesian Version)
# ==========================================

import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args


# ==========================================
# 1 Load trained ML model
# ==========================================

print("Loading trained model...")
model = joblib.load("best_model.pkl")
print("Model loaded successfully")


# ==========================================
# 2 Bayesian Search Space
# ==========================================

search_space = [
    Categorical(["randread", "randwrite", "randrw", "read", "write", "rw"], name="workload_type"),
    Categorical([1024, 2048, 4096], name="block_size"),
    Categorical(["bfq", "kyber", "none"], name="io_schedulers"),
    Categorical([64, 128, 256], name="read_ahead_sizes"),
    Categorical(["on", "off"], name="barriers"),
    Categorical(["atime", "noatime"], name="noatime_options"),
    Categorical([1, 5, 10], name="commit_intervals"),
    Categorical(["libaio", "sync"], name="io_engines"),
]


# ==========================================
# 3 Security Constraint
# ==========================================

def apply_security_constraint(config, security_level):

    if security_level == "low":
        config["journal_modes"] = "writeback"
    elif security_level == "medium":
        config["journal_modes"] = "ordered"
    elif security_level == "high":
        config["journal_modes"] = "data"

    return config


# ==========================================
# 4 Feature Engineering
# ==========================================

def feature_engineering(df):

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

    return df


# ==========================================
# 5 Evaluate Configuration
# ==========================================

def evaluate_configuration(config):

    df = pd.DataFrame([config])
    df = feature_engineering(df)

    preds = model.predict(df)[0]

    latency = preds[0]
    bandwidth = preds[1]
    iops = preds[2]

    return latency, bandwidth, iops


# ==========================================
# 6 Pareto Dominance
# ==========================================

def dominates(a, b):
    return (
        (a[0] <= b[0] and a[1] >= b[1] and a[2] >= b[2]) and
        (a[0] < b[0] or a[1] > b[1] or a[2] > b[2])
    )


# ==========================================
# 7 Remove Duplicates
# ==========================================

def remove_duplicates(pareto):

    unique = []
    seen = set()

    for p in pareto:
        key = tuple(sorted(p["config"].items()))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ==========================================
# 8 Scoring Functions
# ==========================================

def performance_score(p):
    return p["bandwidth"] + p["iops"]

def latency_score(p):
    return -p["latency"]

def balanced_score(p):
    return (p["bandwidth"] * 0.4) + (p["iops"] * 0.4) - (p["latency"] * 0.2)


# ==========================================
# 9 User Input
# ==========================================

print("\nAvailable workloads:")
print("randread, randwrite, randrw, read, write, rw")

valid_workloads = ["randread","randwrite","randrw","read","write","rw"]

while True:
    workload = input("\nEnter workload type: ").strip()
    if workload in valid_workloads:
        break
    else:
        print("Invalid workload")

print("\nSecurity Levels:")
print("low    -> max performance")
print("medium -> balanced")
print("high   -> max safety")

valid_security = ["low","medium","high"]

while True:
    security_level = input("\nEnter security level: ").strip().lower()
    if security_level in valid_security:
        break
    else:
        print("Invalid security level.")

print("\nSelect Optimization Goal:")
print("1. Performance (Max Throughput)")
print("2. Low Latency")
print("3. Balanced")

while True:
    choice = input("\nEnter choice (1/2/3): ").strip()
    if choice in ["1", "2", "3"]:
        break
    else:
        print("Invalid choice")


# ==========================================
# 10 Bayesian Optimization Objective
# ==========================================

all_results = []

@use_named_args(search_space)
def objective(**params):

    # Override workload from user input
    params["workload_type"] = workload

    config = params.copy()
    config = apply_security_constraint(config, security_level)

    latency, bandwidth, iops = evaluate_configuration(config)

    all_results.append({
        "config": config,
        "latency": latency,
        "bandwidth": bandwidth,
        "iops": iops
    })

    # Convert multi-objective → scalar
    if choice == "1":
        return -(bandwidth + iops)

    elif choice == "2":
        return latency

    else:
        return -(0.4 * bandwidth + 0.4 * iops - 0.2 * latency)


# ==========================================
# 11 Run Bayesian Optimization
# ==========================================

print("\nRunning Bayesian Optimization...")

res = gp_minimize(
    objective,
    search_space,
    n_calls=50,
    n_initial_points=10,
    random_state=42
)

print(f"\nEvaluated Configurations: {len(all_results)}")


# ==========================================
# 12 Pareto Optimization
# ==========================================

pareto = []

for c in all_results:
    if not any(dominates(
        (o["latency"], o["bandwidth"], o["iops"]),
        (c["latency"], c["bandwidth"], c["iops"])
    ) for o in all_results):
        pareto.append(c)

pareto = remove_duplicates(pareto)

print(f"Total Pareto Solutions: {len(pareto)}")


# ==========================================
# 13 Select Best Based on Preference
# ==========================================

if choice == "1":
    best = max(pareto, key=performance_score)
    mode = "Performance"

elif choice == "2":
    best = max(pareto, key=latency_score)
    mode = "Low Latency"

else:
    best = max(pareto, key=balanced_score)
    mode = "Balanced"


# ==========================================
# 14 Display Result
# ==========================================

print("\n===================================")
print(f"Best Configuration ({mode} Mode)")
print("===================================\n")

for k, v in best["config"].items():
    print(f"{k}: {v}")

print("\nPredicted Performance:")
print(f"Latency   : {best['latency']:.2f}")
print(f"Bandwidth : {best['bandwidth']:.2f}")
print(f"IOPS      : {best['iops']:.2f}")