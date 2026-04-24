# ==========================================
# Adaptive Filesystem Configuration Optimizer
# ==========================================

import joblib
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1 Load trained ML model
# ==========================================

print("Loading trained model...")

model = joblib.load("best_filesystem_model.pkl")

print("Model loaded successfully")


# ==========================================
# 2 Parameter Search Space
# ==========================================

search_space = {

    "workload_type": ["randread", "randwrite", "randrw", "read", "write", "rw"],

    "block_size": [1024, 2048, 4096],

    "io_schedulers": ["bfq", "kyber", "none"],

    "read_ahead_sizes": [64, 128, 256],

    "barriers": ["on", "off"],

    "noatime_options": ["atime", "noatime"],

    "commit_intervals": [1, 5, 10],

    "io_engines": ["libaio", "sync"]
}


# ==========================================
# 3 Security Constraint (controls journaling)
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
# 4 Feature Engineering (same as training)
# ==========================================

def feature_engineering(df):

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

    return df


# ==========================================
# 5 Random Configuration Generator
# ==========================================

def generate_random_config():

    config = {}

    for param in search_space:
        config[param] = random.choice(search_space[param])

    return config


# ==========================================
# 6 Evaluate Configuration Using ML Model
# ==========================================

def evaluate_configuration(config):

    df = pd.DataFrame([config])

    df = feature_engineering(df)

    preds = model.predict(df)

    latency = preds[0][0]
    bandwidth = preds[0][1]
    iops = preds[0][2]

    # Optimization objective
    score = (bandwidth + iops) - latency

    return score, latency, bandwidth, iops


# ==========================================
# 7 Optimization Loop
# ==========================================

def optimize_filesystem(workload, security_level, iterations=1000):

    best_score = -np.inf
    best_config = None
    best_metrics = None

    for i in range(iterations):

        config = generate_random_config()

        config["workload_type"] = workload

        config = apply_security_constraint(config, security_level)

        score, latency, bandwidth, iops = evaluate_configuration(config)

        if score > best_score:

            best_score = score
            best_config = config
            best_metrics = (latency, bandwidth, iops)

    return best_config, best_metrics


# ==========================================
# 8 User Input
# ==========================================

print("\nAvailable workloads:")
print("randread, randwrite, randrw, read, write, rw")

valid_workloads = ["randread","randwrite","randrw","read","write","rw"]

while True:

    workload = input("\nEnter workload type: ").strip()

    if workload in valid_workloads:
        break
    else:
        print("Invalid workload. Try again.")


print("\nSecurity Levels:")
print("low    -> maximum performance")
print("medium -> balanced")
print("high   -> maximum safety")

valid_security = ["low","medium","high"]

while True:

    security_level = input("\nEnter security level: ").strip().lower()

    if security_level in valid_security:
        break
    else:
        print("Invalid security level.")


# ==========================================
# 9 Run Optimization
# ==========================================

print("\nRunning optimization...")

best_config, metrics = optimize_filesystem(
    workload,
    security_level,
    1200
)


# ==========================================
# 10 Display Results
# ==========================================

print("\n===================================")
print("Best Configuration Found")
print("===================================\n")

for k, v in best_config.items():
    print(f"{k}: {v}")


print("\nPredicted Performance")

print(f"Latency   : {metrics[0]:.2f}")
print(f"Bandwidth : {metrics[1]:.2f}")
print(f"IOPS      : {metrics[2]:.2f}")