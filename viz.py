from graphviz import Digraph

# Create graph
g = Digraph("filesystem_ml_pipeline", format="png")

# Global graph settings
g.attr(rankdir="TB", dpi="300", bgcolor="white")

# Node styling
g.attr(
    "node",
    shape="ellipse",
    style="filled",
    fillcolor="white",
    color="black",
    penwidth="2",
    fontname="Helvetica",
    fontsize="14",
    fontcolor="black"
)

# Edge styling
g.attr(
    "edge",
    color="black",
    penwidth="2"
)

# -------------------------
# Nodes
# -------------------------

g.node("A", "Filesystem Benchmark Experiments")
g.node("B", "Dataset Collection (~10k records)")
g.node("C", "Data Preprocessing\n(Cleaning + Encoding)")
g.node("D", "Feature Engineering\nread_ahead_ratio\ncommit_density\nworkload_pattern\nscheduler_type")

g.node("E", "ML Model Training")

g.node("F1", "Random Forest")
g.node("F2", "Extra Trees")
g.node("F3", "XGBoost")
g.node("F4", "LightGBM")

g.node("G", "Cross Validation (k-fold)")
g.node("H", "Model Evaluation (R², RMSE)")
g.node("I", "Model Comparison")
g.node("J", "Best Model Selection\n(LightGBM)")

# Security layer
g.node("K", "User Input\nSecurity Level")
g.node("L", "Configuration Constraint Layer\n(Restrict journaling modes)")

# Optimization + Output
g.node("M", "Bayesian Optimization\n(Search optimal configs)")
g.node("N", "Final Filesystem Configuration\nRecommendation")

# -----------------------------
# Edges
# -----------------------------

g.edge("A", "B")
g.edge("B", "C")
g.edge("C", "D")
g.edge("D", "E")

# Model branches
g.edge("E", "F1")
g.edge("E", "F2")
g.edge("E", "F3")
g.edge("E", "F4")

# Merge after models
g.edge("F1", "G")
g.edge("F2", "G")
g.edge("F3", "G")
g.edge("F4", "G")

g.edge("G", "H")
g.edge("H", "I")
g.edge("I", "J")

# Security constraint flow
g.edge("J", "L")
g.edge("K", "L")

# Optimization
g.edge("L", "M")
g.edge("M", "N")

# Generate diagram
g.render("filesystem_ml_architecture_final", view=True)