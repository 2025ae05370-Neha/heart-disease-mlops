"""Generate the MLOps architecture diagram for the report.

Produces report/screenshots/architecture.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent / "screenshots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

COLORS = {
    "data": "#4C9BE8",
    "train": "#8E7CC3",
    "ci": "#E8A33D",
    "serve": "#5FA85F",
    "monitor": "#D9534F",
}


def box(x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5, edgecolor="#333333",
            facecolor=color, alpha=0.9, mutation_aspect=1,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, color="white", weight="bold", wrap=True)
    return (x + w / 2, y + h / 2)


def arrow(p1, p2):
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                        linewidth=1.3, color="#555555",
                        connectionstyle="arc3,rad=0.0")
    )


# Title
ax.text(6.5, 7.7, "Heart Disease Prediction — MLOps Architecture",
        ha="center", fontsize=15, weight="bold")

# Row 1: Data pipeline
d1 = box(0.4, 6.0, 2.3, 0.9, "UCI Heart Disease\n(4 hospital files)", COLORS["data"])
d2 = box(3.3, 6.0, 2.3, 0.9, "download_data.py\npreprocess.py", COLORS["data"])
d3 = box(6.2, 6.0, 2.3, 0.9, "heart.csv\n(cleaned dataset)", COLORS["data"])
d4 = box(9.1, 6.0, 3.3, 0.9, "EDA notebook\n(histograms, heatmap)", COLORS["data"])
arrow(d1, d2)
arrow(d2, d3)
arrow(d3, d4)

# Row 2: Training + tracking
t1 = box(3.3, 4.4, 2.3, 0.9, "train.py\nLogReg + RandomForest", COLORS["train"])
t2 = box(6.2, 4.4, 2.3, 0.9, "MLflow\n(experiment tracking)", COLORS["train"])
t3 = box(9.1, 4.4, 3.3, 0.9, "heart_model.joblib\n(Pipeline + model)", COLORS["train"])
arrow(d3, t1)
arrow(t1, t2)
arrow(t1, t3)

# Row 3: CI/CD
c1 = box(0.4, 2.8, 2.3, 0.9, "Developer\ngit push", COLORS["ci"])
c2 = box(3.3, 2.8, 5.2, 0.9,
         "GitHub Actions CI/CD\nlint -> test -> train -> docker build -> smoke test",
         COLORS["ci"])
arrow(c1, c2)
arrow((5.9, 4.4), (5.9, 3.7))  # train feeds CI concept

# Row 4: Serving + deployment
s1 = box(0.4, 1.2, 2.3, 0.9, "FastAPI\n/predict  /health", COLORS["serve"])
s2 = box(3.3, 1.2, 2.3, 0.9, "Docker image\n(heart-disease-api)", COLORS["serve"])
s3 = box(6.2, 1.2, 2.3, 0.9, "Kubernetes (Minikube)\n2 replicas + LoadBalancer", COLORS["serve"])
arrow(t3, s1)
arrow(s1, s2)
arrow(s2, s3)
arrow((5.9, 2.8), (4.45, 2.1))  # CI builds image

# Monitoring
m1 = box(9.1, 1.2, 3.3, 0.9, "Prometheus + Grafana\n(/metrics dashboard)", COLORS["monitor"])
arrow(s3, m1)

# Legend
handles = [
    mpatches.Patch(color=COLORS["data"], label="Data & EDA"),
    mpatches.Patch(color=COLORS["train"], label="Training & tracking"),
    mpatches.Patch(color=COLORS["ci"], label="CI/CD"),
    mpatches.Patch(color=COLORS["serve"], label="Serving & deployment"),
    mpatches.Patch(color=COLORS["monitor"], label="Monitoring"),
]
ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
          frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out = FIG_DIR / "architecture.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
