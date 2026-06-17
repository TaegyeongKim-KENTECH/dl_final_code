import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

csv_path  = "/home/work/ktg0829/final_project/clipforfakedetection/weights/model_save/sweep_results3.csv"
save_dir  = "/home/work/ktg0829/final_project/clipforfakedetection"

df = pd.read_csv(csv_path)
x  = df["conf_threshold"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confidence Threshold Sweep", fontsize=14, fontweight='bold')

# ── 그래프 1: Accuracy ───────────────────────────────────────────────────
axes[0].plot(x, df["accuracy"], marker='o', color='steelblue')
axes[0].set_title("Accuracy")
axes[0].set_xlabel("conf_threshold")
axes[0].set_ylabel("Accuracy")
axes[0].set_xticks(x)
axes[0].grid(True, linestyle='--', alpha=0.5)

# ── 그래프 2: Average Precision ──────────────────────────────────────────
axes[1].plot(x, df["average_precision"], marker='o', color='darkorange')
axes[1].set_title("Average Precision")
axes[1].set_xlabel("conf_threshold")
axes[1].set_ylabel("AP")
axes[1].set_xticks(x)
axes[1].grid(True, linestyle='--', alpha=0.5)

# ── 그래프 3: Semantic Branch Called (%) ─────────────────────────────────
axes[2].plot(x, df["semantic_called"] * 100, marker='o', color='gray')
axes[2].set_title("Semantic Branch Called")
axes[2].set_xlabel("conf_threshold")
axes[2].set_ylabel("Semantic called (%)")
axes[2].set_xticks(x)
axes[2].set_ylim(0, 105)
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = f"{save_dir}/sweep_plot3.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")