import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import load_data
from ols import run_ols
from regression import run_gd
from nn import run_nn

plt.rcParams.update({'font.size': 12})
charts_dir = Path(__file__).resolve().parents[1] / "charts"
os.makedirs(charts_dir, exist_ok=True)

NUM_RUNS = 5
learning_rates = [0.001, 0.01, 0.1, 0.5, 0.75, 1]
best_lr = 0.01
tolerance = 0.05

def average_runs(model_func, X_train, X_test, y_train, y_test, lr, num_runs=NUM_RUNS):
    runs = [model_func(X_train, X_test, y_train, y_test, lr) for _ in range(num_runs)]
    return {
        "lr": lr,
        "time": np.mean([r["time"] for r in runs]),
        "r2": np.mean([r["r2"] for r in runs]),
        "mse": np.mean([r["mse"] for r in runs]),
        "loss": np.mean([r["loss"] for r in runs], axis=0),
        "preds": np.mean([r["preds"] for r in runs], axis=0)
    }

# Load data once and pass to all models
X_train, X_test, y_train, y_test = load_data()

# Run OLS
ols_results = run_ols(X_train, X_test, y_train, y_test)
y_pred_ols = ols_results["preds"]

# Run GD and NN with different learning rates
gd_results = [average_runs(run_gd, X_train, X_test, y_train, y_test, lr) for lr in learning_rates]
nn_results = [average_runs(run_nn, X_train, X_test, y_train, y_test, lr) for lr in learning_rates]

gd_best = next(r for r in gd_results if r["lr"] == best_lr)
nn_best = next(r for r in nn_results if r["lr"] == best_lr)

plt.figure(figsize=(10, 6))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Perfect Prediction')

plt.scatter(y_test, y_pred_ols, alpha=0.7, color='blue', s=70,
            label=f'OLS\n(R²={ols_results["r2"]:.3f}, MSE={ols_results["mse"]:.3f})')

overlap_gd = np.isclose(y_pred_ols, gd_best["preds"], atol=tolerance)
plt.scatter(y_test[~overlap_gd], gd_best["preds"][~overlap_gd], alpha=0.7, color='orange', s=70,
            label=f'GD (LR={best_lr})\n(R²={gd_best["r2"]:.3f}, MSE={gd_best["mse"]:.3f})')
plt.scatter(y_test[overlap_gd], gd_best["preds"][overlap_gd], facecolors='none', edgecolors='orange', s=100,
            label=f'Overlap with OLS ({np.sum(overlap_gd)} pts)')

overlap_nn = np.isclose(y_pred_ols, nn_best["preds"], atol=tolerance)
plt.scatter(y_test[~overlap_nn], nn_best["preds"][~overlap_nn], alpha=0.7, color='green', s=70,
            label=f'NN (LR={best_lr})\n(R²={nn_best["r2"]:.3f}, MSE={nn_best["mse"]:.3f})')
plt.scatter(y_test[overlap_nn], nn_best["preds"][overlap_nn], facecolors='none', edgecolors='green', s=100,
            label=f'Overlap with OLS ({np.sum(overlap_nn)} pts)')

plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Model Accuracy Comparison at Learning Rate 0.01")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "accuracy_comparison_with_overlap.png"))
plt.close()

plt.figure(figsize=(10, 6))
for res in gd_results:
    plt.plot(res["loss"], label=f"GD LR={res['lr']}")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Training Loss Across Learning Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "gd_loss_curves.png"))
plt.close()

plt.figure(figsize=(10, 6))
for res in nn_results:
    plt.plot(res["loss"], label=f"NN LR={res['lr']}")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Neural Network Training Loss Across Learning Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "nn_loss_curves.png"))
plt.close()

plt.figure(figsize=(10, 6))
for res in gd_results:
    plt.plot(res["loss"], linestyle='-', alpha=0.8, label=f"GD LR={res['lr']}")
for res in nn_results:
    plt.plot(res["loss"], linestyle='--', alpha=0.7, label=f"NN LR={res['lr']}")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("GD vs NN Training Loss Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "combined_loss_curves.png"))
plt.close()

all_times = [ols_results["time"]] + [r["time"] for r in gd_results] + [r["time"] for r in nn_results]
labels = ["OLS"] + [f"GD LR={r['lr']}" for r in gd_results] + [f"NN LR={r['lr']}" for r in nn_results]
colors = ['skyblue'] * (1 + len(gd_results)) + ['lightgreen'] * len(nn_results)

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, all_times, color=colors)
plt.ylabel("Time (seconds)")
plt.title("Average Compute Time per Model (5 Runs)")

plt.ylim(0, max(all_times) * 1.15)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (0.01 * max(all_times)),
        f"{height:.6f}s",
        ha='center',
        va='bottom',
        rotation=90
    )

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "compute_time_comparison.png"))
plt.close()

gd_lr = [r['lr'] for r in gd_results]
gd_r2 = [r['r2'] for r in gd_results]
nn_r2 = [r['r2'] for r in nn_results]
gd_mse = [r['mse'] for r in gd_results]
nn_mse = [r['mse'] for r in nn_results]

plt.figure(figsize=(10, 6))
plt.plot(gd_lr, gd_r2, marker='o', label="GD R²", color='orange')
plt.plot(gd_lr, nn_r2, marker='s', linestyle='--', label="NN R²", color='green')
plt.xlabel("Learning Rate")
plt.ylabel("R² Score")
plt.title("R² vs Learning Rate for GD and NN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "r2_vs_lr.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(gd_lr, gd_mse, marker='o', label="GD MSE", color='orange')
plt.plot(gd_lr, nn_mse, marker='s', linestyle='--', label="NN MSE", color='green')
plt.xlabel("Learning Rate")
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error vs Learning Rate for GD and NN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "mse_vs_lr.png"))
plt.close()

summary = [
    {
        "Model": "OLS",
        "Learning Rate": "N/A",
        "R²": round(ols_results["r2"], 4),
        "MSE": round(ols_results["mse"], 4),
        "Compute Time (s)": round(ols_results["time"], 6)
    }
]
summary += [
    {
        "Model": "GD",
        "Learning Rate": r["lr"],
        "R²": round(r["r2"], 4),
        "MSE": round(r["mse"], 4),
        "Compute Time (s)": round(r["time"], 6)
    } 
    for r in gd_results
]
summary += [
    {
        "Model": "NN",
        "Learning Rate": r["lr"],
        "R²": round(r["r2"], 4),
        "MSE": round(r["mse"], 4),
        "Compute Time (s)": round(r["time"], 6)
    } 
    for r in nn_results
]
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(charts_dir, "results_summary.csv"), index=False)

print("Final model results saved to:", os.path.join(charts_dir, "results_summary.csv"))
print(summary_df)

