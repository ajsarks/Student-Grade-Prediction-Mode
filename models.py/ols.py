import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

def run_ols(X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model = LinearRegression().fit(X_train, y_train)
    duration = time.perf_counter() - start

    preds = model.predict(X_test)
    return {
        "r2":    r2_score(y_test, preds),
        "mse":   mean_squared_error(y_test, preds),
        "time":  duration,
        "preds": preds,
        "y_test": y_test
    }

def plot_ols(y_test, preds, r2, mse):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, preds, alpha=0.7, s=70, label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    plt.title(f"OLS: R²={r2:.3f}, MSE={mse:.3f}")
    plt.xlabel("Actual Grade")
    plt.ylabel("Predicted Grade")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    from data_utils import load_data
    X_train, X_test, y_train, y_test = load_data()
    res = run_ols(X_train, X_test, y_train, y_test)
    print("OLS Results:", res)
    plot_ols(res["y_test"], res["preds"], res["r2"], res["mse"])

