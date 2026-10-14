"""Method x dataset AUC matrix in the style of the SRBench 2025 paper
(postprocessing/notebooks/perfprof_plots.ipynb, figure 'r2_AUC_tuning_with_sizes').

Usage, from the srbench repo root (srbench-host environment):
    python plot_auc_matrix.py results/my1-top3/results.feather results/my1-top3/auc_matrix

Writes <prefix>.png, <prefix>.pdf and <prefix>.csv (the numbers behind each cell).

Each cell, as in the paper:
  number + colour  AUC of the performance profile of test R2 over the 30 runs
                   (same computation as the paper's perf_table: runs with
                   R2 < 0 count as failures; 0 = never good, 1 = always perfect)
  square size      model size of the run with the best test R2
                   (small <= 50 nodes, medium 51-99, large >= 100)
  black border     best AUC on that dataset (ties after rounding to 2 decimals)
"""
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import auc

# ---- settings you may want to change --------------------------------------
METHOD_ORDER = ["Brush", "GP-GOMEA", "GPZGD"]          # columns, left to right
DATASET_ORDER = ["my1_daily_o3", "my1_daily_pm25",      # rows, top to bottom
                 "my1_hourly_o3", "my1_hourly_pm25"]
CMAP = plt.get_cmap("coolwarm_r")                       # paper's colour map, AUC 0..1
# ---------------------------------------------------------------------------


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paper's perf_table: AUC of the R2 performance profile per (method, dataset)."""
    base = np.linspace(0, 1, 51)
    n_runs = df["random_state"].nunique()
    rows = []
    for (alg, ds), view in df.groupby(["algorithm", "dataset"]):
        view = view[view["r2_test"] >= 0].sort_values("r2_test")
        n_ok = view.shape[0]
        if n_ok == 0:
            rows.append((alg, ds, np.nan, 0))
            continue
        xs, ys = [0], [n_ok / n_runs]
        for k, count in Counter(view["r2_test"].values).items():
            if k == 0:
                n_ok -= count
                continue
            xs.append(k)
            ys.append(n_ok / n_runs)
            n_ok -= count
        rows.append((alg, ds, auc(base, np.interp(base, xs, ys)), view.shape[0]))
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "AUC", "runs_r2_ge_0"])


def best_run_size(df: pd.DataFrame) -> pd.DataFrame:
    """Paper: model size of the best-R2 run (ties -> smaller model); invalid sizes -> median."""
    d = df.dropna(subset=["r2_test", "model_size"]).copy()
    med = np.nanmedian(d.loc[d["model_size"] > 0, "model_size"])
    d.loc[d["model_size"] <= 0, "model_size"] = med
    d["r2_zero_test"] = d["r2_test"].clip(lower=0)
    best = (d.sort_values(["algorithm", "dataset", "r2_zero_test", "model_size"],
                          ascending=[True, True, False, True])
             .drop_duplicates(["algorithm", "dataset"]))
    return best[["algorithm", "dataset", "model_size", "r2_test"]].rename(
        columns={"model_size": "best_run_model_size", "r2_test": "best_run_r2_test"})


def square_side(size: float) -> float:
    """Paper's discrete mapping: >=100 large, >50 medium, else small."""
    rate = -0.05 if size >= 100 else (0.4 if size > 50 else 0.9)
    return 0.9 - rate * 0.3


def text_colour(rgba) -> str:
    r, g, b, _ = rgba
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


def main(results_path: str, prefix: str) -> None:
    df = pd.read_feather(results_path)
    df = df[df["algorithm"].isin(METHOD_ORDER) & df["dataset"].isin(DATASET_ORDER)]
    if df.empty:
        sys.exit(f"No rows for {METHOD_ORDER} x {DATASET_ORDER}; check the names with "
                 f"pd.read_feather('{results_path}')[['algorithm','dataset']].drop_duplicates()")

    table = auc_table(df).merge(best_run_size(df), on=["algorithm", "dataset"], how="outer")
    table["best_on_dataset"] = (table["AUC"].round(2)
                                == table.groupby("dataset")["AUC"].transform("max").round(2))
    table = table.sort_values(["dataset", "algorithm"])
    table.to_csv(f"{prefix}.csv", index=False)

    methods = [m for m in METHOD_ORDER if m in set(table["algorithm"])]
    datasets = [d for d in DATASET_ORDER if d in set(table["dataset"])]
    cell = table.set_index(["dataset", "algorithm"])

    fig, ax = plt.subplots(figsize=(1.1 * len(methods), 1.1 * len(datasets)))
    for i, ds in enumerate(datasets):
        for j, alg in enumerate(methods):
            ax.add_patch(Rectangle((j, i), 1, 1, fc="white", ec="gray", lw=0.5, ls=":"))
            if (ds, alg) not in cell.index or np.isnan(cell.loc[(ds, alg), "AUC"]):
                continue
            row = cell.loc[(ds, alg)]
            side = square_side(row["best_run_model_size"])
            colour = CMAP(float(np.clip(row["AUC"], 0, 1)))
            best = bool(row["best_on_dataset"])
            ax.add_patch(Rectangle((j + (1 - side) / 2, i + (1 - side) / 2), side, side,
                                   fc=colour, ec="black" if best else "none", lw=2 if best else 0))
            ax.text(j + 0.5, i + 0.5, f"{row['AUC']:.2f}", ha="center", va="center",
                    fontsize=10, color=text_colour(colour))

    ax.set_xlim(0, len(methods))
    ax.set_ylim(len(datasets), 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(methods)) + 0.5, methods, rotation=90)
    ax.set_yticks(np.arange(len(datasets)) + 0.5, datasets)
    ax.yaxis.tick_right()
    ax.tick_params(length=4)
    for side_name, spine in ax.spines.items():
        spine.set_visible(side_name in ("right", "bottom"))

    # colour bar (top right) and size/best legend (bottom right), as in the paper,
    # placed just past the dataset labels (measured, so long names still fit)
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    label_right = max(inv.transform(t.get_window_extent())[1][0] for t in ax.get_yticklabels())
    x0 = label_right + 0.08
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cax = ax.inset_axes([x0, 0.93, 0.85, 0.035])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("AUC Score")
    handles = [
        plt.Line2D([], [], marker="s", ms=8, ls="", mfc="white", mec="black", mew=2, label="Best AUC on dataset"),
        plt.Line2D([], [], marker="s", ms=7, ls="", color="gray", label="Small (size ≤ 50)"),
        plt.Line2D([], [], marker="s", ms=10, ls="", color="gray", label="Medium (size 51–99)"),
        plt.Line2D([], [], marker="s", ms=13, ls="", color="gray", label="Large (size ≥ 100)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(x0, 0.0), fontsize=9,
              frameon=True, borderaxespad=0)

    for ext in ("png", "pdf"):
        fig.savefig(f"{prefix}.{ext}", dpi=200, bbox_inches="tight")
    print(table.to_string(index=False))
    print(f"\nSaved {prefix}.png, {prefix}.pdf and {prefix}.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])