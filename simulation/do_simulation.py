# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import eval_estimation_performance, eval_calculability, plotter

# Standard out settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Figure settings
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.markerscale"] = 5

# Whether to reuse previously generated results.
READ_RESULT = True
# The number of contingency tables that will be sampled.
SIMULATION_SIZE = int(1e6)
# Number of events included in each contingency table.
SAMPLE_SIZES = [10, 100, 1000]


# Simulates, or loads previous results.
results = []
for sample_size in SAMPLE_SIZES:
    output_dir = os.path.join(
        "csv", f"sample_size={sample_size}_sim_size={SIMULATION_SIZE}"
    )
    if READ_RESULT:
        df = pd.read_csv(os.path.join(output_dir, "estimation_performance.csv"))
        results.append(df)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"sample_size: {sample_size}")
        df = eval_estimation_performance(sample_size, SIMULATION_SIZE)
        df.to_csv(os.path.join(output_dir, "estimation_performance.csv"))
        results.append(df)

# Plots Fig.4. (Correlation coefficient between model and population mutual information.)
prefix = "corr_"
for si, sample_size in enumerate(SAMPLE_SIZES):
    plotter(prefix, results[si], sample_size, SIMULATION_SIZE)

# Plots Fig.5. (For each model, the proportion of cases where they can be calculated.)
prefix = "prop_"
for sample_size in SAMPLE_SIZES:
    def_df = eval_calculability(sample_size, SIMULATION_SIZE)
    plotter(prefix, def_df, sample_size, SIMULATION_SIZE)


# %%
