# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import eval_estimation_performance, eval_calculability, plotter

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

READ_RESULT = True  # Whether to reuse generated CSVs
sim_size = 1000000
sample_sizes = [10, 100, 1000]
dataframes = []

for sample_size in sample_sizes:
    output_dir = os.path.join("csv", f"sample_size={sample_size}_sim_size={sim_size}")
    if READ_RESULT:
        df = pd.read_csv(os.path.join(output_dir, "estimation_performance.csv"))
        dataframes.append(df)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"sample_size: {sample_size}")
        df = eval_estimation_performance(sample_size, sim_size)
        df.to_csv(os.path.join(output_dir, "estimation_performance.csv"))
        dataframes.append(df)

# plot Fig. 4
prefix = "corr_"
for si, sample_size in enumerate(sample_sizes):
    plotter(prefix, dataframes[si], sample_size, sim_size)

# plot Fig.5
prefix = "prop_"
for sample_size in sample_sizes:
    def_df = eval_calculability(sample_size, sim_size)
    plotter(prefix, def_df, sample_size, sim_size)

# %%
