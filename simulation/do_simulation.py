# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions import eval_estimation_performance, eval_calculability, plotter

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Figure settings
plt.rcParams["font.family"] = "Arial"  # font familyの設定
plt.rcParams["font.size"] = 20  # 全体のフォントサイズが変更されま
plt.rcParams["xtick.direction"] = "out"  # x軸の目盛りの向き
plt.rcParams["ytick.direction"] = "out"  # y軸の目盛りの向き
plt.rcParams["xtick.minor.visible"] = False  # x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = False  # y軸補助目盛りの追加
plt.rcParams["xtick.top"] = False  # x軸の上部目盛り
plt.rcParams["ytick.right"] = False  # y軸の右部目盛り
plt.rcParams["legend.fancybox"] = False  # 丸角OFF
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.markerscale"] = 5  # markerサイズの倍率

row = 3
col = 2

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
