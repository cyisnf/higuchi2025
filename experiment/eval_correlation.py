# %%
import numpy as np
import warnings
import pandas as pd
import models
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)

raw_data = pd.read_csv("experimental_data.csv")
# raw_data = raw_data.drop("est_i", axis=1)
raw_data = raw_data.drop("number", axis=1)
raw_data = raw_data.rename(columns={"frequency": "stim"})
mean_df = raw_data.groupby("stim")["estimation"].mean()
human = np.array(mean_df / 100)
stims = np.array(
    [
        [7, 1, 1, 1, 1, 7],
        [6, 1, 1, 7, 2, 1],
        [4, 1, 1, 7, 4, 1],
        [5, 2, 2, 2, 2, 5],
        [5, 4, 5, 1, 1, 2],
        [3, 3, 3, 3, 3, 3],
    ]
)

mean_prs_vals = []
max_prs_vals = []
min_prs_vals = []
term1_prs_vals = []
term2_prs_vals = []
weighted_mean_prs_vals_arr = [[] for i in range(11)]
merged_prs_vals = []

for stim in stims:
    stim = stim.reshape(3, 2)
    merged_prs_vals.append(models.paris_merge(stim))
    mean_prs_vals.append(models.paris_mean(stim))
    max_prs_vals.append(models.paris_max(stim))
    min_prs_vals.append(models.paris_min(stim))
    term1_prs_vals.append(models.paris_term1(stim))
    term2_prs_vals.append(models.paris_term2(stim))
    for w_i, w in enumerate(np.round(np.arange(0.0, 1.1, 0.1), 1)):
        weighted_mean_prs_vals_arr[w_i].append(
            models.paris_weighted_mean(stim, [w, 1 - w])
        )

df = pd.DataFrame(
    [
        human,
        mean_prs_vals,
        merged_prs_vals,
        max_prs_vals,
        min_prs_vals,
        term1_prs_vals,
        term2_prs_vals,
    ],
    index=["human", "mean", "merge", "max", "min", "term1", "term2"],
    columns=np.arange(1, 7),
).T

corr = df.corr(method=lambda x, y: pearsonr(x, y)[0])
pvalues = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(df.columns))
asterisk = pvalues.map(lambda x: "".join(["*" for t in [0.05, 0.01, 0.001] if x <= t]))

print("correlations:")
print(corr.iloc[0, 1:])
print("\npvalues:")
print(pvalues.iloc[0, 1:])
print("\nresults:")
print(corr.iloc[0, 1:].round(3).astype(str) + asterisk.iloc[0, 1:])
# %%

w_df = pd.DataFrame([human], index=["human"], columns=np.arange(1, 7))
for w_i, w in enumerate(np.round(np.arange(0.0, 1.1, 0.1), 1)):
    w_df.loc[str(w)] = weighted_mean_prs_vals_arr[w_i]
w_df = w_df.T

w_corr = w_df.corr(method=lambda x, y: pearsonr(x, y)[0])
w_pvalues = w_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(w_df.columns))
w_asterisk = w_pvalues.map(
    lambda x: "".join(["*" for t in [0.05, 0.01, 0.001] if x <= t])
)

print("correlations:")
print(w_corr.iloc[0, 1:])
print("\npvalues:")
print(w_pvalues.iloc[0, 1:])
print("\nresults:")
print(w_corr.iloc[0, 1:].round(3).astype(str) + w_asterisk.iloc[0, 1:])

plt.xlabel("weight of 1st term")
plt.ylabel("correlation with human")
w_corr.iloc[0, 1:].plot(grid=True, label="weighted_mean_prs", marker="o")
plt.show()
# %%
