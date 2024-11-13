# %%
import numpy as np
import warnings
import pandas as pd
import models
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)

# Loads experimental data
raw_data = pd.read_csv("experimental_data.csv")

# Calculates the mean response value for each stimulus among all participants.
human = np.array(raw_data.groupby("stim_id")["response_value"].mean() / 100)

# Stimuli used in our experiment
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

# Weights of the weighted mean
weights = np.round(np.arange(0.0, 1.1, 0.1), 1)

# Variables that stores the model values for each stimulus
prs_mean_vals = np.zeros(len(stims))
prs_max_vals = np.zeros(len(stims))
prs_min_vals = np.zeros(len(stims))
prs_term1_vals = np.zeros(len(stims))
prs_term2_vals = np.zeros(len(stims))
prs_weighted_mean_vals = np.zeros((len(weights), len(stims)))
prs_merged_vals = np.zeros(len(stims))

# Calculates the model values for each stimulus
for s_i, stim in enumerate(stims):
    stim = stim.reshape(3, 2)
    prs_merged_vals[s_i] = models.paris_merge(stim)
    prs_mean_vals[s_i] = models.paris_mean(stim)
    prs_max_vals[s_i] = models.paris_max(stim)
    prs_min_vals[s_i] = models.paris_min(stim)
    prs_term1_vals[s_i] = models.paris_term1(stim)
    prs_term2_vals[s_i] = models.paris_term2(stim)
    for i, w in enumerate(weights):
        prs_weighted_mean_vals[i, s_i] = models.paris_weighted_mean(stim, [w, 1 - w])

# Converts the calculation results to a data frame.
df = pd.DataFrame(
    [
        human,
        prs_mean_vals,
        prs_merged_vals,
        prs_max_vals,
        prs_min_vals,
        prs_term1_vals,
        prs_term2_vals,
    ],
    index=["human", "mean", "merge", "max", "min", "term1", "term2"],
).T

# Calculates the correlation coefficient between the mean response value and each model value.
correlations = df.corr(method=lambda x, y: pearsonr(x, y)[0])
pvalues = df.corr(method=lambda x, y: pearsonr(x, y)[1])

# Prints out the results.
print("correlations:\n", correlations.iloc[0, 1:])
print("\npvalues:\n", pvalues.iloc[0, 1:])

# In the same way as the above procedure, calculates the correlation coefficient for each weight of the weighted mean model.
# Converts the calculation results to a data frame.
df = pd.DataFrame(
    np.concatenate([[human], prs_weighted_mean_vals], axis=0).T,
    columns=np.concatenate([["human"], weights.astype("str")]),
)

# Calculates the correlation coefficient between the mean response value and each model value.
correlations = df.corr(method=lambda x, y: pearsonr(x, y)[0])
pvalues = df.corr(method=lambda x, y: pearsonr(x, y)[1])

# Prints out the results.
print("correlations:\n", correlations.iloc[0, 1:])
print("\npvalues:\n", pvalues.iloc[0, 1:])

# Plots the results
plt.xlabel("weight of 1st term")
plt.ylabel("correlation with human")
correlations.iloc[0, 1:].plot(grid=True, label="weighted_mean_prs", marker="o")
plt.show()
