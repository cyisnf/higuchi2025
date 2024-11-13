# %%
import numpy as np
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import models
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

# Loads experimental data
raw_data = pd.read_csv("experimental_data.csv")
raw_data["response_value"] /= 100

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

# Calculates the weighted mean model value for all combinations of weight and stimulus.
prs_weighted_mean_vals = np.zeros((len(weights), len(stims)))
for s_i, stim in enumerate(stims):
    stim = stim.reshape(3, 2)
    for w_i, w in enumerate(weights):
        raw_data.loc[raw_data["stim_id"] == s_i + 1, f"w{w_i}"] = (
            models.paris_weighted_mean(stim, [w, 1 - w])
        )

# Calculates the descriptive performances using a linear mixed model that assumes a random slope and random intercept for each participant.
bics = {}
aiccs = {}
for w_i, w in enumerate(weights):
    model = f"w{w_i}"
    md = smf.mixedlm(
        f"response_value ~ {model}",
        raw_data,
        re_formula=f"~{model}",
        groups=raw_data["part_id"],
    )
    mdf = md.fit(reml=False)
    print(mdf.summary())
    bics[f"{model}"] = mdf.bic
    aiccs[f"{model}"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)


# Plots the results
xticks = np.round(np.arange(0.0, 1.1, 0.1), 1)
df = pd.DataFrame(
    np.array([list(bics.values()), list(aiccs.values())]).T,
    columns=["BIC", "AICc"],
    index=xticks,
)
df.plot(marker="o", figsize=(5, 2.3))
plt.xlabel(r"Weight of $\mathrm{pARIs}_{C=1, E=1}$")
plt.ylabel("Fitness with data")
plt.xticks(xticks)
plt.grid()
plt.legend()
plt.tight_layout()
os.makedirs("./figs", exist_ok=True)
plt.savefig("./figs/fitness_of_weighted.png")
