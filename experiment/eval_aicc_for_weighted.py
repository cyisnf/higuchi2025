# %%
import numpy as np
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import models
import importlib
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)

importlib.reload(models)


raw_data = pd.read_csv("experimental_data.csv")
raw_data = raw_data.drop("number", axis=1)
raw_data = raw_data.rename(columns={"frequency": "stim"})
raw_data["estimation"] /= 100
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

# %%
weighted_mean_prs_vals_arr = [[] for i in range(11)]
for stim in stims:
    stim = stim.reshape(3, 2)
    for w_i, w in enumerate(np.round(np.arange(0.0, 1.1, 0.1), 1)):
        weighted_mean_prs_vals_arr[w_i].append(
            models.paris_weighted_mean(stim, [w, 1 - w])
        )

bics = {}
aiccs = {}

for i in range(len(stims)):
    for w_i, w in enumerate(np.round(np.arange(0.0, 1.1, 0.1), 1)):
        idx = "w" + str(int(w * 10))
        raw_data.loc[raw_data["stim"] == i + 1, idx] = weighted_mean_prs_vals_arr[w_i][
            i
        ]


for i in range(len(stims)):
    for w_i, w in enumerate(np.round(np.arange(0.0, 1.1, 0.1), 1)):
        idx = "w" + str(int(w * 10))
        md = smf.mixedlm(
            f"estimation ~ {idx}",
            raw_data,
            re_formula=f"~{idx}",
            groups=raw_data["user_id"],
        )
        mdf = md.fit(reml=False)
        print(mdf.summary())
        raw_data[f"fitted_{w*10}"] = mdf.predict()
        bics[f"{idx}"] = mdf.bic
        aiccs[f"{idx}"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

# %%
index = [round(0.1 * i, 1) for i in range(11)]
arr = np.array([list(bics.values()), list(aiccs.values())])
df = pd.DataFrame(arr.T, columns=["BIC", "AICc"], index=index)
df.plot(marker="o", figsize=(5, 2.3))
plt.xlabel("Weight of $\mathrm{pARIs}_{C=1, E=1}$")
plt.ylabel("Fitness with data")
plt.xticks(index)
plt.grid()
plt.legend()
plt.tight_layout()
os.makedirs('./figs', exist_ok=True)
plt.savefig("./figs/fitness_of_weighted.png")
# %%
