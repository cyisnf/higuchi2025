# %%
import numpy as np
import warnings
from functions import tab2data
import pandas as pd
import models
import statsmodels.formula.api as smf
from statsmodels.tools import eval_measures

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)

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

merged_prs_vals = []
max_prs_vals = []
min_prs_vals = []
term1_prs_vals = []
term2_prs_vals = []
mean_prs_vals = []
weighted_mean_prs_vals_arr = [[] for i in range(11)]

for stim in stims:
    stim = stim.reshape(3, 2)
    x, y = tab2data(stim)
    merged_prs_vals.append(models.paris_merge(stim))
    mean_prs_vals.append(models.paris_mean(stim))
    max_prs_vals.append(models.paris_max(stim))
    min_prs_vals.append(models.paris_min(stim))
    term1_prs_vals.append(models.paris_term1(stim))
    term2_prs_vals.append(models.paris_term2(stim))

bics = {}
aiccs = {}

raw_data = raw_data.assign(mean_prs=0)
raw_data = raw_data.assign(max_prs=0)
raw_data = raw_data.assign(min_prs=0)
raw_data = raw_data.assign(term1_prs=0)
raw_data = raw_data.assign(term2_prs=0)
raw_data = raw_data.assign(merge_prs=0)

for i in range(len(stims)):
    raw_data.loc[raw_data["stim"] == i + 1, "mean_prs"] = mean_prs_vals[i]
    raw_data.loc[raw_data["stim"] == i + 1, "max_prs"] = max_prs_vals[i]
    raw_data.loc[raw_data["stim"] == i + 1, "min_prs"] = min_prs_vals[i]
    raw_data.loc[raw_data["stim"] == i + 1, "term1_prs"] = term1_prs_vals[i]
    raw_data.loc[raw_data["stim"] == i + 1, "term2_prs"] = term2_prs_vals[i]
    raw_data.loc[raw_data["stim"] == i + 1, "merge_prs"] = merged_prs_vals[i]

md = smf.mixedlm(
    "estimation ~ mean_prs",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["fitted_mean_prs"] = mdf.predict()
bics["mean_prs"] = mdf.bic
aiccs["mean_prs"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "estimation ~ max_prs",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["fitted_max_prs"] = mdf.predict()
bics["max_prs"] = mdf.bic
aiccs["max_prs"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "estimation ~ min_prs",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["fitted_min_prs"] = mdf.predict()
bics["min_prs"] = mdf.bic
aiccs["min_prs"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "estimation ~ term1_prs",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["fitted_term1_prs"] = mdf.predict()
bics["term1_prs"] = mdf.bic
aiccs["term1_prs"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "estimation ~ term2_prs",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["fitted_term2_prs"] = mdf.predict()
bics["term2_prs"] = mdf.bic
aiccs["term2_prs"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "estimation ~ merge_prs",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["fitted_merge_prs"] = mdf.predict()
bics["merge_prs"] = mdf.bic
aiccs["merge_prs"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

# %%
bic_df = pd.Series(bics)
aiccs_df = pd.Series(aiccs)
print("BIC")
print(bic_df)
print("\nAICC")
print(aiccs_df)
# %%
