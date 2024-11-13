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

# 実験データの読み込み
raw_data = pd.read_csv("experimental_data.csv")
raw_data["response_value"] /= 100

# 実験に用いた刺激セット
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

# 各刺激に対するモデル値を計算する
prs_merged_vals = []
prs_max_vals = []
prs_min_vals = []
prs_term1_vals = []
prs_term2_vals = []
prs_mean_vals = []
weighted_mean_prs_vals_arr = [[] for i in range(11)]

for stim in stims:
    stim = stim.reshape(3, 2)
    x, y = tab2data(stim)
    prs_merged_vals.append(models.paris_merge(stim))
    prs_mean_vals.append(models.paris_mean(stim))
    prs_max_vals.append(models.paris_max(stim))
    prs_min_vals.append(models.paris_min(stim))
    prs_term1_vals.append(models.paris_term1(stim))
    prs_term2_vals.append(models.paris_term2(stim))

# 計算したモデル値を
raw_data = raw_data.assign(prs_mean=0)
raw_data = raw_data.assign(prs_max=0)
raw_data = raw_data.assign(prs_min=0)
raw_data = raw_data.assign(prs_term1=0)
raw_data = raw_data.assign(prs_term2=0)
raw_data = raw_data.assign(prs_merge=0)

for i in range(len(stims)):
    raw_data.loc[raw_data["stimulation_id"] == i + 1, "prs_mean"] = prs_mean_vals[i]
    raw_data.loc[raw_data["stimulation_id"] == i + 1, "prs_max"] = prs_max_vals[i]
    raw_data.loc[raw_data["stimulation_id"] == i + 1, "prs_min"] = prs_min_vals[i]
    raw_data.loc[raw_data["stimulation_id"] == i + 1, "prs_term1"] = prs_term1_vals[i]
    raw_data.loc[raw_data["stimulation_id"] == i + 1, "prs_term2"] = prs_term2_vals[i]
    raw_data.loc[raw_data["stimulation_id"] == i + 1, "prs_merge"] = prs_merged_vals[i]

# %%

# 結果を格納するための変数
bics = {}
aiccs = {}

# pARIs_mean
md = smf.mixedlm(
    "response_value ~ prs_mean",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["prs_mean"] = mdf.predict()
bics["prs_mean"] = mdf.bic
aiccs["prs_mean"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "response_value ~ prs_max",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["prs_max"] = mdf.predict()
bics["prs_max"] = mdf.bic
aiccs["prs_max"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "response_value ~ prs_min",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["prs_min"] = mdf.predict()
bics["prs_min"] = mdf.bic
aiccs["prs_min"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "response_value ~ prs_term1",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["prs_term1"] = mdf.predict()
bics["prs_term1"] = mdf.bic
aiccs["prs_term1"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "response_value ~ prs_term2",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["prs_term2"] = mdf.predict()
bics["prs_term2"] = mdf.bic
aiccs["prs_term2"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

md = smf.mixedlm(
    "response_value ~ prs_merge",
    raw_data,
    groups=raw_data["user_id"],
)
mdf = md.fit(reml=False, method="cg")
print(mdf.summary())
raw_data["prs_merge"] = mdf.predict()
bics["prs_merge"] = mdf.bic
aiccs["prs_merge"] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)

# %%
bic_df = pd.Series(bics)
aiccs_df = pd.Series(aiccs)
print("BIC")
print(bic_df)
print("\nAICC")
print(aiccs_df)
# %%
