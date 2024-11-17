# %%
import numpy as np
import warnings
import pandas as pd
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

# Calculates the model values for each stimulus
for i, stim in enumerate(stims):
    stim = stim.reshape(3, 2)
    raw_data.loc[raw_data["stim_id"] == i + 1, "prs_mean"] = models.paris_mean(stim)
    raw_data.loc[raw_data["stim_id"] == i + 1, "prs_max"] = models.paris_max(stim)
    raw_data.loc[raw_data["stim_id"] == i + 1, "prs_min"] = models.paris_min(stim)
    raw_data.loc[raw_data["stim_id"] == i + 1, "prs_term1"] = models.paris_term1(stim)
    raw_data.loc[raw_data["stim_id"] == i + 1, "prs_term2"] = models.paris_term2(stim)
    raw_data.loc[raw_data["stim_id"] == i + 1, "prs_merge"] = models.paris_merge(stim)


# Calculates the descriptive performances using a linear mixed model that assumes a random slope and random intercept for each participant.
bics = {}
aiccs = {}
models = ["prs_mean", "prs_max", "prs_min", "prs_term1", "prs_term2", "prs_merge"]
for model in models:
    md = smf.mixedlm(
        f"response_value ~ {model}",
        raw_data,
        re_formula=f"~{model}",
        groups=raw_data["part_id"],
    )
    mdf = md.fit(reml=False, method="cg")
    print(mdf.summary())
    raw_data[model] = mdf.predict()
    bics[model] = mdf.bic
    aiccs[model] = eval_measures.aicc(mdf.llf, mdf.nobs, mdf.df_modelwc)


# Prints out the results.
bic_df = pd.Series(bics)
aiccs_df = pd.Series(aiccs)
print("BIC")
print(bic_df)
print("\nAICC")
print(aiccs_df)
