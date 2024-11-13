import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import font_manager
from scipy.interpolate import griddata
import models
import warnings

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)

ROW = 3
COL = 2


def val2ct(x_vals, y_vals):
    # Converts two variable values into a contingency table.
    ct = np.zeros([ROW, COL])
    for x_val, y_val in zip(x_vals, y_vals):
        ct[int(x_val), int(y_val)] += 1
    return ct


def sampling(sample_size, probs):
    # Samples two variables' values that follows the probability distribution of events.
    samples = random.choices(np.arange(0, ROW * COL, 1), k=sample_size, weights=probs)
    x_vals = []
    y_vals = []
    for i in samples:
        x_vals.append(i // COL)
        y_vals.append(i % COL)
    return x_vals, y_vals


def eval_estimation_performance(sample_size, sim_size):
    # Evaluates the correlation coefficient between model and population mutual information.

    # Makes output directory.
    output_dir = os.path.join("csv", f"sample_size={sample_size}_sim_size={sim_size}")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        index=[],
        columns=[
            "p_any_cause",  # P(C^prime)
            "p_any_effect",  # P(E^prime)
            "mi",  # Mutual-information
            "paris",  # pARIs
        ],
    )

    prob_space = np.round(np.arange(0.1, 1.0, 0.1), 1)
    for p_any_cause in tqdm(prob_space):
        for p_any_effect in tqdm(prob_space, disable=True):
            pop_vals = []
            sample_mis = []
            sample_paris = []

            for _ in range(sim_size):
                # Determine base probabilities.
                p_c10 = random.uniform(0, p_any_cause)  # P(C=1)
                p_c05 = p_any_cause - p_c10  # P(C=.5)
                p_c00 = 1 - p_any_cause  # P(C=0)
                p_e10 = p_any_effect  # P(E=1)
                if p_e10 - p_c00 > np.min([p_c10, p_e10]):
                    continue

                # Determine a population distribution.
                while True:
                    p_a_cell = random.uniform(
                        np.max([0, p_e10 - p_c00]), np.min([p_c10, p_e10])
                    )
                    if (p_e10 - p_c00 - p_a_cell) > np.min([p_c05, (p_e10 - p_a_cell)]):
                        continue
                    p_m_cell = random.uniform(
                        np.max([0, (p_e10 - p_c00 - p_a_cell)]),
                        np.min([p_c05, (p_e10 - p_a_cell)]),
                    )
                    if p_a_cell >= (p_e10 - p_c00 - p_m_cell) and p_a_cell <= (
                        p_e10 - p_m_cell
                    ):
                        break
                p_c_cell = p_e10 - (p_a_cell + p_m_cell)
                p_b_cell = p_c10 - p_a_cell
                p_n_cell = p_c05 - p_m_cell
                p_d_cell = p_c00 - p_c_cell
                pop_ct = [p_a_cell, p_b_cell, p_m_cell, p_n_cell, p_c_cell, p_d_cell]

                # Calculate population MI.
                pop_vals.append(
                    models.mutual_information(np.array(pop_ct).reshape((ROW, COL)))
                )

                # Sample a contingency table
                x, y = sampling(sample_size, pop_ct)
                sampled_ct = val2ct(x, y)

                # Calculate model's values.
                sample_mis.append(models.mutual_information(sampled_ct))
                sample_paris.append(models.paris_mean(sampled_ct))

            # Save the results as csv.
            model_values = pd.DataFrame(
                data={
                    "pop": pop_vals,
                    "mi": sample_mis,
                    "paris": sample_paris,
                }
            )
            model_values.to_csv(
                os.path.join(
                    output_dir,
                    f"p_any_cause={p_any_cause}_p_any_effect={p_any_effect}.csv",
                )
            )

            # Calculate the correlations.
            corr_tab = model_values.corr(method="spearman")
            mi_corr = corr_tab.iloc[0, 1]
            paris_corr = corr_tab.iloc[0, 2]

            # Stores the results.
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            [
                                p_any_cause,
                                p_any_effect,
                                mi_corr,
                                paris_corr,
                            ]
                        ],
                        columns=df.columns,
                    ),
                ],
                ignore_index=True,
                axis=0,
            )

    return df.astype("float")


def eval_calculability(sample_size, sim_size):
    # For each model, evaluates the proportion of cases where they can be calculated.

    # Makes output directory.
    output_dir = os.path.join("csv", f"sample_size={sample_size}_sim_size={sim_size}")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        index=[],
        columns=[
            "p_any_cause",
            "p_any_effect",
            "mi",
            "paris",
        ],
    )

    prob_space = np.round(np.arange(0.1, 1.0, 0.1), 1)
    for p_any_cause in prob_space:
        for p_any_effect in prob_space:
            # Load csv files.
            val_df = pd.read_csv(
                os.path.join(
                    output_dir,
                    f"p_any_cause={p_any_cause}_p_any_effect={p_any_effect}.csv",
                )
            )

            # Evaluate the proportions for which the models are calculable.
            mi_def = 1 - len(val_df[val_df["mi"].isna()]) / len(val_df["mi"])
            paris_def = 1 - len(val_df[val_df["paris"].isna()]) / len(val_df["paris"])

            # Stores the results.
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            [
                                p_any_cause,
                                p_any_effect,
                                mi_def,
                                paris_def,
                            ]
                        ],
                        columns=df.columns,
                    ),
                ],
                ignore_index=True,
                axis=0,
            )
    return df.astype("float")


def plotter(prefix, df, sample_size, sim_size):
    # Makes a figure.
    fig = plt.figure(figsize=(18, 5))

    # Adds the axis of pARIs_mean.
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title(r"$\it{pARIs_{mean}}$", size=20)
    ax1.set_xticks(np.arange(0.2, 1.0, 0.2))
    ax1.set_yticks(np.arange(0.2, 1.0, 0.2))
    x = df["p_any_cause"]
    y = df["p_any_effect"]
    x_new, y_new = np.meshgrid(np.unique(x), np.unique(y))
    z = df["paris"]
    z_new = griddata((x, y), z, (x_new, y_new))
    c = ax1.pcolormesh(x_new, y_new, z_new, cmap="bwr")
    c.set_clim(-1, 1)
    for (j, i), label in np.ndenumerate(z_new):
        ax1.text(
            i * 0.1 + 0.1,
            j * 0.1 + 0.1,
            np.round(label, 2),
            ha="center",
            va="center",
            fontsize=12,
        )
    cbar = fig.colorbar(c, ax=ax1, label=r"Proportion of calculable cases")
    font = font_manager.FontProperties(size=20)
    cbar.ax.yaxis.label.set_font_properties(font)

    # Adds the axis of MI.
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title(r"$\it{MI}$", size=20)
    ax2.set_xticks(np.arange(0.2, 1.0, 0.2))
    ax2.set_yticks(np.arange(0.2, 1.0, 0.2))
    x = df["p_any_cause"]
    y = df["p_any_effect"]
    x_new, y_new = np.meshgrid(np.unique(x), np.unique(y))
    z = df["mi"]
    z_new = griddata((x, y), z, (x_new, y_new))
    c = ax2.pcolormesh(x_new, y_new, z_new, cmap="bwr")
    c.set_clim(-1, 1)
    for (j, i), label in np.ndenumerate(z_new):
        ax2.text(
            i * 0.1 + 0.1,
            j * 0.1 + 0.1,
            np.round(label, 2),
            ha="center",
            va="center",
            fontsize=12,
        )
    cbar = fig.colorbar(c, ax=ax2, label=r"Proportion of calculable cases")
    font = font_manager.FontProperties(size=20)
    cbar.ax.yaxis.label.set_font_properties(font)

    # Adds the axis of the difference.
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title(r"$\it{pARIs_{mean}}-\it{MI}$", size=20)
    ax3.set_xticks(np.arange(0.2, 1.0, 0.2))
    ax3.set_yticks(np.arange(0.2, 1.0, 0.2))
    x = df["p_any_cause"]
    y = df["p_any_effect"]
    x_new, y_new = np.meshgrid(np.unique(x), np.unique(y))
    z = df["paris"] - df["mi"]
    z_new = griddata((x, y), z, (x_new, y_new))
    c = ax3.pcolormesh(x_new, y_new, z_new, cmap="bwr")
    c.set_clim(-1, 1)
    cbar = fig.colorbar(c, ax=ax3, label=r"Difference in proportions")
    for (j, i), label in np.ndenumerate(z_new):
        ax3.text(
            i * 0.1 + 0.1,
            j * 0.1 + 0.1,
            np.round(label, 2),
            ha="center",
            va="center",
            fontsize=12,
        )
    font = font_manager.FontProperties(size=20)
    cbar.ax.yaxis.label.set_font_properties(font)

    # Adds the axis labels.
    fig.text(0.5, 0, r"$P(C=c^\prime)$", ha="center", va="center", fontsize=20)
    fig.text(
        0,
        0.5,
        r"$P(E=e^\prime)$",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=20,
    )

    # Plots the figure.
    plt.tight_layout()
    os.makedirs("./figs", exist_ok=True)
    plt.savefig(
        f"./figs/{prefix}sample_size={sample_size}_sims={sim_size}.png",
        bbox_inches="tight",
    )
