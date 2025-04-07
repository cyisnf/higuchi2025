# Replication Codes for: *A dependence detection heuristic in causal induction to handle nonbinary variables*

Published article link: <https://doi.org/10.1038/s41598-025-91051-7>

## Abstract

How humans estimate causality is one of the central questions of cognitive science, and many studies have attempted to model this estimation mechanism. Previous studies indicate that the pARIs model is the most descriptive of human causality estimation among 42 normative and descriptive models defined using two binary variables. In this study, we build on previous research and attempt to develop a new descriptive model of human causal induction with multi-valued variables. We extend the applicability of pARIs to non-binary variables and verify the descriptive performance of our model, pARIs_mean, by conducting a causal induction experiment involving human participants. We also conduct computer simulations to analyse the properties of our model (and, indirectly, some tendencies in human causal induction). The experimental results showed that the correlation coefficient between the human response values and the values of our model was r = .976, indicating that our model is a valid descriptive model of human causal induction with multi-valued variables. The simulation results also indicated that our model is a good estimator of population mutual information when only a small amount of observational data is available and when the probabilities of cause and effect are nearly equal and both probabilities are small.

## Requirements

- Python (Ver. 3.9.13)
- Packages listed in requirements.txt

## Usage

1. Install the relevant version of Python.
2. At the terminal, execute the following commands to set up the environment for running these programs.

    ```pip install -r requirements.txt```

3. You can reproduce our results by executing the following command at the terminal.

    ```python [file_name].py```

## Experimental Analysis

The following codes can be executed to replicate the results in the *Experiment* section.

- eval_colleration.py (Table 5)
- eval_aicc.py (Table 5)
- eval_aicc_for_weighted.py (Figure 1)

## Simulation

The following code can be executed to replicate the results in the *Simulation* section.
Note that the first run takes several hours.

- do_simulation.py (Figures 2 and 3)
