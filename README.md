# Replication codes for XXX

Published article link: doi:

## Abstract

How humans estimate causality is one of the central questions of cognitive science, and many studies aimed to model this estimation mechanism. Previous studies indicate the pARIs model is the most descriptive of human causality estimation among 42 normative and descriptive models defined on two binary variables. In this study, we amplify the previous study and attempt to elucidate a descriptive model of human causal induction in multi-valued variables. We extended pARIs to make it applicable to non-binary variables and verified the descriptive performance of the proposed model by conducting causal induction experiments on human subjects. We also conducted computer simulations to analyze the properties of the proposed model (or, indirectly, some tendencies in human causal induction). The experimental results showed that the correlation coefficient between the human response values and the proposed model values was r=.976, indicating that the proposed model is a valid descriptive model of human causal induction over multi-valued variables. On the other hand, the simulation results indicate that the proposed model is a good estimator of population mutual information when few observational data are available, especially when the rarity assumption applies in a generalized sense.

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

The following codes can be executed to replicate the results in the _Experiment_ section.

- eval_colleration.py (Table 5)
- eval_aicc.py (Table 5)
- eval_aicc_for_weighted.py (Figure 1)

## Simulation

The following code can be executed to replicate the results in the _Simulation_ section.
Note that the first run takes several hours.

- do_simulation.py (Figures 2 and 3)
