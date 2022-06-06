import json
import os

import numpy as np
import pandas as pd
from conditional_inference.bayes import Nonparametric

EXTERNAL_DATA_DIR = "data/external"
SIMULATION_DATA_DIR = "simulation_data"

if __name__ == "__main__":
    df = pd.read_stata(os.path.join(EXTERNAL_DATA_DIR, "NudgeUnits.dta"))

    # store control takeup distribution
    controltakeup = (
        df.controltakeup[(20 < df.controltakeup) & (df.controltakeup < 80)] / 100
    )
    with open(os.path.join(SIMULATION_DATA_DIR, "control.json"), "w") as f:
        json.dump(list(controltakeup), f)

    # store the prior distribution of nudge effects
    quantiles = df.treatmenteffect.quantile([0.05, 0.95])
    df = df[
        (quantiles.iloc[0] < df.treatmenteffect)
        & (df.treatmenteffect < quantiles.iloc[1])
    ]
    dist = Nonparametric(df.treatmenteffect, np.diag(df.SE**2)).get_marginal_prior(0)
    with open(os.path.join(SIMULATION_DATA_DIR, "prior.json"), "w") as f:
        json.dump((list(dist.xk / 100), list(dist.pk)), f)
