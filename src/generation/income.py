"""
In this file, I sample income for Full Time & Part Time Employees.

This will entail fitting candidate distributions (lognormal, Gamma/Weibull, GB2) to each
age x occupation cell's known ONS percentile points via quantile matching, compares fit
quality using AIC/BIC, and then uses the winner out of the three to sample realistic 
continuous income values.

Data sources: src/archetypes/full_time.py & src/archetypes/part_time.py

Lognormal fitting:

Tested on 30-39 Professional (full-time): s=0.368, scale=49010, real median
was 48190 so ~1.7% off, fine.

Checked all 11 percentiles though and there's a pattern -> both tails
(p10, p90) undershoot by ~3%, p60/p70 overshoot by ~2-3%, p40 basically
bang on. Same tail curve behaviour I saw in the QQ plots earlier
(notebooks/01_income_distribution_exploration.py). So lognormal's good
but not perfect - need to actually check if gamma/GB2 do better via
AIC/BIC rather than just assuming lognormal's good enough.

"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
REFERENCE_DIR = os.path.join(REPO_ROOT, "data", "reference")

PERCENTILE_COLS = ['p10', 'p20', 'p25', 'p30', 'p40', 'median', 'p60', 'p70', 'p75', 'p80', 'p90']
PERCENTILE_PROBS = np.array([10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]) / 100


def lognormal_predict(u, s, scale):
    """
    given a probability (u) and lognormal params (s, scale), return the income value
    the lognormal distribution predicts at the probability
    """

    #scipy has built in lognorm.ppf allowing us to get income from probability
    return stats.lognorm.ppf(u, s, scale = scale)

def fit_lognormal_single_row(percentile_values: np.ndarray):
    """
    fitting lognorm distribution to ONE row's known percentile values via quantile matching 
    using PERCENTILE_PROBS as the known probabilities 
    """

    #starting guess -> s (set to 1.0 initially), scale = exp(mu) -> median of lognormal = exp(mu)
    #so scale = exp(median of percentile_values) 

    s = 1.0
    median_idx = PERCENTILE_COLS.index('median')
    scale = percentile_values[median_idx]

    p0 = [s, scale] #-> starting guess

    param, param_cov = curve_fit(lognormal_predict, PERCENTILE_PROBS, percentile_values, p0 = p0)

    return param


#quick sanity check to see if fitted scale is close to rows real median value before generalising to all 54 rows
if __name__ == "__main__":
    path = os.path.join(REFERENCE_DIR, "salary_lookup_age_occupation_fulltime_2025.csv")
    df = pd.read_csv(path)

    row = df[(df['age_band'] == '30-39') & (df['occupation'] == 'Professional occupations')]
    percentile_values = row[PERCENTILE_COLS].values.flatten()

    print("Real percentile values:", percentile_values)

    params = fit_lognormal_single_row(percentile_values)
    print("Fitted params [s, scale]:", params)

    
    predicted = lognormal_predict(PERCENTILE_PROBS, *params) #feeds wrapper function all 11 probabilities AT ONCE (rather than one at a time)
    comparison = pd.DataFrame({
        'percentile': PERCENTILE_COLS,
        'real': percentile_values,
        'predicted': predicted,
    })
    comparison['pct_error'] = (comparison['predicted'] - comparison['real']) / comparison['real'] * 100
    print(comparison)

     