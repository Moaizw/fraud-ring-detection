"""
In this file, I sample income for Full Time & Part Time Employees.

This will entail fitting candidate distributions (lognormal, Gamma/Weibull, GB2) to each
age x occupation cell's known ONS percentile points via quantile matching, compares fit
quality using AIC/BIC, and then uses the winner out of the three to sample realistic 
continuous income values.

Data sources: src/archetypes/full_time.py & src/archetypes/part_time.py
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


def _lognormal_predict(u, s, scale):
    """
    given a probability (u) and lognormal params (s, scale), return the income value
    the lognormal distribution predicts at the probability
    """

    #scipy has built in lognorm.ppf allowing us to get income from probability
    return stats.lognorm.ppf(u, s, scale = scale)

def fit_lognormal_single_row(percentile_values: np.ndarray, percentile_probs = None):
    """
    fitting lognorm distribution to ONE row's known percentile values via quantile matching 
    using PERCENTILE_PROBS as the known probabilities 
    """

    if percentile_probs is None:
        percentile_probs = PERCENTILE_PROBS 

    #starting guess -> s set to 1.0 (reasonable default), scale = exp(mu)
    s = 1.0

    #since median of lognormal = exp(mu), scale directly equals the real median
    #(no need to take exp() ourselves - median val already IS exp(mu))

    #-> find median DYNAMICALLY since percentile_vals may be shorter
    median_idx = np.where(np.isclose(percentile_probs, 0.5))[0] #np.where returns tuple so extract first element from tuple to get idx of median

    if len(median_idx) > 0:
        scale = percentile_values[median_idx[0]]
    else:
        #median itself missing for this row -> fall back to the middle
        #available value as a reasonable starting guess
        scale = percentile_values[len(percentile_values) // 2]

    p0 = [s, scale] #-> starting guess

    param, param_cov = curve_fit(_lognormal_predict, percentile_probs, percentile_values, p0 = p0)

    return param #fit lognormal, returns best [s, scale]


def fit_lognormal_all_rows(salary_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fitting lognormal to every (age_band, occupation) pair in salary_lookup via 
    quantile matching.

    N.B. Some rows have missing quantiles, HOWEVER, curve_fit can still
    match these quantiles albeit with somewhat less precision. n_points < 11 
    means some were missing for that row (will be noted). Rows with fewer than 2 real points 
    are skipped entirely (can't fit 2 parameters to fewer than 2 points) 
    and reported.
    """

    res = {'age_band':[], 'occupation':[], 'soc_code':[], 's':[], 'scale':[], 'n_points':[]}
    skipped = []

    for idx, row in salary_lookup.iterrows():
        percentile_vals = row[PERCENTILE_COLS].values.astype(float)

        keep_mask = ~np.isnan(percentile_vals) #keep_mask is TRUE where a val IS PRESENT (no NAN)
        n_points = keep_mask.sum()

        if n_points < 2:
            skipped.append((row['age_band'], row['occupation'])) #note (age_band, pairs) that have < 2 quantiles
            continue 

        clean_vals = percentile_vals[keep_mask]
        clean_probs = PERCENTILE_PROBS[keep_mask]

        params = fit_lognormal_single_row(clean_vals, clean_probs)
        s, scale = params[0], params[1]

        res['age_band'].append(row['age_band'])
        res['occupation'].append(row['occupation'])
        res['soc_code'].append(row['soc_code'])
        res['s'].append(s)
        res['scale'].append(scale)
        res['n_points'].append(n_points)


    #pairs that had < 2 quantiles
    if skipped:
        print(f"Skipped {len(skipped)} rows with fewer than 2 real percentile points:")
        for age_band, occupation in skipped:
            print(f"  - {age_band}, {occupation}")

    df = pd.DataFrame(res) 

    #dof to assess HOW well can I trust the fit NOT how well it fits
    df['degrees_of_freedom'] = df['n_points'] - 2  #2 params being fit (s, scale)
    df['confidence'] = np.where(df['degrees_of_freedom'] >= 4, 'high', 'low')

    return df


def _gamma_predict(u, a, scale):
    """
    given a probability (u), and params (a = shape/k, scale = theta), find 
    corresponding income value at that probability.

    for lognormal, we were able to use median as our starting scale in (s, scale),
    however, exp(mu) != median for gamma distribution. Therefore will be using 
    method of moments technique (see notebooks/02_income_fitting_findings.md for
    further details) to estimate a starting guess for curve fitting. 
    """

    return stats.gamma.ppf(u, a, scale = scale)

def fit_gamma_single_row(percentile_values: np.ndarray, percentile_probs = None, mean_value = None):

    """
    Fit Gamma to ONE row's known percentile values via quantile matching.
    Starting guess via method of moments (see notebooks/02_income_fitting_findings.md).
    """

    if percentile_probs is None:
        percentile_probs = PERCENTILE_PROBS 

    #chain of fallbacks
    if mean_value is None:
        median_idx = np.where(np.isclose(percentile_probs, 0.5))[0] 
        if len(median_idx) > 0: #median present
            mean_value = percentile_values[median_idx[0]]
        else:
            mean_value = np.mean(percentile_values)

    p_upper, p_lower = max(percentile_probs), min(percentile_probs)
    income_upper, income_lower = max(percentile_values), min(percentile_values)

    variance = ((income_upper - income_lower) / (stats.norm.ppf(p_upper) - stats.norm.ppf(p_lower))) ** 2

    a = mean_value ** 2 / variance #shape param
    scale = variance / mean_value

    p0 = [a, scale] 
    param, param_cov = curve_fit(_gamma_predict, percentile_probs, percentile_values, p0 = p0)

    return param

def fit_gamma_all_rows(salary_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fitting Gamma to every (age_band, occupation) pair in salary_lookup via
    quantile matching. Same missing-data approach as lognormal: fit on
    whatever real percentiles are present.
    """

    res = {'age_band':[], 'occupation':[], 'soc_code':[], 'a':[], 'scale':[], 'n_points':[]}
    skipped = []
    
    for idx, row in salary_lookup.iterrows():
        percentile_vals = row[PERCENTILE_COLS].values.astype(float)
    
        keep_mask = ~np.isnan(percentile_vals) 
        n_points = keep_mask.sum()
    
        if n_points < 2:
            skipped.append((row['age_band'], row['occupation'])) 
            continue 
    
        clean_vals = percentile_vals[keep_mask]
        clean_probs = PERCENTILE_PROBS[keep_mask]
        mean = row['mean']

        params = fit_gamma_single_row(clean_vals, clean_probs, mean_value = mean)
        a, scale = params[0], params[1]
    
        res['age_band'].append(row['age_band'])
        res['occupation'].append(row['occupation'])
        res['soc_code'].append(row['soc_code'])
        res['a'].append(a)
        res['scale'].append(scale)
        res['n_points'].append(n_points)
    
    
    if skipped:
        print(f"Skipped {len(skipped)} rows with fewer than 2 real percentile points:")
        for age_band, occupation in skipped:
            print(f"  - {age_band}, {occupation}")
    
    df = pd.DataFrame(res) 

    df['degrees_of_freedom'] = df['n_points'] - 2 
    df['confidence'] = np.where(df['degrees_of_freedom'] >= 4, 'high', 'low')
    
    return df
    

#quick sanity check to see if fitted scale is close to rows real median value before generalising to all 54 rows
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    path = os.path.join(REFERENCE_DIR, "salary_lookup_age_occupation_fulltime_2025.csv")
    df = pd.read_csv(path)

    row = df[(df['age_band'] == '30-39') & (df['occupation'] == 'Professional occupations')]
    percentile_values = row[PERCENTILE_COLS].values.flatten()

    print("Real percentile values:", percentile_values)

    #above I manually compared the printed 'scale' value against the row's
    #real median by eye - close (49010 vs 48190) but that only checks ONE point
    #below: proper check across all 11 percentiles, not just the median
    params = fit_lognormal_single_row(percentile_values) #returns best [s, scale] vals 
    print("Fitted params [s, scale]:", params)

    predicted = _lognormal_predict(PERCENTILE_PROBS, s = params[0], scale = params[1]) #feeds wrapper function all 11 probabilities AT ONCE (rather than one at a time) WITH BEST [s, scale] vals
    comparison = pd.DataFrame({
        'percentile': PERCENTILE_COLS,
        'real': percentile_values,
        'predicted': predicted,
    })
    comparison['pct_error'] = (comparison['predicted'] - comparison['real']) / comparison['real'] * 100
    print(comparison)

    #sanity check for fitting across all rows
    lognormal_params = fit_lognormal_all_rows(df)
    print(lognormal_params)
    print(lognormal_params['n_points'].value_counts())

    #gamma sanity check for specific row
    mean_value = row['mean'].values[0]
    print("Real mean:", mean_value)

    gamma_params = fit_gamma_single_row(percentile_values, mean_value=mean_value)
    print("Fitted gamma params [a, scale]:", gamma_params)

    gamma_predicted = _gamma_predict(PERCENTILE_PROBS, a=gamma_params[0], scale=gamma_params[1])
    gamma_comparison = pd.DataFrame({
        'percentile': PERCENTILE_COLS,
        'real': percentile_values,
        'predicted': gamma_predicted,
    })
    gamma_comparison['pct_error'] = (gamma_comparison['predicted'] - gamma_comparison['real']) / gamma_comparison['real'] * 100
    print(gamma_comparison)

    #sanity check for fitting gamma across all rows
    gamma_all_params = fit_gamma_all_rows(df)
    print(gamma_all_params)
    print(gamma_all_params['n_points'].value_counts())



        

        



     