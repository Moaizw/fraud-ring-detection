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
from scipy.special import gamma as gamma_func #used gamma elsewhere so better to use alias gamma_func
from scipy.special import betaincinv as inverse_beta_func
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

def _weibull_predict(u, c, scale):
    """
    given a probability (u) and weibull params (c = shape/k, scale = lambda),
    find corresponding income value at that probability. 
    """

    return stats.weibull_min.ppf(u, c, scale = scale)

def fit_weibull_single_row(percentile_values: np.ndarray, percentile_probs=None, mean_value=None):
    """
    fit Weibull to ONE row's known percentile values via quantile matching.
    Starting guess via CV-based approximation (see
    notebooks/02_income_fitting_findings.md for the reasoning).
    """ 

    if percentile_probs is None:
            percentile_probs = PERCENTILE_PROBS 
    
    #chain of fallbacks -> identical to fit_gamma_single_row
    if mean_value is None:
        median_idx = np.where(np.isclose(percentile_probs, 0.5))[0] 
        if len(median_idx) > 0: #median present
            mean_value = percentile_values[median_idx[0]]
        else:
            mean_value = np.mean(percentile_values)
    
    p_upper, p_lower = max(percentile_probs), min(percentile_probs)
    income_upper, income_lower = max(percentile_values), min(percentile_values)

    sd = ((income_upper - income_lower) / (stats.norm.ppf(p_upper) - stats.norm.ppf(p_lower)))
    cv = sd / mean_value 

    k = cv**-1.086 #-> CV to k APPROXIMATE formula 
    scale = mean_value / gamma_func(1 + 1/k) #find lambda (scale) using mean formula rearranged
    p0 = [k, scale]

    param, param_cov = curve_fit(_weibull_predict, percentile_probs, percentile_values, p0 = p0)

    return param

def fit_weibull_all_rows(salary_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fitting Weibull to every (age_band, occupation) pair in salary_lookup via
    quantile matching. Identical missing-data approach as lognormal/gamma: fit on
    whatever real percentiles are present.
    """

    res = {'age_band': [], 'occupation': [], 'soc_code': [], 'k': [], 'scale': [], 'n_points': []}
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

        params = fit_weibull_single_row(clean_vals, clean_probs, mean_value=mean)
        k, scale = params[0], params[1]

        res['age_band'].append(row['age_band'])
        res['occupation'].append(row['occupation'])
        res['soc_code'].append(row['soc_code'])
        res['k'].append(k)
        res['scale'].append(scale)
        res['n_points'].append(n_points)

    if skipped:
        print(f"Skipped {len(skipped)} rows with fewer than 2 real percentile points:")
        for age_band, occupation in skipped:
            print(f"  - {age_band}, {occupation}")

    df = pd.DataFrame(res)

    # dof to assess HOW well can I trust the fit NOT how well it fits
    df['degrees_of_freedom'] = df['n_points'] - 2  # 2 params being fit (k, scale)
    df['confidence'] = np.where(df['degrees_of_freedom'] >= 4, 'high', 'low')

    return df

def _gb2_ppf(u, a, b, p, q):
    """
    given probability u and GB2 params (a, b, p, q), return the income
    value GB2 predicts at that probability. not available in scipy.stats
    directly. See notebooks/02_income_fitting_findings.md for the full
    derivation and hand-worked test case.
    """

    #get z first
    z = inverse_beta_func(p, q, u) 

    #use rearranged formula (finding z) to compute x
    x = b * (z / (1 - z))**(1/a)

    return x

def fit_gb2_single_row(percentile_values: np.ndarray, percentile_probs=None, mean_value=None):
    """
    Fit GB2 to ONE row's known percentile values via quantile matching.
    Starting guess -> reuse already-fitted Gamma params for a, b; neutral
    p=1, q=1 (no tail assumption). See notebooks/02_income_fitting_findings.md
    for the reasoning.
    """
    if percentile_probs is None:
        percentile_probs = PERCENTILE_PROBS

    #same fallback chain as gamma/weibull
    if mean_value is None:
        median_idx = np.where(np.isclose(percentile_probs, 0.5))[0]
        if len(median_idx) > 0:
            mean_value = percentile_values[median_idx[0]]
        else:
            mean_value = np.mean(percentile_values)

    #reuse gamma's fit as starting a, b for GB2
    gamma_params = fit_gamma_single_row(percentile_values, percentile_probs, mean_value=mean_value)
    gamma_a, gamma_scale = gamma_params[0], gamma_params[1]

    p0 = [gamma_a, gamma_scale, 1, 1]  #[a, b, p, q] -> order must match _gb2_ppf
    bounds = (0, np.inf)

    params, param_cov = curve_fit(_gb2_ppf, percentile_probs, percentile_values, p0=p0, bounds=bounds)

    return params

def fit_gb2_all_rows(salary_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fitting GB2 to every (age_band, occupation) pair in salary_lookup via
    quantile matching. Same missing-data approach as lognormal/gamma/weibull:
    fit on whatever real percentiles are present. See
    notebooks/02_income_fitting_findings.md for findings/reasoning.
    """

    res = {'age_band': [], 'occupation': [], 'soc_code': [], 'a': [], 'b': [], 'p': [], 'q': [], 'n_points': []}
    skipped = []

    for idx, row in salary_lookup.iterrows():
        percentile_vals = row[PERCENTILE_COLS].values.astype(float)

        keep_mask = ~np.isnan(percentile_vals)
        n_points = keep_mask.sum()

        #updated to < 5 -> see notebooks/02_income_fitting_findings.md for reason
        if n_points < 5:
            skipped.append((row['age_band'], row['occupation']))
            continue

        clean_vals = percentile_vals[keep_mask]
        clean_probs = PERCENTILE_PROBS[keep_mask]
        mean = row['mean']

        params = fit_gb2_single_row(clean_vals, clean_probs, mean_value=mean)
        a, b, p, q = params[0], params[1], params[2], params[3]

        res['age_band'].append(row['age_band'])
        res['occupation'].append(row['occupation'])
        res['soc_code'].append(row['soc_code'])
        res['a'].append(a)
        res['b'].append(b)
        res['p'].append(p)
        res['q'].append(q)
        res['n_points'].append(n_points)

    if skipped:
        print(f"Skipped {len(skipped)} rows with fewer than 5 real percentile points (GB2 needs n_points >= 5 for at least 1 degree of freedom):")
        for age_band, occupation in skipped:
            print(f"  - {age_band}, {occupation}")

    df = pd.DataFrame(res)

    #4 params being fit (a, b, p, q) -> different from lognormal/gamma/weibull's 2
    #why i do n_points - 4 instead of n_points - 2
    df['degrees_of_freedom'] = df['n_points'] - 4 
    df['confidence'] = np.where(df['degrees_of_freedom'] >= 4, 'high', 'low')

    return df

def compute_aic_bic(real_values: np.ndarray, predicted_values: np.ndarray, k: int):
    """
    Compute AIC and BIC from residuals, for models fit via least squares
    (curve_fit gives residuals, not a likelihood, so this uses the standard
    RSS-based approximation rather than the textbook maximum-likelihood
    formula directly). See notebooks/02_income_fitting_findings.md.

    k = number of params the distribution has (2 for lognormal/gamma/
    weibull, 4 for GB2).
    """
    n = np.sum(~np.isnan(predicted_values))
    rss = 0

    for i in range(len(predicted_values)):
        rss += (predicted_values[i] - real_values[i])**2

    aic = n * np.log(rss / n) + 2 * k 
    bic = n * np.log(rss / n) + k * np.log(n)

    return (aic, bic)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    path = os.path.join(REFERENCE_DIR, "salary_lookup_age_occupation_fulltime_2025.csv")
    df = pd.read_csv(path)

    row = df[(df['age_band'] == '30-39') & (df['occupation'] == 'Professional occupations')]
    percentile_values = row[PERCENTILE_COLS].values.flatten()
    mean_value = row['mean'].values[0]


    lognormal_params = fit_lognormal_single_row(percentile_values)
    gamma_params = fit_gamma_single_row(percentile_values, mean_value=mean_value)
    weibull_params = fit_weibull_single_row(percentile_values, mean_value=mean_value)
    gb2_params = fit_gb2_single_row(percentile_values, mean_value=mean_value)

    lognormal_predicted = _lognormal_predict(PERCENTILE_PROBS, s=lognormal_params[0], scale=lognormal_params[1])
    gamma_predicted = _gamma_predict(PERCENTILE_PROBS, a=gamma_params[0], scale=gamma_params[1])
    weibull_predicted = _weibull_predict(PERCENTILE_PROBS, c=weibull_params[0], scale=weibull_params[1])
    gb2_predicted = _gb2_ppf(PERCENTILE_PROBS, a=gb2_params[0], b=gb2_params[1], p=gb2_params[2], q=gb2_params[3])

    results = []
    for name, predicted, k in [
    ('lognormal', lognormal_predicted, 2),
    ('gamma', gamma_predicted, 2),
    ('weibull', weibull_predicted, 2),
    ('gb2', gb2_predicted, 4),
    ]:
        aic, bic = compute_aic_bic(percentile_values, predicted, k)
        results.append({'distribution': name, 'k': k, 'aic': aic, 'bic': bic})

    results_df = pd.DataFrame(results).sort_values('bic')
    print(results_df)