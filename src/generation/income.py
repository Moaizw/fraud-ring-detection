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
    Given probability (u), and params (a = shape/k, scale = theta), find 
    corresponding income value at that probability.
    """

    return stats.gamma.ppf(u, a, scale = scale)

def fit_gamma_single_row(percentile_values: np.ndarray, percentile_probs = None, mean_value = None):

    """
    Fit Gamma to ONE row's known percentile values via quantile matching.
    
    Unlike lognormal (where median = exp(mu) gives a clean starting scale),
    exp(mu) != median for Gamma, so method of moments is used instead to
    estimate a starting guess (see notebooks/02_income_fitting_findings.md).
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

    #dof to assess HOW well can I trust the fit NOT how well it fits
    df['degrees_of_freedom'] = df['n_points'] - 2  #2 params being fit (k, scale)
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

    *when running compare_distributions came across error where curve_fit
    genuinely struggled to converge on certain row(s) (not sure which yet)
    so gave up before finding an answer. Can happen with a 4 param model 
    on tricky shaped row, where search space harder to navigate through. 
    Added a few lines, which allows code to continue AND will flag the row
    where convergence failed.  
    """

    res = {'age_band': [], 'occupation': [], 'soc_code': [], 'a': [], 'b': [], 'p': [], 'q': [], 'n_points': []}
    skipped_insufficient_data = [] #-> rows where n < 5
    skipped_convergence_failure = []#-> rows where optimizer failed to converge

    for idx, row in salary_lookup.iterrows():
        percentile_vals = row[PERCENTILE_COLS].values.astype(float)

        keep_mask = ~np.isnan(percentile_vals)
        n_points = keep_mask.sum()

        if n_points < 5:
            skipped_insufficient_data.append((row['age_band'], row['occupation']))
            continue

        clean_vals = percentile_vals[keep_mask]
        clean_probs = PERCENTILE_PROBS[keep_mask]
        mean = row['mean']

        try:
            params = fit_gb2_single_row(clean_vals, clean_probs, mean_value=mean)
        except RuntimeError:
            skipped_convergence_failure.append((row['age_band'], row['occupation']))
            continue

        a, b, p, q = params[0], params[1], params[2], params[3]

        res['age_band'].append(row['age_band'])
        res['occupation'].append(row['occupation'])
        res['soc_code'].append(row['soc_code'])
        res['a'].append(a)
        res['b'].append(b)
        res['p'].append(p)
        res['q'].append(q)
        res['n_points'].append(n_points)

    if skipped_insufficient_data:
        print(f"Skipped {len(skipped_insufficient_data)} rows with fewer than 5 real percentile points:")
        for age_band, occupation in skipped_insufficient_data:
            print(f"  - {age_band}, {occupation}")

    if skipped_convergence_failure:
        print(f"Skipped {len(skipped_convergence_failure)} rows where curve_fit failed to converge:")
        for age_band, occupation in skipped_convergence_failure:
            print(f"  - {age_band}, {occupation}")

    df = pd.DataFrame(res)
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

def compare_distributions(
    salary_lookup: pd.DataFrame,
    lognormal_all: pd.DataFrame,
    gamma_all: pd.DataFrame,
    weibull_all: pd.DataFrame,
    gb2_all: pd.DataFrame) -> pd.DataFrame:
    """
    For every row, recompute each distribution's predicted values using its
    already-fitted params, compute AIC/BIC, and record the winner (lowest
    BIC). GB2 may be missing for some rows (n_points < 5); handled as NaN
    for that row rather than dropping the row entirely.
    """
    results = []

    for idx, row in salary_lookup.iterrows():
        age_band, occupation = row['age_band'], row['occupation']

        percentile_vals = row[PERCENTILE_COLS].values.astype(float)
        keep_mask = ~np.isnan(percentile_vals)
        n_points = keep_mask.sum()

        if n_points < 2:
            continue  #same skip seen in fitting functions -> nothing to compare

        clean_vals = percentile_vals[keep_mask]
        clean_probs = PERCENTILE_PROBS[keep_mask]

        row_result = {'age_band': age_band, 'occupation': occupation, 'n_points': n_points}
        bics = {}

        #lognormal
        match = lognormal_all[(lognormal_all['age_band'] == age_band) & (lognormal_all['occupation'] == occupation)]
        if not match.empty:
            s, scale = match.iloc[0]['s'], match.iloc[0]['scale']
            predicted = _lognormal_predict(clean_probs, s=s, scale=scale)
            aic, bic = compute_aic_bic(clean_vals, predicted, k=2)
            row_result['lognormal_aic'], row_result['lognormal_bic'] = aic, bic
            bics['lognormal'] = bic
        else:
            row_result['lognormal_aic'], row_result['lognormal_bic'] = np.nan, np.nan

        #gamma
        match = gamma_all[(gamma_all['age_band'] == age_band) & (gamma_all['occupation'] == occupation)]
        if not match.empty:
            a, scale = match.iloc[0]['a'], match.iloc[0]['scale']
            predicted = _gamma_predict(clean_probs, a=a, scale=scale)
            aic, bic = compute_aic_bic(clean_vals, predicted, k=2)
            row_result['gamma_aic'], row_result['gamma_bic'] = aic, bic
            bics['gamma'] = bic
        else:
            row_result['gamma_aic'], row_result['gamma_bic'] = np.nan, np.nan

        #weibull
        match = weibull_all[(weibull_all['age_band'] == age_band) & (weibull_all['occupation'] == occupation)]
        if not match.empty:
            k_param, scale = match.iloc[0]['k'], match.iloc[0]['scale']
            predicted = _weibull_predict(clean_probs, c=k_param, scale=scale)
            aic, bic = compute_aic_bic(clean_vals, predicted, k=2)
            row_result['weibull_aic'], row_result['weibull_bic'] = aic, bic
            bics['weibull'] = bic
        else:
            row_result['weibull_aic'], row_result['weibull_bic'] = np.nan, np.nan

        #gb2 -> may genuinely be absent (n_points < 5 during fitting)
        match = gb2_all[(gb2_all['age_band'] == age_band) & (gb2_all['occupation'] == occupation)]
        if not match.empty:
            a, b, p, q = match.iloc[0]['a'], match.iloc[0]['b'], match.iloc[0]['p'], match.iloc[0]['q']
            predicted = _gb2_ppf(clean_probs, a=a, b=b, p=p, q=q)
            aic, bic = compute_aic_bic(clean_vals, predicted, k=4)
            row_result['gb2_aic'], row_result['gb2_bic'] = aic, bic
            bics['gb2'] = bic
        else:
            row_result['gb2_aic'], row_result['gb2_bic'] = np.nan, np.nan

        row_result['winner'] = min(bics, key=lambda k: bics[k]) if bics else None

        results.append(row_result)

    return pd.DataFrame(results)

def sample_income(
        age_band: str,
        occupation: str,
        winner_table: pd.DataFrame,
        lognormal_all: pd.DataFrame,
        gamma_all: pd.DataFrame,
        weibull_all: pd.DataFrame,
        gb2_all: pd.DataFrame,
        rng: np.random.Generator = None
    ) -> float:
    """
    Draw one realistic gross annual income for a single simulated account,
    given its age_band and occupation. Looks up which distribution won for
    this cell -> from compare_distributions output, draws one random
    probability, and feeds it through that distribution's own ppf
    function (inverse transform sampling).
    """
    if rng is None:
        rng = np.random.default_rng()

    match = winner_table[
        (winner_table['age_band'] == age_band) & (winner_table['occupation'] == occupation)
    ]
    if match.empty:
        raise ValueError(f"No winner found for {age_band}, {occupation}")

    winner = match.iloc[0]['winner']

    #u = 0 or 1 are probs that can generate nonsense vals like 0 or infinite
    #particularly for GB2 ppf function therefore 0.001 & 0.999 used
    u = rng.uniform(0.001, 0.999)

    if winner == 'lognormal':
        row = lognormal_all[(lognormal_all['age_band'] == age_band) & (lognormal_all['occupation'] == occupation)].iloc[0]
        return _lognormal_predict(u, s=row['s'], scale=row['scale'])

    elif winner == 'gamma':
        row = gamma_all[(gamma_all['age_band'] == age_band) & (gamma_all['occupation'] == occupation)].iloc[0]
        return _gamma_predict(u, a=row['a'], scale=row['scale'])

    elif winner == 'weibull':
        row = weibull_all[(weibull_all['age_band'] == age_band) & (weibull_all['occupation'] == occupation)].iloc[0]
        return _weibull_predict(u, c=row['k'], scale=row['scale'])

    elif winner == 'gb2':
        row = gb2_all[(gb2_all['age_band'] == age_band) & (gb2_all['occupation'] == occupation)].iloc[0]
        return _gb2_ppf(u, a=row['a'], b=row['b'], p=row['p'], q=row['q'])

    else:
        raise ValueError(f"Unrecognised winner '{winner}' for {age_band}, {occupation}")

def check_fit(salary_df, params_table, distribution, age_band, occupation):
    row = salary_df[(salary_df['age_band'] == age_band) & (salary_df['occupation'] == occupation)].iloc[0]
    percentile_vals = row[PERCENTILE_COLS].values.astype(float)
    keep_mask = ~np.isnan(percentile_vals)
    clean_vals = percentile_vals[keep_mask]
    clean_probs = PERCENTILE_PROBS[keep_mask]

    param_row = params_table[(params_table['age_band'] == age_band) & (params_table['occupation'] == occupation)]
    if param_row.empty:
        print(f"{age_band}, {occupation}: no fitted params found for {distribution}")
        return
    param_row = param_row.iloc[0]

    if distribution == 'lognormal':
        predicted = _lognormal_predict(clean_probs, s=param_row['s'], scale=param_row['scale'])
    elif distribution == 'gamma':
        predicted = _gamma_predict(clean_probs, a=param_row['a'], scale=param_row['scale'])
    elif distribution == 'weibull':
        predicted = _weibull_predict(clean_probs, c=param_row['k'], scale=param_row['scale'])
    elif distribution == 'gb2':
        predicted = _gb2_ppf(clean_probs, a=param_row['a'], b=param_row['b'], p=param_row['p'], q=param_row['q'])

    check = pd.DataFrame({'prob': clean_probs, 'real': clean_vals, 'predicted': predicted})
    check['pct_error'] = (check['predicted'] - check['real']) / check['real'] * 100
    max_err = check['pct_error'].abs().max()
    print(f"\n{age_band}, {occupation} ({distribution}) -> max abs error: {max_err:.2f}%")
    print(check[['prob', 'real', 'predicted', 'pct_error']])

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    out_dir = os.path.join(REPO_ROOT, "data", "generated")
    os.makedirs(out_dir, exist_ok=True)

    archetypes = [
        ('fulltime', 'salary_lookup_age_occupation_fulltime_2025.csv'),
        ('parttime', 'salary_lookup_age_occupation_parttime_2025.csv'),
    ]

    for label, filename in archetypes:
        print(f"\n {label}")
        path = os.path.join(REFERENCE_DIR, filename)
        salary_df = pd.read_csv(path)

        #cleaning -> address part_time table has missing rows as 'x' 
        numeric_cols = PERCENTILE_COLS + ['jobs_thousand', 'mean']
        for col in numeric_cols:
            if col in salary_df.columns:
                salary_df[col] = pd.to_numeric(salary_df[col], errors='coerce') #coerce turns to real NaN

        lognormal_r = fit_lognormal_all_rows(salary_df)
        gamma_r = fit_gamma_all_rows(salary_df)
        weibull_r = fit_weibull_all_rows(salary_df)
        gb2_r = fit_gb2_all_rows(salary_df)
        comparison_r = compare_distributions(salary_df, lognormal_r, gamma_r, weibull_r, gb2_r)

        lognormal_r.to_csv(os.path.join(out_dir, f"lognormal_params_{label}.csv"), index=False)
        gamma_r.to_csv(os.path.join(out_dir, f"gamma_params_{label}.csv"), index=False)
        weibull_r.to_csv(os.path.join(out_dir, f"weibull_params_{label}.csv"), index=False)
        gb2_r.to_csv(os.path.join(out_dir, f"gb2_params_{label}.csv"), index=False)
        comparison_r.to_csv(os.path.join(out_dir, f"income_comparison_{label}.csv"), index=False)

        print(comparison_r['winner'].value_counts())

        gb2_winners = comparison_r[comparison_r['winner'] == 'gb2'][['age_band', 'occupation']]
        gb2_winner_params = gb2_r.merge(gb2_winners, on=['age_band', 'occupation'])

        print(gb2_winner_params[['age_band', 'occupation', 'a', 'b', 'p', 'q']])
        print(gb2_winner_params['q'].describe())

        #pick 5 rows per winning distribution from the part-time comparison table
        for dist in ['lognormal', 'gamma', 'weibull', 'gb2']:
            winners = comparison_r[comparison_r['winner'] == dist][['age_band', 'occupation']]
            sample = winners.head(5) 

            if sample.empty:
                print(f"\nNo rows won by {dist}")
                continue

            params_table = {'lognormal': lognormal_r, 'gamma': gamma_r, 'weibull': weibull_r, 'gb2': gb2_r}[dist]

            for _, r in sample.iterrows():
                check_fit(salary_df, params_table, dist, r['age_band'], r['occupation'])

        

