"""
This file is for DIAGNOSTIC purposes only, not part of the core pipeline. 
Checks whether simulated spending, generated across a batch of real
accounts, aggregates to something resembling real A26 category 
proportions. 
"""

import numpy as np
import pandas as pd

from src.generation.accounts import generate_account_batch
from src.generation.spending import (
    load_spending_table,
    get_net_quintile_data,
    load_participation_rates,
    interpolate_parameters,
    draw_personal_profile,
    draw_personal_participation_rates,
    get_active_categories_this_week,
    CATEGORY_COLS,
)

from src.archetypes.full_time import load_age_band_distribution, load_salary_lookup, build_joint_table

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
GENERATED_DIR = os.path.join(REPO_ROOT, "data", "generated")

def check_aggregate_spending(accounts_df: pd.DataFrame, spending_table: pd.DataFrame, quintile_data: pd.DataFrame, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Accounts already have (age, occupation) + income from generate_account_batch.
    Tests whether spending.py's Layer 1 (personal mix + total, drawn once)
    is well-calibrated, by comparing against real A26 data. Not testing
    Layer 2 here, it only wobbles around an account's own Layer 1 profile,
    so a real miscalibration (like the alcohol bug) would show up in
    Layer 1, not Layer 2.

    For each account: find its quintile (interpolate_parameters), look up
    that quintile's real category proportions, draw ONE personal mix from
    draw_personal_profile.

    Accounts don't split evenly across quintiles, so a plain average of
    A26's 5 rows would misrepresent the simulated population. Instead,
    weight each quintile's real proportions by how many of the 1000
    accounts actually landed there (count / 1000), sum across quintiles.
    This gives the real, weighted TARGET, compared against the SIMULATED
    average (what draw_personal_profile actually produced across all 1000
    accounts).
    """

    if rng is None:
        rng = np.random.default_rng()

    results = []
    for _, account in accounts_df.iterrows():
        params = interpolate_parameters(account['net_income'], quintile_data)
        quintile_row = spending_table[spending_table['quintile'] == params['assigned_quintile']].iloc[0]
        profile = draw_personal_profile(quintile_row, params, rng=rng)

        row = dict(zip(CATEGORY_COLS, profile['personal_mix']))
        row['assigned_quintile'] = params['assigned_quintile']
        results.append(row)

    sim_df = pd.DataFrame(results)

    #simulated side: average of what draw_personal_profile actually produced
    simulated_avg = sim_df[CATEGORY_COLS].mean()

    #real side: weighted average of A26's real proportions, weighted by
    #how many simulated accounts landed in each quintile
    quintile_counts = sim_df['assigned_quintile'].value_counts().sort_index()
    weights = quintile_counts / quintile_counts.sum()

    real_target = {}
    for category in CATEGORY_COLS:
        quintile_props = []
        for q in weights.index:
            q_row = spending_table[spending_table['quintile'] == q].iloc[0]
            total = q_row[CATEGORY_COLS].astype(float).sum()
            quintile_props.append(q_row[category] / total)
        real_target[category] = np.average(quintile_props, weights=weights.values)

    real_target = pd.Series(real_target)

    comparison = pd.DataFrame({
        'simulated_avg': simulated_avg,
        'real_weighted_target': real_target,
    })
    comparison['difference'] = comparison['simulated_avg'] - comparison['real_weighted_target']
    comparison['pct_error'] = (comparison['difference'] / comparison['real_weighted_target']) * 100

    return comparison

def check_participation_calibration(personal_rates: dict, n_weeks: int = 1000, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Diagnostic only. Simulate many weeks of get_active_categories_this_week
    for one account's personal_rates, check whether the observed active
    frequency per category converges to that account's own personal rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    for _ in range(n_weeks):
        active = get_active_categories_this_week(personal_rates, rng=rng)
        results.append(active)

    active_df = pd.DataFrame(results)
    observed_freq = active_df.mean()  #fraction of weeks each category was active

    comparison = pd.DataFrame({
        'personal_rate': pd.Series(personal_rates),
        'observed_frequency': observed_freq,
    })
    comparison['difference'] = comparison['observed_frequency'] - comparison['personal_rate']

    return comparison


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    #rebuild full-time identity + load already fitted income results
    age_dist = load_age_band_distribution()
    salary_lookup = load_salary_lookup()
    full_time_joint_table = build_joint_table(age_dist, salary_lookup)

    lognormal_r = pd.read_csv(os.path.join(GENERATED_DIR, "lognormal_params_fulltime.csv"))
    gamma_r = pd.read_csv(os.path.join(GENERATED_DIR, "gamma_params_fulltime.csv"))
    weibull_r = pd.read_csv(os.path.join(GENERATED_DIR, "weibull_params_fulltime.csv"))
    gb2_r = pd.read_csv(os.path.join(GENERATED_DIR, "gb2_params_fulltime.csv"))
    comparison_r = pd.read_csv(os.path.join(GENERATED_DIR, "income_comparison_fulltime.csv"))

    spending_table = load_spending_table()
    quintile_data = get_net_quintile_data(spending_table)

    accounts = generate_account_batch(
        n=1000, joint_table=full_time_joint_table, comparison_table=comparison_r,
        lognormal_all=lognormal_r, gamma_all=gamma_r, weibull_all=weibull_r, gb2_all=gb2_r,
        spending_table=spending_table, quintile_data=quintile_data,
        archetype='full_time', rng=np.random.default_rng(seed=42),
    )

    result = check_aggregate_spending(accounts, spending_table, quintile_data, rng=np.random.default_rng(seed=42))
    print(result)

    #participation calibration check (self-contained, not from the batch,
    #since personal_rates isn't wired into generate_single_account yet)
    participation_table = load_participation_rates()
    rng = np.random.default_rng(seed=42)

    test_net_income = 35000
    params = interpolate_parameters(test_net_income, quintile_data)
    personal_rates = draw_personal_participation_rates(participation_table, rng=rng)

    print("\nParticipation calibration check:")
    calibration = check_participation_calibration(personal_rates, n_weeks=1000, rng=rng)
    print(calibration)