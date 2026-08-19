"""
Two-step process to generate complete simulated account:
- Generates identity (age_band, occupation) from src/archetypes/{full_time,part_time}.py
- For each simulated identity, find WINNING distribution (from src/generation/income.py) and sample income.
"""

import os
import numpy as np
import pandas as pd
from src.generation.income import sample_income, PERCENTILE_COLS
from src.archetypes.full_time import load_age_band_distribution, load_salary_lookup, build_joint_table
from src.archetypes.part_time import load_age_band_distribution as pt_load_age, load_salary_lookup as pt_load_salary, build_joint_table as pt_build_joint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
GENERATED_DIR = os.path.join(REPO_ROOT, "data", "generated")

def generate_single_account(
    joint_table, comparison_table, lognormal_all, gamma_all, weibull_all, gb2_all,
    archetype, rng=None, max_retries=10,
) -> dict:
    """
    Generate one complete simulated account for the given archetype.
    Draws (age_band, occupation) from the archetype's joint probability
    table, then draws a gross income from whichever distribution won
    for that specific cell. Retries with a new identity draw if the
    sampled cell has no income data (identity and income pipelines have
    slightly different skip lists - see findings notes).
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_retries):
        idx = rng.choice(joint_table.index, p=joint_table['joint_probability'])
        identity_row = joint_table.loc[idx]
        age_band = identity_row['age_band']
        occupation = identity_row['occupation']

        try:
            gross_income = sample_income(
                age_band, occupation, comparison_table,
                lognormal_all, gamma_all, weibull_all, gb2_all,
                rng=rng,
            )
        except ValueError:
            continue

        return {
            'archetype': archetype,
            'age_band': age_band,
            'occupation': occupation,
            'soc_code': identity_row['soc_code'],
            'gross_income': gross_income,
        }

    raise RuntimeError(f"Failed to generate a valid account after {max_retries} retries")


def generate_account_batch(n, joint_table, comparison_table, lognormal_all, gamma_all,
                            weibull_all, gb2_all, archetype, rng=None) -> pd.DataFrame:
    """
    Generate n complete accounts for the given archetype. Returns a
    DataFrame, one row per account.
    """
    if rng is None:
        rng = np.random.default_rng()

    accounts = [
        generate_single_account(joint_table, comparison_table, lognormal_all, gamma_all,
                                 weibull_all, gb2_all, archetype, rng=rng)
        for _ in range(n)
    ]
    return pd.DataFrame(accounts)


if __name__ == "__main__":

    #FULL TIME ACCOUNT:
    #identity -> build the full-time joint table 
    age_dist = load_age_band_distribution()
    salary_lookup = load_salary_lookup()
    full_time_joint_table = build_joint_table(age_dist, salary_lookup)

    #load already built income tables (from income.py)
    lognormal_r = pd.read_csv(os.path.join(GENERATED_DIR, "lognormal_params_fulltime.csv"))
    gamma_r = pd.read_csv(os.path.join(GENERATED_DIR, "gamma_params_fulltime.csv"))
    weibull_r = pd.read_csv(os.path.join(GENERATED_DIR, "weibull_params_fulltime.csv"))
    gb2_r = pd.read_csv(os.path.join(GENERATED_DIR, "gb2_params_fulltime.csv"))
    comparison_r = pd.read_csv(os.path.join(GENERATED_DIR, "income_comparison_fulltime.csv"))

    accounts = generate_account_batch(
        n=1000, joint_table=full_time_joint_table, comparison_table=comparison_r,
        lognormal_all=lognormal_r, gamma_all=gamma_r, weibull_all=weibull_r, gb2_all=gb2_r,
        archetype='full_time', rng=np.random.default_rng(seed=42),
    )
    print(accounts.head())
    print(accounts.groupby('occupation')['gross_income'].median())

    #comparing real vs simulated account to see if ordering of median salary for occupations lines up
    real_salary_df = pd.read_csv(
    os.path.join(REPO_ROOT, "data", "reference", "salary_lookup_age_occupation_fulltime_2025.csv")
    )

    real_medians = real_salary_df.groupby('occupation')['median'].mean()
    print("Real ONS medians (averaged across age bands):")
    print(real_medians.sort_values())

    sim_medians = accounts.groupby('occupation')['gross_income'].median()
    print("\nSimulated medians:")
    print(sim_medians.sort_values())

    #PART-TIME ACCOUNTS:
    age_dist_pt = pt_load_age()
    salary_lookup_pt = pt_load_salary()
    part_time_joint_table = pt_build_joint(age_dist_pt, salary_lookup_pt)

    lognormal_pt = pd.read_csv(os.path.join(GENERATED_DIR, "lognormal_params_parttime.csv"))
    gamma_pt = pd.read_csv(os.path.join(GENERATED_DIR, "gamma_params_parttime.csv"))
    weibull_pt = pd.read_csv(os.path.join(GENERATED_DIR, "weibull_params_parttime.csv"))
    gb2_pt = pd.read_csv(os.path.join(GENERATED_DIR, "gb2_params_parttime.csv"))
    comparison_pt = pd.read_csv(os.path.join(GENERATED_DIR, "income_comparison_parttime.csv"))

    accounts_pt = generate_account_batch(
        n=1000, joint_table=part_time_joint_table, comparison_table=comparison_pt,
        lognormal_all=lognormal_pt, gamma_all=gamma_pt, weibull_all=weibull_pt, gb2_all=gb2_pt,
        archetype='part_time', rng=np.random.default_rng(seed=42),
    )
    print("\nPART-TIME")
    print(accounts_pt.head())

    real_salary_df_pt = pd.read_csv(
        os.path.join(REPO_ROOT, "data", "reference", "salary_lookup_age_occupation_parttime_2025.csv")
    )
    numeric_cols = PERCENTILE_COLS + ['jobs_thousand', 'mean', 'median']
    for col in numeric_cols:
        if col in real_salary_df_pt.columns:
            real_salary_df_pt[col] = pd.to_numeric(real_salary_df_pt[col], errors='coerce')

    real_medians_pt = real_salary_df_pt.groupby('occupation')['median'].mean()
    sim_medians_pt = accounts_pt.groupby('occupation')['gross_income'].median()

    print("\nReal ONS medians (part-time, averaged across age bands):")
    print(real_medians_pt.sort_values())
    print("\nSimulated medians (part-time):")
    print(sim_medians_pt.sort_values())