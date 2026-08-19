"""
Two-step process to generate complete simulated account:
- Generates identity (age_band, occupation) from src/archetypes/{full_time,part_time}.py
- For each simulated identity, find WINNING distribution (from src/generation/income.py) and sample income.
"""

import os
import numpy as np
import pandas as pd
from src.generation.income import sample_income
from src.archetypes.full_time import load_age_band_distribution, load_salary_lookup, build_joint_table

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
GENERATED_DIR = os.path.join(REPO_ROOT, "data", "generated")



def generate_account(
    joint_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    lognormal_all: pd.DataFrame,
    gamma_all: pd.DataFrame,
    weibull_all: pd.DataFrame,
    gb2_all: pd.DataFrame,
    archetype: str,
    rng: np.random.Generator = None,
) -> dict:
    """
    Generate one complete simulated account for the given archetype
    ('full_time' or 'part_time'). Draws (age_band, occupation) from the
    archetype's joint probability table, then draws a gross income from
    whichever distribution won for that specific cell.
    """
    if rng is None:
        rng = np.random.default_rng()

    idx = rng.choice(joint_table.index, p=joint_table['joint_probability'])
    identity_row = joint_table.loc[idx]
    age_band = identity_row['age_band']
    occupation = identity_row['occupation']

    gross_income = sample_income(
        age_band, occupation, comparison_table,
        lognormal_all, gamma_all, weibull_all, gb2_all,
        rng=rng,
    )

    return {
        'archetype': archetype,
        'age_band': age_band,
        'occupation': occupation,
        'soc_code': identity_row['soc_code'],
        'gross_income': gross_income,
    }


def generate_accounts(n, joint_table, comparison_table, lognormal_all, gamma_all,
                       weibull_all, gb2_all, archetype, rng=None) -> pd.DataFrame:
    """
    Generate n complete accounts for the given archetype. Returns a
    DataFrame, one row per account.
    """
    if rng is None:
        rng = np.random.default_rng()

    accounts = [
        generate_account(joint_table, comparison_table, lognormal_all, gamma_all,
                          weibull_all, gb2_all, archetype, rng=rng)
        for _ in range(n)
    ]
    return pd.DataFrame(accounts)


if __name__ == "__main__":
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

    accounts = generate_accounts(
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