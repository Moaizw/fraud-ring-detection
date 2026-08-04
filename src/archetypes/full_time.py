"""
Full Time Employee Archetype:

This file will contain code to produce a realistic (age band, occupation) pair
for a single simulated full time account, weighted by real ONS data. It does
NOT handle income sampling, gross to net conversion, or spending. Those belong
in a separate file.

Data sources used here:
- data/reference/age_band_distribution_by_archetype_2025.csv
- data/reference/salary_lookup_age_occupation_fulltime_2025.csv
"""

import os
import math
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
REFERENCE_DIR = os.path.join(REPO_ROOT, "data", "reference")

ARCHETYPE = "full_time"


def load_age_band_distribution() -> pd.DataFrame:

    path = os.path.join(REFERENCE_DIR, "age_band_distribution_by_archetype_2025.csv")
    df = pd.read_csv(path)
    full_time = df[df['archetype'] == ARCHETYPE][['age_band', 'population_share']]

    # sanity check
    total = sum(full_time['population_share'])
    if math.isclose(total, 1.0):
        print('Successfully retrieved all data')
    else:
        print(f'Error: population_share sums to {total}, not 1.0')

    return full_time

def load_salary_lookup() -> pd.DataFrame:

    path = os.path.join(REFERENCE_DIR, "salary_lookup_age_occupation_fulltime_2025.csv")
    df = pd.read_csv(path)

    occupation = df[['age_band','soc_code','occupation','jobs_thousand']] #relevant features selected ONLY

    #checked and no missing rows in jobs_thousands in Full Time so no dropping required 

    return occupation

def build_joint_table(
    age_distribution: pd.DataFrame, salary_lookup: pd.DataFrame
) -> pd.DataFrame:
    """
    Here, I use conditional probability to find the probability of two events occuring. 
    These events are age group (6) and occupation (9):
    P(X age AND Y occupation) = P(X age) * P(Y occupation | X age)

    Once conditional probability computed, create flattened table containing features:
    1. age_band
    2. occupation
    3. soc_code
    4. joint_probability

    This table will be used for the direct weighted sampling.

    """

    merged = pd.merge(salary_lookup, age_distribution, on = 'age_band')[['age_band','population_share','soc_code','occupation','jobs_thousand']]
    merged['occupation_share'] = merged['jobs_thousand'] / merged.groupby('age_band')['jobs_thousand'].transform('sum')
    merged['joint_probability'] = merged['population_share'] * merged['occupation_share']

    res = merged[['age_band','occupation','soc_code','joint_probability']]

    total_prob = res['joint_probability'].sum()

    if math.isclose(total_prob, 1.0):
        print('Success')
    else:
        print('Error')

    return res

def sample_age_occupation(joint_table: pd.DataFrame, n: int, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Draw n (age_band, occupation) pairs from the joint table, weighted by
    joint_probability.
    """
    if rng is None:
        rng = np.random.default_rng()

    random_idx = rng.choice(a = joint_table.index, size = n, p = joint_table['joint_probability'])
    df = joint_table.loc[random_idx]

    return df

if __name__ == "__main__":
    age = load_age_band_distribution()
    salary = load_salary_lookup()
    joint = build_joint_table(age, salary)

    accounts = sample_age_occupation(joint, n=10000, rng=np.random.default_rng(seed=42))
    print(accounts[['age_band', 'occupation']].value_counts(normalize=True))