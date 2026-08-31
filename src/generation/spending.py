"""
Samples weekly spending, per category, for a simulated account. Uses a
two-layer compositional model:
- Layer 1 (personal mix + personal total): drawn once at account
  creation, centred on the account's income quintile from A26, loose
  spread/concentration -> creates person-to-person variation
- Layer 2 (week mix + week total): drawn fresh every week, centred on
  the account's OWN stored layer 1 numbers, tight spread/concentration
  -> creates realistic week-to-week noise

Category mix uses Dirichlet (compositional, shares sum to 1). Total uses
lognormal (positive, right-skewed, same reasoning as income). Neither
distribution is fitted to real data, no ONS spread data exists for
spending, both are documented assumptions anchored on real A26 averages.

Spread/concentration values interpolate linearly with an account's
actual NET income position between quintile medians (not just quintile
number), based on the assumption that discretionary spending freedom,
and therefore variation, grows with income.

See notebooks/03_spending_model_findings.md for full reasoning.
"""

import os
import numpy as np
import pandas as pd
from src.generation.tax import gross_to_net

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
REFERENCE_DIR = os.path.join(REPO_ROOT, "data", "reference")

CATEGORY_COLS = [
    'food_nonalcoholic', 'alcohol_tobacco', 'clothing_footwear',
    'housing_fuel_power', 'household_goods_services', 'health',
    'transport', 'communication', 'recreation_culture', 'education',
    'restaurants_hotels', 'misc_goods_services',
]

#layer 1: (personal, loose) and layer 2: (weekly, tight)
#lowest quintile -> highest quintile
LOGNORMAL_SPREAD_LAYER1 = (0.15, 0.35) #widens with income
LOGNORMAL_SPREAD_LAYER2 = (0.08, 0.14) #widens with income
DIRICHLET_CONC_LAYER1 = (25, 12) #SHRINKS with income (more variety)
DIRICHLET_CONC_LAYER2 = (80, 55) #SHRINKS with income (more variety)


def load_spending_table() -> pd.DataFrame:
    """
    Load spending_by_income_quintile_single_adult_2025.csv.
    """

    path = os.path.join(REFERENCE_DIR, "spending_by_income_quintile_single_adult_2025.csv")
    df = pd.read_csv(path)

    return df


def get_net_quintile_data(spending_table: pd.DataFrame) -> pd.DataFrame:
    """
    Convert both the lower boundary and median (gross weekly, already
    columns in spending_table) into annual NET figures.
    """

    spending_table['net_lower_boundary'] = spending_table['lower_boundary_weekly_gross'].apply(
        lambda x: gross_to_net(x * 52)
    )
    spending_table['net_median'] = spending_table['gross_income_median_weekly'].apply(
        lambda x: gross_to_net(x * 52)
    )
    return spending_table


def interpolate_parameters(net_income: float, quintile_data: pd.DataFrame) -> dict:
    """
    Given an account's net income:
    1. ASSIGNMENT: find which quintile net_income falls into, using
       quintile_data's net_lower_boundary column (which quintile's
       lower boundary is the highest one still <= net_income)
    2. INTERPOLATION: use quintile_data's net_median column as the known
       x-points for np.interp, to smoothly interpolate all 4
       spread/concentration values at net_income's exact position
       (clip to the endpoint values if net_income falls outside the
       full median range, rather than extrapolating)

    TODO: implement both steps, return a dict with all 4 interpolated
    values plus which quintile was assigned (useful for debugging/logging)
    """
    raise NotImplementedError


def draw_personal_profile(quintile_row: pd.Series, params: dict, rng: np.random.Generator = None) -> dict:
    """
    Layer 1: draw ONCE per account, using params' LAYER 1 values.
    """
    raise NotImplementedError


def draw_weekly_spending(personal_profile: dict, params: dict, rng: np.random.Generator = None) -> dict:
    """
    Layer 2: draw FRESH every week, using params' LAYER 2 values,
    centred on personal_profile (not the quintile again).
    """
    raise NotImplementedError


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df = load_spending_table()
    df = get_net_quintile_data(df)
    print(df)