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
from scipy import stats
from src.generation.tax import gross_to_net

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
REFERENCE_DIR = os.path.join(REPO_ROOT, "data", "reference")

CATEGORY_COLS = [
    'food_nonalcoholic', 'alcohol_tobacco', 'clothing_footwear',
    'housing_fuel_power', 'household_goods_services',
    'transport', 'communication', 'recreation_culture',
    'restaurants_hotels', 'misc_goods_services', 'other_expenditure_items'
]

#layer 1: (personal, loose) and layer 2: (weekly, tight)
#lowest quintile -> highest quintile
LOGNORMAL_SPREAD_LAYER1 = (0.15, 0.35) #widens with income
LOGNORMAL_SPREAD_LAYER2 = (0.08, 0.14) #widens with income
DIRICHLET_CONC_LAYER1 = (150, 80) #SHRINKS with income (more variety); updated concentration range, see 03_spending_model_findings.md
DIRICHLET_CONC_LAYER2 = (500, 300) #SHRINKS with income (more variety)


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
       (clip endpoint vals if net income falls outside median range, DON'T 
       extrapolate)
    """

    quintile_band = quintile_data[quintile_data['net_lower_boundary'] <= net_income]
    assigned_quintile = quintile_band['quintile'].max()

    xp = quintile_data['net_median']

    spread_layers = {
        'lognormal_spread_1': LOGNORMAL_SPREAD_LAYER1,
        'lognormal_spread_2': LOGNORMAL_SPREAD_LAYER2,
        'dirichlet_conc_1': DIRICHLET_CONC_LAYER1,
        'dirichlet_conc_2': DIRICHLET_CONC_LAYER2,
    }

    result = {}
    for name, spread in spread_layers.items():
        fp = np.linspace(spread[0], spread[1], num=5)
        result[name] = np.interp(net_income, xp, fp)

    result['assigned_quintile'] = assigned_quintile

    return result

def draw_personal_profile(quintile_row: pd.Series, params: dict, rng: np.random.Generator = None) -> dict:
    """
    Layer 1: draw ONCE per account, using params' LAYER 1 values.
    """
    if rng is None:
        rng = np.random.default_rng()

    category_values = quintile_row[CATEGORY_COLS].astype(float)
    total_spend = category_values.sum()

    personal_total = stats.lognorm.rvs(s=params['lognormal_spread_1'], scale=total_spend, random_state=rng)

    proportions = category_values / total_spend
    alpha = proportions * params['dirichlet_conc_1']
    personal_mix = rng.dirichlet(alpha)

    return {
        'personal_total': personal_total,
        'personal_mix': personal_mix,
    }


def draw_weekly_spending(personal_profile: dict, params: dict, rng: np.random.Generator = None) -> dict:
    """
    Layer 2: draw FRESH every week, using params LAYER 2 values,
    centred on personal_profile (not the quintile again).
    """

    if rng is None:
        rng = np.random.default_rng()

    week_total = stats.lognorm.rvs(
        s=params['lognormal_spread_2'],
        scale=personal_profile['personal_total'],
        random_state=rng,
    )

    alpha_week = personal_profile['personal_mix'] * params['dirichlet_conc_2']
    week_mix = rng.dirichlet(alpha_week)

    category_amounts = week_total * week_mix

    return {
        'week_total': week_total,
        'week_mix': week_mix,
        'category_amounts': dict(zip(CATEGORY_COLS, category_amounts)),
    }


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    df = load_spending_table()
    df = get_net_quintile_data(df)

    rng = np.random.default_rng(seed=42)

    test_net_income = 35000
    params = interpolate_parameters(test_net_income, df)
    print("Interpolated params:", params)

    quintile_row = df[df['quintile'] == params['assigned_quintile']].iloc[0]

    profile = draw_personal_profile(quintile_row, params, rng=rng)
    print("\nPersonal profile:")
    print("Personal total:", profile['personal_total'])
    print("Personal mix sums to:", profile['personal_mix'].sum())
    print("Personal mix:", dict(zip(CATEGORY_COLS, profile['personal_mix'])))

    print("\nFive weeks of spending for this account:")
    for week in range(5):
        week_result = draw_weekly_spending(profile, params, rng=rng)
        total_check = sum(week_result['category_amounts'].values())
        print(f"Week {week+1}: total={week_result['week_total']:.2f}, "
              f"sum of categories={total_check:.2f}")