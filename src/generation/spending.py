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

Not every category is spent on every week. Categories with real ONS
participation under 95% (alcohol_tobacco, clothing_footwear,
restaurants_hotels, transport, household_goods_services) get a
per-account personal weekly participation PROBABILITY, drawn once via
Beta, centred on the real rate. Each week, a fresh draw against that
probability decides whether the category is active (normal Layer 2
draw happens) or inactive (£0 that category, that week, and the
week's total shrinks accordingly rather than being redistributed to
other categories).

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

PARTICIPATION_THRESHOLD = 0.95
FLAGGED_CATEGORIES = ['alcohol_tobacco', 'clothing_footwear', 'restaurants_hotels', 'transport', 'household_goods_services']

#Layer 1 spread for the personal participation rate itself,
#same as the spending spread/concentration choices above
PARTICIPATION_RATE_CONCENTRATION = 50  


def load_spending_table() -> pd.DataFrame:
    """
    Load spending_by_income_quintile_single_adult_2025.csv.
    """

    path = os.path.join(REFERENCE_DIR, "spending_by_income_quintile_single_adult_2025.csv")
    df = pd.read_csv(path)

    return df

def load_participation_rates() -> pd.DataFrame:
    """
    Load category_participation_rates_2025.csv.
    """
    path = os.path.join(REFERENCE_DIR, "category_participation_rates_2025.csv")
    return pd.read_csv(path)


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

def draw_personal_participation_rates(participation_table: pd.DataFrame, rng: np.random.Generator = None) -> dict:
    """
    Layer 1: draw ONCE per account. For each FLAGGED category, draw a
    personal weekly participation probability from a Beta distribution
    centred on the real ONS rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    rates = {} #stores participation rates for EACH flagged entry/account
    for category in FLAGGED_CATEGORIES:
        real_rate = participation_table[participation_table['category'] == category]['participation_rate'].iloc[0] #gets real ONS category proportion

        #Beta(a, b) has mean = a / (a+b).
        #a+b is CONCENTRATION/SPREAD, and this is important
        #because it determines how far draws are from avg 
        #smaller conc -> more spread out samples drawn 
        #high conc -> clustered closely around avg
        a = real_rate * PARTICIPATION_RATE_CONCENTRATION #a = success weight (spends on X category for THAT week)
        b = (1 - real_rate) * PARTICIPATION_RATE_CONCENTRATION #b = non participation weight (£0 spent on X category)

        rates[category] = rng.beta(a, b)

    return rates

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
    participation_table = load_participation_rates()
    rng = np.random.default_rng(seed=42)

    rates_sample = [draw_personal_participation_rates(participation_table, rng=rng) for _ in range(1000)]
    rates_df = pd.DataFrame(rates_sample)
    print(rates_df.describe())