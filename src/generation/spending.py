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

#card transaction categories
CARD_CATEGORIES = [
    'food_nonalcoholic', 'alcohol_tobacco', 'clothing_footwear',
    'household_goods_services', 'transport', 'recreation_culture',
    'restaurants_hotels', 'misc_goods_services',
]

#debit card (recurring) categories
DIRECT_DEBIT_CATEGORIES = ['housing_fuel_power', 'communication']

EXCLUDED_FROM_TRANSACTIONS = ['other_expenditure_items']

#rough transaction cnt range per category, when active
CARD_TRANSACTION_COUNTS = {
    'food_nonalcoholic': (3, 5),
    'alcohol_tobacco': (1, 2),
    'clothing_footwear': (1, 1),
    'household_goods_services': (1, 1),
    'transport': (2, 4),
    'recreation_culture': (1, 2),
    'restaurants_hotels': (1, 2),
    'misc_goods_services': (1, 2),
}

#how likely someone would spend on specific category for each day in a week (ASSUMPTION)
#e.g. person likely to spend on restaurant/hotels on weekends than weekdays
DAY_OF_WEEK_WEIGHTS = {
    'food_nonalcoholic':          [0.12, 0.12, 0.13, 0.13, 0.14, 0.18, 0.18],
    'alcohol_tobacco':            [0.10, 0.10, 0.10, 0.12, 0.16, 0.21, 0.21],
    'clothing_footwear':          [0.13, 0.13, 0.13, 0.14, 0.14, 0.17, 0.16],
    'household_goods_services':   [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
    'transport':                  [0.18, 0.18, 0.18, 0.18, 0.16, 0.06, 0.06],
    'recreation_culture':         [0.11, 0.11, 0.11, 0.12, 0.15, 0.20, 0.20],
    'restaurants_hotels':         [0.09, 0.09, 0.09, 0.11, 0.16, 0.23, 0.23],
    'misc_goods_services':        [0.14, 0.14, 0.14, 0.14, 0.14, 0.15, 0.15],
}


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

#-- LAYER 1 -> DRAWN ONCE / ACCOUNT AT CREATION --

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

#-- LAYER 2 -> DRAWN FRESH EVERY WEEK -- 

def get_active_categories_this_week(personal_rates: dict, rng: np.random.Generator = None) -> dict:
    """
    For each flagged category, draw one fresh random number 0-1 and
    compare against the account's personal rate. Returns a dict of
    category -> True/False (active/inactive) for THIS week only.
    """
    if rng is None:
        rng = np.random.default_rng()

    active = {}
    for category, rate in personal_rates.items():
        draw = rng.uniform(0, 1)
        active[category] = draw <= rate

    return active

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

def draw_weekly_spending_with_participation(
    personal_profile: dict, params: dict, personal_rates: dict, rng: np.random.Generator = None
) -> dict:
    """
    Layer 2, with participation. Same as draw_weekly_spending, but
    flagged categories may be inactive this week (£0, total shrinks
    accordingly, money not redistributed). See
    notebooks/03_spending_model_findings.md for the full reasoning.
    """
    if rng is None:
        rng = np.random.default_rng()

    active_flags = get_active_categories_this_week(personal_rates, rng=rng)

    personal_mix = dict(zip(CATEGORY_COLS, personal_profile['personal_mix']))

    #a category IS ACTIVE if it's not flagged at all (always spent on),
    #OR if it IS flagged and this week's coin-flip came back True
    inactive_categories = [
        cat for cat in FLAGGED_CATEGORIES
        if not active_flags.get(cat, True)
    ]
    active_categories = [cat for cat in CATEGORY_COLS if cat not in inactive_categories]

    #fraction of the personal_mix that belongs to inactive categories,
    #this is the share of a "normal" week's spend that isn't happening
    inactive_share = sum(personal_mix[cat] for cat in inactive_categories)

    #normal week_total draw, exactly as before
    raw_week_total = stats.lognorm.rvs(
        s=params['lognormal_spread_2'],
        scale=personal_profile['personal_total'],
        random_state=rng,
    )
    #shrink it, since the inactive share of spending simply isn't happening
    week_total = raw_week_total * (1 - inactive_share)

    #smaller Dirichlet, only over the active categories
    active_proportions = np.array([personal_mix[cat] for cat in active_categories])
    alpha_week = active_proportions * params['dirichlet_conc_2']
    week_mix_active = rng.dirichlet(alpha_week)

    category_amounts = {cat: 0.0 for cat in CATEGORY_COLS}
    for cat, share in zip(active_categories, week_mix_active):
        category_amounts[cat] = week_total * share

    return {
        'week_total': week_total,
        'active_categories': active_categories,
        'inactive_categories': inactive_categories,
        'category_amounts': category_amounts,
    }

#-- CARD TRANSACTIONS --

def split_into_card_transactions(category: str, weekly_amount: float, week_start_date, rng: np.random.Generator = None) -> list:
    """
    Split one category's weekly £ amount into several individual, dated
    card transactions within that week.
    """
    if rng is None:
        rng = np.random.default_rng()

    if weekly_amount <= 0:
        return []

    low, high = CARD_TRANSACTION_COUNTS[category]
    n_transactions = rng.integers(low, high + 1)

    #purposely split total across n_transactions, with a little randomness
    #rather than perfectly even amounts
    raw_splits = rng.dirichlet(np.ones(n_transactions) * 20)
    amounts = weekly_amount * raw_splits

    weights = DAY_OF_WEEK_WEIGHTS[category]
    days_offset = rng.choice(7, size=n_transactions, replace=True, p=weights)
    dates = [week_start_date + pd.Timedelta(days=int(d)) for d in days_offset]

    return [
        {'category': category, 'amount': amt, 'date': date}
        for amt, date in zip(amounts, dates)
    ]

#-- DIRECT DEBIT TRANSACTIONS --

WEEKS_PER_MONTH = 4.33  #required to sum up weekly avg 

def generate_monthly_direct_debit_amounts(week_result:dict, personal_profile: dict) -> dict:
    """
    Scale a single week's Layer 2 result up to a monthly Direct Debit
    amount for DIRECT_DEBIT_CATEGORIES (~4.33 weeks per month).
    """
    monthly_amounts = {}
    for category in DIRECT_DEBIT_CATEGORIES:
        weekly_amount = week_result['category_amounts'][category]
        monthly_amounts[category] = weekly_amount * WEEKS_PER_MONTH

    return monthly_amounts

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

    #participation adjusted weekly spending
    participation_table = load_participation_rates()
    personal_rates = draw_personal_participation_rates(participation_table, rng=rng)
    print("\nPersonal participation rates:", personal_rates)

    print("\nFive weeks of spending WITH participation for this account:")
    for week in range(5):
        week_result = draw_weekly_spending_with_participation(profile, params, personal_rates, rng=rng)
        total_check = sum(week_result['category_amounts'].values())
        print(f"Week {week+1}: total={week_result['week_total']:.2f}, "
              f"sum of categories={total_check:.2f}, "
              f"inactive={week_result['inactive_categories']}")

    print("\nDirect Debit sanity check:")

    week_result = draw_weekly_spending(profile, params, rng=rng)
    monthly_amounts = generate_monthly_direct_debit_amounts(week_result, profile)

    for category in DIRECT_DEBIT_CATEGORIES:
        weekly_amount = week_result['category_amounts'][category]
        monthly_amount = monthly_amounts[category]
        ratio = monthly_amount / weekly_amount
        print(f"{category}: weekly={weekly_amount:.2f}, monthly={monthly_amount:.2f}, ratio={ratio:.2f}")

    #card transaction check:
    #for each category spend -> does sum match weekly spend
    #n_transactions fall within defined range 
    print("\nCard Transaction Logic:")
    week_result = draw_weekly_spending(profile, params, rng=rng)
    week_start = pd.Timestamp('2025-01-06')  #Monday, for testing

    for category in CARD_CATEGORIES:
        weekly_amount = week_result['category_amounts'][category]
        transactions = split_into_card_transactions(category, weekly_amount, week_start, rng=rng)

        total_check = sum(t['amount'] for t in transactions)
        print(f"\n{category}: weekly_amount={weekly_amount:.2f}, "
              f"n_transactions={len(transactions)}, sum={total_check:.2f}")
        for t in transactions:
            print(f"  {t['date'].strftime('%Y-%m-%d (%A)')}: £{t['amount']:.2f}")

    #check to see whether inactive category returns empty list and NOT TRY to split
    #£0 across categories
    print("\nEdge case, weekly_amount = 0:")
    empty_result = split_into_card_transactions('alcohol_tobacco', 0.0, week_start, rng=rng)
    print(empty_result)