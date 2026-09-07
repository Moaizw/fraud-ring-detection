"""
Builds the actual 38 week timeline for each simulated account. Simply calls 
the already-built spending mechanisms (Layer 2, participation, card/direct 
debit splitting, monthly income variation) to produce a genuine dated transaction 
table per account. See docs/simulation_timeline_design.md for the full design.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from src.generation.tax import gross_to_net

#spending logic functions 
from src.generation.spending import (
    CATEGORY_COLS,
    CARD_CATEGORIES,
    DIRECT_DEBIT_CATEGORIES,
    EXCLUDED_FROM_TRANSACTIONS,
    load_spending_table,
    get_net_quintile_data,
    load_participation_rates,
    interpolate_parameters,
    draw_personal_profile,
    draw_personal_participation_rates,
    draw_weekly_spending_with_participation,
    split_into_card_transactions,
    generate_monthly_direct_debit_amounts,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
REFERENCE_DIR = os.path.join(REPO_ROOT, "data", "reference")
GENERATED_DIR = os.path.join(REPO_ROOT, "data", "generated")

TRAIN_TEST_BOUNDARY = pd.Timestamp('2025-06-27')

#monthly income variation, see docs/simulation_timeline_design.md
INCOME_HIGH_VARIATION_OCCUPATIONS = [
    'Sales and customer service occupations',
    'Elementary occupations',
    'Process plant and machine operatives',
    'Skilled trades occupations',
]
INCOME_SPREAD_HIGH = 0.12
INCOME_SPREAD_LOW = 0.04

def generate_week_start_dates(start_date: str = '2025-01-06', n_weeks: int = 38) -> list:
    """
    Generate n_weeks Monday-start dates, beginning at start_date. These
    anchor every weekly Layer 2 draw.
    """
    start = pd.Timestamp(start_date)
    return [start + pd.Timedelta(weeks=i) for i in range(n_weeks)]

def tag_train_test(date: pd.Timestamp) -> str:
    """
    Tag a date as 'train' or 'test' based on TRAIN_TEST_BOUNDARY.
    """
    return 'train' if date <= TRAIN_TEST_BOUNDARY else 'test'

def generate_account_transactions(account: dict, week_dates: list, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Generate the full 38-week transaction history for ONE account.
    Returns a DataFrame: account_id, date, category, amount,
    transaction_type, tag.
    """
    if rng is None:
        rng = np.random.default_rng()

    transactions = []
    seen_months = set()

    personal_profile = account['personal_profile']
    spending_params = account['spending_params']
    personal_rates = account['personal_rates']
    occupation = account['occupation']
    net_income = account['net_income']

    monthly_net_income = net_income / 12

    #occupation-based income wobble tier
    income_spread = (
        INCOME_SPREAD_HIGH if occupation in INCOME_HIGH_VARIATION_OCCUPATIONS
        else INCOME_SPREAD_LOW
    )

    for week_start in week_dates:
        month_key = (week_start.year, week_start.month)
        is_new_month = month_key not in seen_months

        #SPENDING
        week_result = draw_weekly_spending_with_participation(
            personal_profile, spending_params, personal_rates, rng=rng
        )

        for category in CARD_CATEGORIES:
            amount = week_result['category_amounts'][category]
            if amount > 0:
                for t in split_into_card_transactions(category, amount, week_start, rng=rng):
                    transactions.append({
                        'account_id': account['account_id'],
                        'date': t['date'],
                        'category': t['category'],
                        'amount': t['amount'],
                        'transaction_type': 'outbound',
                    })

        if is_new_month:
            monthly_dd = generate_monthly_direct_debit_amounts(week_result, personal_profile)
            for category, amount in monthly_dd.items():
                transactions.append({
                    'account_id': account['account_id'],
                    'date': week_start,
                    'category': category,
                    'amount': amount,
                    'transaction_type': 'outbound',
                })

            #INCOME
            monthly_income = stats.lognorm.rvs(s=income_spread, scale=monthly_net_income, random_state=rng)
            transactions.append({
                'account_id': account['account_id'],
                'date': week_start,
                'category': 'income',
                'amount': monthly_income,
                'transaction_type': 'inbound',
            })

            seen_months.add(month_key)

    df = pd.DataFrame(transactions)
    df['tag'] = df['date'].apply(tag_train_test)
    return df

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    # - FULL TIME CHECK - 
    from src.archetypes.full_time import load_age_band_distribution, load_salary_lookup, build_joint_table
    from src.generation.accounts import generate_single_account

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
    participation_table = load_participation_rates()

    rng = np.random.default_rng(seed=42)

    test_account = generate_single_account(
        full_time_joint_table, comparison_r, lognormal_r, gamma_r, weibull_r, gb2_r,
        spending_table, quintile_data, participation_table,
        archetype='full_time', rng=rng,
    )
    print("Test account:", test_account['account_id'], test_account['occupation'], test_account['net_income'])

    weeks = generate_week_start_dates()
    transactions_df = generate_account_transactions(test_account, weeks, rng=rng)

    print("\nShape:", transactions_df.shape)
    print(transactions_df.head(20))
    print("\nTag counts:")
    print(transactions_df['tag'].value_counts())
    print("\nBy category:")
    print(transactions_df.groupby('category')['amount'].agg(['count', 'sum']))

    # - PART TIME CHECK - 

    from src.archetypes.part_time import load_age_band_distribution as pt_load_age, load_salary_lookup as pt_load_salary, build_joint_table as pt_build_joint

    age_dist_pt = pt_load_age()
    salary_lookup_pt = pt_load_salary()
    part_time_joint_table = pt_build_joint(age_dist_pt, salary_lookup_pt)

    lognormal_pt = pd.read_csv(os.path.join(GENERATED_DIR, "lognormal_params_parttime.csv"))
    gamma_pt = pd.read_csv(os.path.join(GENERATED_DIR, "gamma_params_parttime.csv"))
    weibull_pt = pd.read_csv(os.path.join(GENERATED_DIR, "weibull_params_parttime.csv"))
    gb2_pt = pd.read_csv(os.path.join(GENERATED_DIR, "gb2_params_parttime.csv"))
    comparison_pt = pd.read_csv(os.path.join(GENERATED_DIR, "income_comparison_parttime.csv"))

    test_account_pt = generate_single_account(
        part_time_joint_table, comparison_pt, lognormal_pt, gamma_pt, weibull_pt, gb2_pt,
        spending_table, quintile_data, participation_table,
        archetype='part_time', rng=rng,
    )
    print("Test account (PT):", test_account_pt['account_id'], test_account_pt['occupation'], test_account_pt['net_income'])

    transactions_df_pt = generate_account_transactions(test_account_pt, weeks, rng=rng)

    print("\nShape:", transactions_df_pt.shape)
    print("\nTag counts:")
    print(transactions_df_pt['tag'].value_counts())
    print("\nBy category:")
    print(transactions_df_pt.groupby('category')['amount'].agg(['count', 'sum']))