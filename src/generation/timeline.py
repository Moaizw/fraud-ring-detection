"""
Builds the actual 38 week timeline for each simulated account. Simply calls 
the already-built spending mechanisms (Layer 2, participation, card/direct 
debit splitting, monthly income variation) to produce a genuine dated transaction 
table per account. See docs/simulation_timeline_design.md for the full design.
"""

import os
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    weeks = generate_week_start_dates()
    print("First week:", weeks[0], weeks[0].day_name())
    print("Last week:", weeks[-1], weeks[-1].day_name())
    print("Total weeks:", len(weeks))

    tags = [tag_train_test(w) for w in weeks]
    print(list(zip([w.date() for w in weeks], tags)))


