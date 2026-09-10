"""
Builds P2P transaction network network across all simulated accounts using
Watts-Strogatz small-world network. Accounts ordered in the ring by region
(considering the proximity of regions) + income + age so local connections 
show real similarity. This addresses the random neighbour grouping of the WS 
algorithm. 
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
REFERENCE_ROOT = os.path.join(REPO_ROOT, 'data', 'reference')

#real UK region adjacency sequence, built from a geofacet grid layout,
#using CLAUDE: confirmed this sequence checks out too
#consecutive regions here are genuinely close, Scotland excluded
REGION_ORDER = [
    'Northern Ireland', 'North West', 'North East', 'Yorkshire and The Humber',
    'East Midlands', 'West Midlands', 'Wales', 'South West', 'London',
    'South East', 'East',
]

AGE_BAND_ORDER = {'18-21': 0, '22-29': 1, '30-39': 2, '40-49': 3, '50-59': 4, '60+': 5}

def order_accounts_for_ring(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orders accounts by region (using REGION_ORDER's geographic sequence,
    NOT alphabetical), then net_income, then age_band. Placing accounts
    on the Watts-Strogatz ring in this order means nearby ring positions
    share similar characteristics, which is frequently seen in real world 
    transaction rings.
    """
    accounts_df = accounts_df.copy()

    region_rank = {region: i for i, region in enumerate(REGION_ORDER)} #rnk regions

    #maps the two categorical columns to numeric but also keeps same ordering
    #required because sorting with categorical columns makes rank-based ordering REDUNDANT
    accounts_df['region_order_num'] = accounts_df['region'].map(region_rank) 
    accounts_df['age_order_num'] = accounts_df['age_band'].map(AGE_BAND_ORDER)

    sorted_df = accounts_df.sort_values(
        by=['region_order_num', 'net_income', 'age_order_num']
    ).reset_index(drop=True)

    sorted_df = sorted_df.drop(columns=['region_order_num', 'age_order_num'])
    sorted_df['ring_node_id'] = sorted_df.index

    return sorted_df

