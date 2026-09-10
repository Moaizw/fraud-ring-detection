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
GENERATED_DIR = os.path.join(REPO_ROOT, 'data', 'generated')
REFERENCE_DIR = os.path.join(REPO_ROOT, 'data', 'reference')

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

def assign_region_and_age(accounts_df: pd.DataFrame, region_age_table: pd.DataFrame, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Each account assigned a region, drawn from region_age_by_archetype_2025.csv,
    using the account's own archetype (full_time/part_time) and its
    already-determined age_band to look up the right distribution.
    """

    if rng is None:
        rng = np.random.default_rng()

    accounts_df = accounts_df.copy()
    regions_assigned = []

    #filter each account by age_band + archetype -> will give us 11 regions
    for (archetype, age_band), group in accounts_df.groupby(['archetype', 'age_band']):
        subset = region_age_table[
            (region_age_table['archetype'] == archetype) &
            (region_age_table['age_band'] == age_band)
            ]

        #normalise joint probability because currently we've taken 11 probabilities,
        #age_band x archetype (11 regions), which DO NOT sum up to 1
        weights = subset['joint_probability'].values
        weights = weights / weights.sum()

        #draw one region from the 11 randomly, HOWEVER the chance of picking
        #region is NOT equal, it depends on the renormalised probability
        drawn = rng.choice(subset['region'].values, size=len(group), replace=True, p=weights)
        regions_assigned.append(pd.Series(drawn, index=group.index)) #GUARANTEES drawn region lands on correct account

    accounts_df['region'] = pd.concat(regions_assigned)

    return accounts_df

def prepare_accounts_for_p2p(accounts_df: pd.DataFrame, region_age_table: pd.DataFrame, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Full chain: Calling the two functions above: first assign_region_and_age,
    which gives us an account assigned to specific region THEN order_accounts_for_ring
    which return a df that is ORDERED (region -> income -> age) for the Watts-Strogatz
    ring. 
    """

    accounts_with_region = assign_region_and_age(accounts_df, region_age_table, rng=rng)
    ordered_accounts = order_accounts_for_ring(accounts_with_region)

    return ordered_accounts

def build_p2p_network(accounts_df: pd.DataFrame, k: int, p: float, seed: int = None) -> nx.Graph:
    """
    Build the P2P transfer network. accounts_df must already be ordered
    (via order_accounts_for_ring) before calling this, since node IDs
    are just row positions (ring_node_id). Note: k param MUST always 
    be even.
    """

    n = len(accounts_df)
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    return G

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    from src.archetypes.full_time import load_age_band_distribution, load_salary_lookup, build_joint_table
    from src.generation.accounts import generate_account_batch

    age_dist = load_age_band_distribution()
    salary_lookup = load_salary_lookup()
    full_time_joint_table = build_joint_table(age_dist, salary_lookup)

    lognormal_r = pd.read_csv(os.path.join(GENERATED_DIR, "lognormal_params_fulltime.csv"))
    gamma_r = pd.read_csv(os.path.join(GENERATED_DIR, "gamma_params_fulltime.csv"))
    weibull_r = pd.read_csv(os.path.join(GENERATED_DIR, "weibull_params_fulltime.csv"))
    gb2_r = pd.read_csv(os.path.join(GENERATED_DIR, "gb2_params_fulltime.csv"))
    comparison_r = pd.read_csv(os.path.join(GENERATED_DIR, "income_comparison_fulltime.csv"))

    spending_table = pd.read_csv(os.path.join(REFERENCE_DIR, "spending_by_income_quintile_single_adult_2025.csv"))
    from src.generation.spending import get_net_quintile_data, load_participation_rates
    quintile_data = get_net_quintile_data(spending_table)
    participation_table = load_participation_rates()

    raw_salary_df = pd.read_csv(
        os.path.join(REFERENCE_DIR, "salary_lookup_age_occupation_fulltime_2025.csv")
    )

    rng = np.random.default_rng(seed=42)

    accounts_df = generate_account_batch(
        n=200, joint_table=full_time_joint_table, comparison_table=comparison_r,
        lognormal_all=lognormal_r, gamma_all=gamma_r, weibull_all=weibull_r, gb2_all=gb2_r,
        spending_table=spending_table, quintile_data=quintile_data,
        participation_table=participation_table, salary_lookup=raw_salary_df,
        archetype='full_time', rng=rng,
    )

    region_age_table = pd.read_csv(os.path.join(REFERENCE_DIR, "region_age_by_archetype_2025.csv"))

    #test case: 30-39/full time
    test = region_age_table[
        (region_age_table['archetype'] == 'full_time') &
        (region_age_table['age_band'] == '30-39')
    ]
    print("\nRows in test subset:", len(test)) #confirm no. of rows = no. of regions (11)

    #compare joint probabilities before renormalising
    #reference for checking region value counts generated
    print(test[['region', 'joint_probability']])

    print("Sum before renormalizing:", test['joint_probability'].sum())

    result = assign_region_and_age(accounts_df, region_age_table, rng=rng)

    print("\nRegion value counts:")
    print(result['region'].value_counts())
    print("\nAny missing regions:", result['region'].isna().sum())
    print("\nSample:")
    print(result[['archetype', 'age_band', 'region']].head(10)) #most likely region based on archetype/age_band renormalised joint prob

    # - BUILDING SMALL-WORLD GRAPH - 

    prepared = prepare_accounts_for_p2p(accounts_df, region_age_table, rng = rng)
    print(prepared[['account_id', 'region', 'archetype', 'net_income', 'age_band', 'ring_node_id']].head(25)) 

    #confirm accounts within same region cluster together in ring_node_id (index) order
    #e.g. Wales occupying ring_node_id 40-58 with no other region accounts seen in this range
    print(prepared.groupby('region')['ring_node_id'].agg(['min', 'max', 'count']))

    G = build_p2p_network(prepared, k=6, p=0.05, seed=42)
    print("\nNodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    degrees = [d for n, d in G.degree()]
    print("Min degree:", min(degrees), "Max degree:", max(degrees), "Mean:", np.mean(degrees))

    #check rewiring is working: majority connections/edges same region with minority cross-region

    cross_region_edges = 0
    same_region_edges = 0

    region_lookup = prepared.set_index('ring_node_id')['region'].to_dict()

    for a, b in G.edges():
        if region_lookup[a] == region_lookup[b]:
            same_region_edges += 1
        else:
            cross_region_edges += 1

    print("Same-region edges:", same_region_edges)
    print("Cross-region edges:", cross_region_edges)
 