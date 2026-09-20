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

# - FREQUENCY TRANSACTION MODELLING - 

TRANSFER_RATE_CLOSEST = 0.35
TRANSFER_RATE_WIDER_LOCAL = 0.10
TRANSFER_RATE_DISTANT = 0.01

#controls how tightly personal rates cluster around the tier target, same thing as participation rates
TIER_CONCENTRATIONS = {
    'closest': 50,
    'wider_local': 50,
    'distant': 300, #HIGHER because transfer rate distance significantly smaller so lower conc could lead alpha val < 1
}  

CLOSEST_RING_THRESHOLD = 1


def classify_edge_tier(a: int, b: int, region_lookup: dict) -> str:
    """
    Classify one edge as 'closest', 'wider_local', or 'distant', using
    ring distance and region as the signal (see
    docs/p2p_transactions.md for the reasoning and threshold
    check).
    """
    if region_lookup[a] != region_lookup[b]: #connecting account not in local cluster
        return 'distant'
    elif abs(a - b) <= CLOSEST_RING_THRESHOLD:
        return 'closest'
    else:
        return 'wider_local'


def draw_personal_transfer_rates(G: nx.Graph, region_lookup: dict, rng: np.random.Generator = None) -> dict:
    """
    Layer 1: draw ONE personal transfer probability PER EDGE, once,
    using a Beta distribution centred on that edge's tier target rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    tier_targets = {
        'closest': TRANSFER_RATE_CLOSEST,
        'wider_local': TRANSFER_RATE_WIDER_LOCAL,
        'distant': TRANSFER_RATE_DISTANT,
    }

    edge_rates = {}
    for a, b in G.edges():
        tier = classify_edge_tier(a, b, region_lookup)
        target = tier_targets[tier]

        alpha = target * TIER_CONCENTRATIONS[tier]
        beta = (1 - target) * TIER_CONCENTRATIONS[tier]

        edge_rates[(a, b)] = rng.beta(alpha, beta)

    return edge_rates

def draw_weekly_transfers(edge_rates: dict, week_dates: list, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Layer 2: for every edge/connection and every week, draw one fresh random 
    number and compare against that edge's own personal transfer rate (Layer 1)
    to decide whether a transfer happens that week.
    """
    if rng is None:
        rng = np.random.default_rng()

    transfers = []

    for (a, b), rate in edge_rates.items(): #for each transfer and receiving node get personal transfer rate
        for week_start in week_dates:
            draw = rng.uniform(0, 1) 
            if draw <= rate: #can also be < :doesn't really matter as we're drawing a continuous value 
                transfers.append({
                    'from_node': a,
                    'to_node': b,
                    'week_start': week_start,
                })

    return pd.DataFrame(transfers)

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

    #checking whether most rewired connections land outside of local cluster
    #need to know before I model P2P transaction frequencies 
    same_region_but_could_be_rewired = 0
    for a, b in G.edges():
        region_a = region_lookup[a]
        region_b = region_lookup[b]
        #a true local connection is same region AND close ring positions
        #(within roughly k/2 of each other), a rewired-but-same-region
        #connection is same region but far apart in ring position
        if region_a == region_b and abs(a - b) > 6:
            same_region_but_could_be_rewired += 1

    print(same_region_but_could_be_rewired)

    #need to check how many edges fall into the proposed tiers
    #before moving onto coding

    region_lookup = prepared.set_index('ring_node_id')['region'].to_dict()

    closest = 0
    wider_local = 0
    distant = 0

    CLOSEST_THRESHOLD = 1  #ring distance considered closest

    for a, b in G.edges():
        if region_lookup[a] != region_lookup[b]:
            distant += 1
        elif abs(a - b) <= CLOSEST_THRESHOLD:
            closest += 1
        else:
            wider_local += 1

    print("Closest:", closest)
    print("Wider local:", wider_local)
    print("Distant:", distant)

    rng = np.random.default_rng(seed=42)
    edge_rates = draw_personal_transfer_rates(G, region_lookup, rng=rng)

    #group edges by tier for inspection
    tier_rates = {'closest': [], 'wider_local': [], 'distant': []}
    for (a, b), rate in edge_rates.items():
        tier = classify_edge_tier(a, b, region_lookup)
        tier_rates[tier].append(rate)

    for tier, rates in tier_rates.items():
        rates = np.array(rates)
        print(f"\n{tier}: n={len(rates)}")
        print(f"  min={rates.min():.4f}, max={rates.max():.4f}, mean={rates.mean():.4f}")

    #confirm target rate calibration: mean of drawn rates should be
    #close to each tier's target
    print("\nTargets: closest=0.35, wider_local=0.10, distant=0.01")

    #check layer 2 random weekly transfers for each TIER average 
    #to TRUE TRANSFER RATE
    #i.e. how many of the 38 weeks actually had a transfer for each CONNECTION across each TIER?
    #once no. of transfers found / 38 weeks, avg rate taken ACROSS ALL TIERS
    #avg rate compared to TRUE rate 
    from src.generation.timeline import generate_week_start_dates

    week_dates = generate_week_start_dates()
    weekly_transfers = draw_weekly_transfers(edge_rates, week_dates, rng=rng)

    print("Total transfers generated:", len(weekly_transfers))

    weekly_transfers['edge'] = list(zip(weekly_transfers['from_node'], weekly_transfers['to_node']))
    transfer_counts = weekly_transfers['edge'].value_counts()

    for tier_name, tier_target in [('closest', 0.35), ('wider_local', 0.10), ('distant', 0.01)]:
        tier_edges = [(a, b) for (a, b) in edge_rates if classify_edge_tier(a, b, region_lookup) == tier_name]
        observed_freqs = [transfer_counts.get((a, b), 0) / len(week_dates) for (a, b) in tier_edges]
        print(f"{tier_name}: mean observed freq = {np.mean(observed_freqs):.4f} (target: {tier_target})")