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
from src.generation.spending import WEEKS_PER_MONTH


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


def attach_account_ids_and_dates(transfers: pd.DataFrame, accounts_ordered: pd.DataFrame,
                                 rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Translate ring node IDs into account IDs and give each transfer an exact
    date: week_start plus a uniformly random day (0-6). Uniform because
    not considering weekend/weekday skew for P2P transfers.
    accounts_ordered must be the output of prepare_accounts_for_p2p.
    """
    if rng is None:
        rng = np.random.default_rng()

    #ring_node_id must be exactly 0..n-1 so it can index a plain array
    assert (np.sort(accounts_ordered['ring_node_id'].to_numpy()) == np.arange(len(accounts_ordered))).all()
    node_to_account = accounts_ordered.sort_values('ring_node_id')['account_id'].to_numpy()

    out = transfers.copy()
    out['from_account_id'] = node_to_account[out['from_node'].to_numpy(dtype=int)]
    out['to_account_id'] = node_to_account[out['to_node'].to_numpy(dtype=int)]

    day_offset = rng.integers(0, 7, size=len(out))
    out['date'] = out['week_start'] + pd.to_timedelta(day_offset, unit='D')

    return out

def transfers_to_transaction_rows(transfers: pd.DataFrame) -> pd.DataFrame:
    """
    Each transfer becomes TWO rows: outbound for the sender, inbound for the
    receiver. Same schema as the card/Direct Debit/income rows (account_id,
    date, category, amount, transaction_type, tag) plus two new columns:
    counterparty_account_id (the other side, needed later to build the fraud
    graph) and transfer_id (shared by the two rows of one transfer).
    amount is NaN until the amount step exists.
    """
    #imported here just in case timeline.py needs to later import transfers_to_transaction_rows
    from src.generation.timeline import TRAIN_TEST_BOUNDARY

    n = len(transfers)
    transfer_id = np.arange(n)
    amount = transfers['amount'].to_numpy() if 'amount' in transfers.columns else np.full(n, np.nan)
    dates = transfers['date'].to_numpy()
    tag = np.where(transfers['date'].to_numpy() <= np.datetime64(TRAIN_TEST_BOUNDARY), 'train', 'test')

    def side(account_col, counterparty_col, direction):
        return pd.DataFrame({
            'account_id': transfers[account_col].to_numpy(),
            'date': dates, #both SENDER & RECEIVER will share the same date
            'category': 'p2p_transfer',
            'amount': amount,
            'transaction_type': direction,
            'tag': tag, #train/test rows
            'counterparty_account_id': transfers[counterparty_col].to_numpy(), #account on the other side e.g. for sender's row, it would be receiver
            'transfer_id': transfer_id, #number shared between two rows of the same transfer -> can pair them later
        })

    return pd.concat([side('from_account_id', 'to_account_id', 'outbound'),
                      side('to_account_id', 'from_account_id', 'inbound')], ignore_index=True)


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

CLOSEST_RING_THRESHOLD = 1

#mean number of transfers per week per edge 
TIER_TARGET_MEANS = {
    'closest': 1.5,                          #->1-2 per week
    'wider_local': 1.5 / WEEKS_PER_MONTH,    #->1-2 per month, about 0.35 per week
    'distant': 0.01,                         
}

#Gamma shape per tier: spread of personal rates across edges is 1/sqrt(shape).
#keep every shape >= 1 (same reasoning as the alpha-below-1 Dirichlet bug).
TIER_GAMMA_SHAPES = {
    'closest': 15,       #edges fairly similar to each other
    'wider_local': 2,    #edges differ a lot, some nearly dormant
    'distant': 3,        #matches the spread of the old distant-tier Beta
}


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
    Layer 1: draw ONE personal transfer rate PER EDGE, once (average number
    of transfers per week), from a Gamma distribution centred on that edge's
    tier target mean.
    """
    if rng is None:
        rng = np.random.default_rng()

    edge_rates = {}
    for a, b in G.edges():
        tier = classify_edge_tier(a, b, region_lookup)
        shape = TIER_GAMMA_SHAPES[tier]
        edge_rates[(a, b)] = rng.gamma(shape=shape, scale=TIER_TARGET_MEANS[tier] / shape)

    return edge_rates


def draw_weekly_transfers(edge_rates: dict, week_dates: list, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Layer 2: for every edge and every week, draw a transfer COUNT from a
    Poisson distribution with that edge's personal rate (Layer 1). One row
    per transfer, so a pair can appear several times in the same week.
    """

    if rng is None:
        rng = np.random.default_rng()

    edges = list(edge_rates.keys())
    if len(edges) == 0:
        return pd.DataFrame(columns=['node_a', 'node_b', 'from_node', 'to_node', 'week_start'])

    rates = np.array([edge_rates[e] for e in edges])
    counts = rng.poisson(rates[:, None], size=(len(edges), len(week_dates)))  #edges x weeks

    edge_idx, week_idx = np.nonzero(counts)
    repeats = counts[edge_idx, week_idx]
    edge_idx = np.repeat(edge_idx, repeats)
    week_idx = np.repeat(week_idx, repeats)

    edge_arr = np.array(edges)
    node_a = edge_arr[edge_idx, 0]
    node_b = edge_arr[edge_idx, 1]

    #direction: each transfer independently picks its sender, 50/50
    swap = rng.random(len(edge_idx)) < 0.5

    return pd.DataFrame({
        'node_a': node_a,
        'node_b': node_b,
        'from_node': np.where(swap, node_b, node_a),
        'to_node': np.where(swap, node_a, node_b),
        'week_start': pd.DatetimeIndex(week_dates)[week_idx],
    })


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

    k = 6
    G = build_p2p_network(prepared, k=k, p=0.05, seed=42)
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
        #a true local connection is same region AND close ring positions
        #(within k of each other), a rewired-but-same-region
        #connection is same region but far apart in ring position
        if region_lookup[a] == region_lookup[b] and abs(a - b) > k:
            same_region_but_could_be_rewired += 1

    print("Same-region but far apart on ring (likely rewired):", same_region_but_could_be_rewired)

    #how many edges fall into each tier ?
    tier_counts = {'closest': 0, 'wider_local': 0, 'distant': 0}
    for a, b in G.edges():
        tier_counts[classify_edge_tier(a, b, region_lookup)] += 1

    print("Closest:", tier_counts['closest'])
    print("Wider local:", tier_counts['wider_local'])
    print("Distant:", tier_counts['distant'])

    # - FREQUENCY: LAYER 1 - 

    rng = np.random.default_rng(seed=42)
    edge_rates = draw_personal_transfer_rates(G, region_lookup, rng=rng)

    #group edges by tier for inspection
    tier_rates = {'closest': [], 'wider_local': [], 'distant': []}
    for (a, b), rate in edge_rates.items():
        tier_rates[classify_edge_tier(a, b, region_lookup)].append(rate)

    #confirm target calibration: mean of drawn personal rates should be
    #close to each tier's target mean (transfers per week)
    for tier, rates in tier_rates.items():
        rates = np.array(rates)
        print(f"\n{tier}: n={len(rates)}")
        print(f"  min={rates.min():.4f}, max={rates.max():.4f}, mean={rates.mean():.4f} (target {TIER_TARGET_MEANS[tier]:.4f})")

    # - FREQUENCY: LAYER 2 - 

    #weekly transfer COUNTS per edge; check each tier's observed mean 
    #transfers/week against its target
    from src.generation.timeline import generate_week_start_dates

    week_dates = generate_week_start_dates()
    weekly_transfers = draw_weekly_transfers(edge_rates, week_dates, rng=rng)
    print("\nTotal transfers generated:", len(weekly_transfers))

    #calibration works on PAIRS (node_a/node_b), not direction
    weekly_transfers['edge'] = list(zip(weekly_transfers['node_a'], weekly_transfers['node_b']))
    weekly_transfers['tier'] = [classify_edge_tier(a, b, region_lookup)
                            for a, b in zip(weekly_transfers['node_a'], weekly_transfers['node_b'])]
    transfer_counts = weekly_transfers['edge'].value_counts()

    for tier_name, tier_target in TIER_TARGET_MEANS.items():
        tier_edges = [(a, b) for (a, b) in edge_rates if classify_edge_tier(a, b, region_lookup) == tier_name]
        observed = [transfer_counts.get((a, b), 0) / len(week_dates) for (a, b) in tier_edges]
        print(f"{tier_name}: mean transfers/week = {np.mean(observed):.4f} (target {tier_target:.4f})")

    per_pair_week = weekly_transfers.groupby(['node_a', 'node_b', 'week_start']).size()
    print("Max transfers by one pair in one week:", per_pair_week.max())

    n_closest = sum(1 for (a, b) in edge_rates if classify_edge_tier(a, b, region_lookup) == 'closest')
    active = weekly_transfers[weekly_transfers['tier'] == 'closest'].drop_duplicates(
        ['node_a', 'node_b', 'week_start']).shape[0]
    shape, mean = TIER_GAMMA_SHAPES['closest'], TIER_TARGET_MEANS['closest']
    expected_zero = (shape / (shape + mean)) ** shape
    print(f"closest weeks with no transfer: {1 - active / (n_closest * len(week_dates)):.3f} (expect about {expected_zero:.3f})")

    # - DIRECTION CHECKS -
    print("\nShare sent by the lower-ID node:",
    round((weekly_transfers['from_node'] == weekly_transfers['node_a']).mean(), 3), "(expect about 0.5)")

    income_lookup = prepared.set_index('ring_node_id')['net_income'].to_dict()
    same_region_mask = [region_lookup[a] == region_lookup[b]
                    for a, b in zip(weekly_transfers['from_node'], weekly_transfers['to_node'])]
    same_region = weekly_transfers[same_region_mask]
    poorer_sends = (same_region['from_node'].map(income_lookup) < same_region['to_node'].map(income_lookup)).mean()
    print("Same-region transfers where the lower-income account sends:", round(poorer_sends, 3), "(expect about 0.5)")

    #net flow per node: without the fix it falls steadily with ring ID, node 0 only sends and node 199 only receives
    ids = range(len(prepared))
    sent = weekly_transfers['from_node'].value_counts().reindex(ids, fill_value=0)
    received = weekly_transfers['to_node'].value_counts().reindex(ids, fill_value=0)
    net_flow = sent - received
    print("Correlation between ring ID and net flow:", round(np.corrcoef(list(ids), net_flow.values)[0, 1], 3), "(expect near 0)")

    # - INTEGRATION: MAPPING NODE ID TO ACCOUNT ID + TRANSFERS ADDED TO TRANSACTION TABLE -
    transfers = attach_account_ids_and_dates(weekly_transfers, prepared, rng=rng)
    p2p_rows = transfers_to_transaction_rows(transfers)

    print("\nP2P rows:", len(p2p_rows), "(expect 2 x", len(transfers), ")")
    print(p2p_rows.head(6))

    sides = p2p_rows.groupby(['transfer_id', 'transaction_type']).size().unstack()
    print("Each transfer has exactly one outbound and one inbound row:",
          bool((sides['outbound'] == 1).all() and (sides['inbound'] == 1).all()))
    print("Sender != receiver everywhere:", bool((p2p_rows['account_id'] != p2p_rows['counterparty_account_id']).all()))
    print("All account IDs known:", bool(p2p_rows['account_id'].isin(prepared['account_id']).all()))
    print("Date range:", p2p_rows['date'].min().date(), "to", p2p_rows['date'].max().date())
    print("Train share:", round((p2p_rows['tag'] == 'train').mean(), 3))

    #merge with the existing per-account transactions
    from src.generation.timeline import generate_account_transactions
    account_tx = pd.concat(
        [generate_account_transactions(acc, week_dates, rng=rng) for acc in prepared.to_dict('records')],
        ignore_index=True,
    )
    transactions = (pd.concat([account_tx, p2p_rows], ignore_index=True)
                    .sort_values(['date', 'account_id']).reset_index(drop=True))
    print("\nRows by category:")
    print(transactions.groupby('category').size())