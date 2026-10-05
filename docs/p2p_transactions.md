## How can we simulate person-to-person transactions ?

So far, we've simulated money going in (income) and money going out (spending) for
each individual account. BUT, real bank transactions also include money being 
transferred to OTHER accounts, which I haven't considered yet. 

### Bank Transaction Behaviour

Taking a step back, what kind of behaviour does bank transactions follow ? It entirely depends on account holder BUT this specific behaviour is predictable. In other words, 
there's a regular inflow (income, investments), predictable outflow because while money 
in your account is intangible, it's incredibly valuable: we have to work to see those 
digits in our bank account. SO, people tend to be incredibly cautious managing their money as they don't want to overspend by a significant margin each month, which is why we see
predictability.  

More importantly though, we know there are common merchants people transfer money to e.g.
supermarkets, restaurents BUT, there's also transferring money to people they KNOW. This 
is where the social network framework comes into play. 

So, these person-to-person transactions will behave similarly to a social network, where
you'd have a local network/social group of people you know so these are the accounts money
is frequently being sent to BUT there's also that long distance connection to an account, which isn't associated with your local network. This could be paying someone for a second-hand car etc. 

This behaviour will be modelled via small world networks, which is built by an algorithm
known as Watts-Strogatz. 

### Wattz-Strogatz Algorithm

This algorithm builds the small world network by:

- Initialising each node (account) to its nearest k neighbour. k refers to the
  number of connections each node has and will be FIXED at the start.
- There's also a rewiring probability, which introduces variability, it looks
  at each node's connection and decides whether to change it to a RANDOM node.
  Rewiring doesn't create or destroy connections, it MOVES them, disconnects
  one node's link to a neighbour, reconnects it elsewhere. So the total number
  of connections in the whole network never changes (always a 1-for-1 swap),
  and the network-wide average (k) stays fixed too, as a CONSEQUENCE of total
  connections being conserved, not a separately enforced rule.
- What actually varies is each INDIVIDUAL node's own connection frequency,
  some nodes gain connections, some lose them, purely from where the
  rewiring happens to land. The network's overall structure (total
  connections, average degree) is fixed once generated, and stays static for
  the whole simulation, consistent with not modelling data drift,
  Watts-Strogatz is a one-time, static generation, not a process where
  connections form or change over time.

### Limitations

While the small world network behaves similarly to person-to-person transactions,
there are a few limitations worth addressing:

- WS algorithm will ONLY give the structure ('who's connected to whom'), BUT it
  doesn't tell us how often two connected accounts transact, or how much. This is 
  a separate layer, which will be decided later.
- Also, if accounts were placed on the ring in a RANDOM order, "nearest
  neighbour" connections would be between arbitrary, unrelated accounts,
  not reflective of real banking data, where connected accounts often
  share similar characteristics e.g. income, region. To address this, I
  source population data from ONS, which will include region + income + 
  age (but this would have less priority compared to the other two) and
  use it to order accounts on the ring so that nearby positions reflect
  real similarity, rather than placing accounts randomly.

## The region data journey: iterating toward something reasonable

This wasn't one clean decision, it was cyclic. Find a dataset, spot a
flaw, find a better one, spot a new gap, repeat. Writing down the
actual path, not just what I ended up using, so I remember WHY I
rejected the earlier attempts, not just what the final answer was.

### Why region, then income, then age

Region first (biggest priority), physical proximity is the strongest
real driver of who you actually transact with, flatmates, local
friends, family nearby. Income second, shared living costs, similar
social spending capacity. Age last, still real but weakest, similar- 
age clustering exists but significant age gap also seen in transaction 
networks (family, partners).

Did this as a straight multi-level sort (region -> income -> age_band),
not some blended weighted score, since the three are on completely
different scales (category, continuous, ordered category) and a sequential 
sort already encodes the priority I wanted without me having to invent
arbitrary weights.

Rewiring probability already covers 'connection outside your normal
circle' (different region, different income), so the sort only needed
to handle LOCAL clustering, not try to model every possible cross-group
connection itself.

### Ethnicity data: considered but decided against it

Found real ONS ethnicity data (Census 2021). Considered using it
alongside region. Decided against it: using a protected characteristic
to shape a FRAUD DETECTION network's connections is a real risk, the
graph structure could end up correlating with ethnicity, and the
eventual detection model could pick that up as a fraud-adjacent signal
without meaning to. Not about the data being wrong, but this being
the wrong place to use it. Used region instead.

### Region dataset went through a couple of attempts before I trusted it

1. First try: pulled region proportions from the same LCFS workbook as
   A26 (Table A33, household counts by region). London came out lower
   than I expected for the UK's most populated region, checked this
   against real-world intuition and it didn't add up. Turns out LCFS is
   a survey, its household counts reflect who got surveyed and weighted
   up, not a real population count. Rejected this for region
   proportions.

2. Second try, the one I actually kept: found a genuinely UK-wide ONS
   file, had real region rows for England's 9 regions AND real country
   rows for Wales, Scotland, NI, with genuine single-year-of-age data
   for all of them, NI included. Real, trustworthy source, checked it
   myself directly.

3. Third, final piece: realised general population isn't the same as
   WORKING population by region, a region's workforce could skew
   differently by age than its general population. Found Nomis (ONS's
   own query tool) could give me real region x age x employment-type
   data directly. Had to reconcile Nomis's 5-year age bands with my own
   6-band scheme, clean for 30-39 to 60+, but 18-21 and 22-29 needed a
   real assumption, settled on 92.5% of the 16-19 band being ages 18-19,
   since most 16-17 year olds are in school/training, not full or
   part-time work.

Scotland excluded throughout, same as everywhere else in this project
(tax scope decision).

### Adding region proximity, not just region identity

Realised a flat region list treats every different-region pair as
equally 'far', London vs South East counted the same as London vs North
East, which isn't right. Found a real UK regional grid layout (geofacet
cartogram, standard for UK regional dashboards) and built a sequence
through it so regions next to each other in my ORDERING are actually
close geographically: Northern Ireland -> North West -> North East ->
Yorkshire and The Humber -> East Midlands -> West Midlands -> Wales ->
South West -> London -> South East -> East -> (wraps back round to NI).

Limitation: a 1D ring can't keep every real 2D adjacency, e.g. West Midlands 
and North West are actually close but end up far apart in my sequence. Not 
something to fix, just what happens collapsing a map into a single line, no 
ordering gets every pair right. Good enough, still much better than alphabetical 
or random.

## The frequency & transaction amount limitation 

Now that I've addressed the random neighbour grouping limitation of the WS
algorithm, there's one more to tackle: frequency & transactin amounts. WS 
will create a small world network, which is great, however, it's not going 
to also generate same-network & cross-network transaction amounts + frequency 
in a way that reflects reality. 

### Frequency

Starting with frequency, I need to model how often transactions occur
between a) accounts in the same local cluster, and b) accounts
connected via rewiring, in different clusters.

Starting off, I can safely assume that connected accounts in the same
cluster will have more transactions than accounts in different
clusters. The solution that came straight to mind was one I'd already
used to solve the spending participation problem, a two-layer design.
Each node (representing an account) has a specific number of edges, so
each edge is assigned a PERSONAL TRANSFER RATE (its average number of
transfers per week), drawn once (Layer 1). Each week, the actual number
of transfers on that edge is drawn around that rate (Layer 2). The rate
stays fixed for the edge, so the week-to-week variation comes from the
weekly draw, not from the rate itself.

I originally used a yes/no draw each week (a Beta-distributed
probability plus a coin flip), but that caps a pair at one transfer per
week, which doesn't fit close contacts who can send money several times
a week. So Layer 1 now draws the personal rate from a Gamma distribution
centred on the tier's target, and Layer 2 draws a weekly count from a
Poisson distribution with that rate.

**Note 1** The personal transfer rate for each edge will be higher for
local connections than distant ones. This is an assumption, not
everyone behaves this way, e.g. someone could have a business partner
who works on the other side of the country and transacts frequently with
them. This wouldn't be captured by the small-world network.


**LOCAL FREQUENCY TRANSACTIONS**

To design local transactions, I thought about how frequently I send
money to a family member or friend, since these are the local accounts.
I realised that instead of assigning one fixed rate (with some
variation) for every local edge, local accounts should be split into
tiers: a CLOSEST tier and a WIDER LOCAL tier. This is more realistic
because, thinking about the saved payees in your bank account, most are
likely to be friends or family, but not all of them get transacted with
at the same frequency. You often have a couple of payees you send money
to frequently, ~1-2x a week (e.g. 'buy me something while you're out'),
while others might only get a transfer once or twice a month, and some
months none at all (e.g. a close friend you meet up with occasionally).
The latter is much harder to model precisely, so I'll use a reasonable
approximation instead.


Small-world networks allow a clean way to separate local accounts by
closeness, since every account has a ring_node_id, I can use the
distance between two connected accounts' node IDs to determine how
close they actually are within the same cluster (region -> income ->
age ordering).

Splitting local edges into CLOSEST and WIDER LOCAL tiers needed an
actual rule for where the line sits, not just 'some accounts are
closer than others.' Since ring distance (abs(node_a - node_b)) already
reflects exactly how close two connected accounts are on the
region -> income -> age ordering, I used it directly as the tiering
signal, testing a couple of threshold values before picking one, rather
than guessing.

Tried threshold=1 (only immediate ring neighbours count as CLOSEST) and
threshold=2 (neighbours up to 2 positions apart count as CLOSEST), on
the 200-account test graph:

threshold=1: Closest=182, Wider local=335, Distant=83
threshold=2: Closest=352, Wider local=165, Distant=83

Went with threshold=1. Reasoning: in reality, the number of people
you're genuinely 'closest' with, in the sense of frequent, casual,
day-to-day money passing, is a small minority of your total
connections, most people have a handful of these relationships, not
the majority of their circle. Threshold=2 makes CLOSEST the majority
of local edges (352 out of 517 same-region edges), which dilutes what
the tier is meant to represent. Threshold=1's split (182 closest vs
335 wider local) matches the real-world pattern much better, a small,
genuinely exclusive closest tier, and a larger, more general local tier
around it.

**TARGET RATES (starting points, not derived from data)**

No dataset gives real P2P transfer frequency by relationship type, so
these are reasoned starting values to be checked once built.

- CLOSEST: mean of 1.5 transfers per week (range 1-2), Gamma shape 10
  so edges are fairly similar to each other.
- WIDER LOCAL: mean of 1.5 transfers per month (about 0.35 per week,
  using 4.33 weeks per month), Gamma shape 2 so edges differ a lot
  (some are nearly dormant, others fairly active).
- DISTANT: mean of 0.01 per week, unchanged.

Gamma shapes are kept at 1 or above, for the same reason as the
alpha-below-1 Dirichlet bug.

**SENDER-RECEIVER DIRECTION LOGIC**

Each transfer picks its sender 50/50, independently. I
originally let the edge tuple order decide, but NetworkX lists every
edge as (lower ring ID, higher ring ID), and the ring is sorted by region
then income. That made the lower-income account in every pair the permanent
sender, a hidden bias that would also have distorted amounts and
mule-detection features like net flow. 50/50 needs no extra assumption
about who pays whom.

**DISTANT FREQUENCY TRANSACTIONS**

These accounts sit in a separate tier from both CLOSEST and WIDER
LOCAL. The probability of sending money to a distant account each week
will be extremely rare, the 'buying a second-hand car from someone'
case.

Before settling on this tier design, I initially considered boosting
the local rate for likely flatmate pairs (similar age, expensive
region), but realised this would DOUBLE-COUNT money already represented
by housing_fuel_power's Direct Debit transaction (each account already
pays its own share of rent/bills directly, so modelling a flatmate P2P
transfer on top would count the same real-world cost twice). Dropped
this idea entirely, going forward with the tier structure instead.

**Volume check (REVISIT once amount is modelled)**

The closest tier is set at a mean of 1.5 transfers per week per edge.
On the 200 account test this gives about 14,700 transfers over the 38 weeks,
so the P2P transaction rows (two per transfer) are about 22% of all rows and
come to roughly 3.9 per account per week. That follows from the spec
(about 1.8 closest connections per account at 1-2 transfers a week each).

Amounts aren't modelled yet, so for now only the row count is known. Once
they are, I'll check total P2P money sent against each account's income and
weekly spend. If accounts end up overspending, the first fix is to lower the
closest mean in TIER_TARGET_MEANS (for example to 1.0, about 18% of rows),
since that is the main driver of volume. I can also lower the wider local
mean or reduce k, but k would mean rechecking the network.

### Do rewired connections mean transfers to accounts outside the local cluster?

Rewiring in Watts-Strogatz picks a genuinely random target anywhere in
the network, it has no awareness of region. So a rewired edge COULD
coincidentally land on someone in the same region as the original
account, which would get it wrongly classified as 'local' (and given a
higher rate) by my tiered system, when it should really be a rare
distant connection.

Checked this directly: on my 200-account test graph, only 4 out of 517
same-region edges were actually mislabelled rewires (same region, but
node positions too far apart to be genuine ring neighbours), 0.77%.
Negligible. Confirmed the same-region proxy is good enough, not worth
building a separate, more invasive rewire-status tracker to fix such a
small effect.

