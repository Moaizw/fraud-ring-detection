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

### Amount

This should be relatively straightforward to model but it will be very
assumption-heavy. The two assumptions I'm making here are:

1) How much the spender normally spends in a week: Someone who spends 
   more will tend to send bigger amounts. The good thing is that I 
   already have this number from the spending model.

2) How close the pair is: I'm going to also assume that closer accounts
   tend to send smaller amounts compared to more distant accounts, where
   transactions are infrequent. 

For example, let's say an account spends £450 (generated from our spending model),
I can use that figure and use three different percentages for the three tiers to
model how MUCH that person will send to an account belonging to one of the tiers 
in a week. For example, let's say an account will send 1.5% of total spend in a week
to a 'closest' account -> ~£7, 5% total spend to 'close' account -> ~£23 and 15% total
spend to 'distant' account -> ~£68. 

These %'s are starting guesses so could be changed. I will implement the same 2-layer
approach to add randomness: 

- **Layer 1** -> every transfer amount is a random draw around TOTAL SPEND / WEEK of account 
- **Layer 2** -> every pair would get its own typical amount, which would be drawn once AND 
  its transfers would then VARY around that typical amount. The reason for this is that real
  repeat payments between the same people tend to cluster such as the usual £10 for coffee.

**RESULTS ANALYSIS** 

| Check | Expected | Actual (200-account test, seed 42) |
|---|---|---|
| Missing amounts, all positive | none, True | 0, True |
| Median, closest | about 1.5% of £422 = £6.32 | £6.49 |
| Median, wider local | about 5% = £21.08 | £20.90 |
| Median, distant | about 15% = £63.2 | £86.30 |
| Within-pair SD (Layer 2) | 0.5 | 0.50 |
| Between-pair SD (Layer 1) | about 0.60 | 0.557 |
| P2P sent as share of own weekly spend | about 0.068 | 0.065 |

In short: Everything else matched what I expected. Only the distant tier 
was off, by about £23. It is probably just luck from a small sample, but I
haven't confirmed that yet.

**Why it is probably luck**

- The distant tier has only 31 transfers, so its median is calculated
  from very few numbers. The closest tier has about 10,400, so its median
  hardly moves between runs.
- Think of judging whether a coin is fair from 31 flips versus 10,000
  flips. With 31, being a bit off is normal.
- Standard error = how far a median typically drifts from its true value
  from one run to the next, purely because of the random draws. A small
  sample has a big standard error and a large sample has a small one.

ROUGH MATHS CHECK:

Amount -> lognormal (normally distributed on the log scale). Two layers
of randomness are added to that log scale: per-pair multiplier
(spread = 0.6), per-transfer noise (spread = 0.5).

Combined -> sqrt(0.6^2 + 0.5^2) = sqrt(0.61) ≈ 0.78, SO a typical
distant transfer should land within exp(0.78) ≈ 2.2 times the median in
either direction.

For normally distributed values, the median of n draws has a standard
error of about 1.25 * spread / sqrt(n):

1.25 * 0.78 / sqrt(31) ≈ 0.175

On the log scale, 0.175 is a multiplicative standard error of about
exp(0.175) ≈ 1.19, so ~19% on the median. In plain terms: a median built
from 31 transfers can easily be about 20% off.

Expected distant median is 15% of weekly spend (£421.57), which is
£63.20. My result was £86.30.

**How big is the gap, measured in standard errors?**

The same gap can be described in two ways:

- As a ratio: 86.30 / 63.20 ≈ 1.37, so my result is about 1.4x the
  expected value.
- In standard errors. One standard error is about 1.19x (the ~19% from
  above). The 1.4x gap is therefore more than one standard error.

Ratios multiply rather than add, so to count standard errors I work on
the log scale, where they add up:

gap on the log scale = ln(1.37) ≈ 0.31
one standard error on the log scale = ln(1.19) ≈ 0.175
0.31 / 0.175 ≈ 1.8

So my result is about 1.8 standard errors above expectation. Checking it
the other way: 1.19 x 1.19 ≈ 1.42, so two standard errors in the same
direction would give about 1.4x, and mine is a bit less than that.

How unusual is that? A result 1.8 standard errors out, in either
direction, happens by chance about 7% of the time (roughly 1 time in 14).
That makes it slightly unusual but not alarming, so I check whether it
is luck rather than assuming it is a bug.

Note: the 0.175 treats the 31 transfers as 31 independent draws, but
transfers between the same pair share one Layer 1 multiplier. Distant
pairs transact so rarely (~0.4 transfers per pair over 38 weeks) that I
expect the real standard error to be only slightly bigger, around 0.19.
With 0.19 the gap is 0.31 / 0.19 ≈ 1.6 standard errors, which happens
by chance roughly 1 time in 10. It makes the gap look slightly less
surprising, not more.

**Two possible explanations**

1. Luck. The random multipliers and noise just happened to come out
   high on a small sample.
2. The senders. Each amount is a % of the SENDER's weekly spend. The
   expected £63.20 assumes an average sender (£421.57 a week). If the
   few distant senders happen to be bigger spenders, the median rises
   for that reason alone.

**Checks to tell them apart**

1. Sender spend by tier. Compare the median weekly spend of the distant
   senders with the other tiers. If distant senders spend well above
   £421, that explains part of the gap.

2. Number of different distant pairs. Transfers between the same pair
   are alike (they share a multiplier), so 31 transfers carry slightly
   less information than 31 independent ones. This check shows how
   much.

3. Redraw test (the main one). Keep the same transfers and senders,
   but redraw the random multipliers and noise 300 times, and look at
   the spread of the distant:closest median ratio. Expected ratio is 10,
   mine was 13.3. How to read it:
   - 13.3 sits inside a spread centred near 10: it is luck.
   - The whole spread is centred near 13: the particular senders explain
     it.
   - 13.3 sits well outside a spread centred near 10: something is wrong
     with the mechanism, investigate.

4. Bigger run. At 50,000 accounts the distant tier should have roughly
   7,700 transfers, plenty to be stable. If the ratio is still about 13
   there, it is not luck.

**Decision**

...