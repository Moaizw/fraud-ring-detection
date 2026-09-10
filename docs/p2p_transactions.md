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