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