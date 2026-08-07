# Fraud ring detection project — my notes

## What this project actually is

I'm building a fraud detection project, but instead of the usual "classify this transaction as fraud or not fraud" approach, I'm treating fraud as a network problem. The idea: individual transactions can look totally normal on their own, but when you map out how accounts connect to each other, patterns show up that a normal classifier would never catch. Think mule networks, where money gets passed between accounts to hide where it came from.

So the core building block here is a graph, not a spreadsheet. Accounts are nodes, transactions and shared stuff (same device, same IP, same payee) are edges.

## Why I picked this

Most portfolio fraud projects are "XGBoost on a Kaggle credit card dataset." Everyone's done that one. It also doesn't reflect how fraud actually works in a lot of cases, since one weird transaction rarely proves anything on its own.

What I want this project to show:
- I can do something more technically interesting than a standard classifier (actual graph analysis, community detection, maybe a GNN if I have time)
- I actually understand how fraud shows up in real banking data, not just how to fit a model
- I know the metrics real fraud/risk teams use, not just accuracy/F1

## The real-world thing that got me thinking about this: Monzo's £21m fine

In July 2025 the FCA fined Monzo about £21m for anti-money-laundering failings. The details are genuinely useful context for this project:

- Their address checks were so weak people opened accounts with "10 Downing Street" and "Buckingham Palace" as their address
- They couldn't spot when the same person opened multiple accounts, including accounts that had already been shut down for fraud
- They didn't even join CIFAS (the shared fraud database banks use) until 2020
- All this happened while they scaled from 590k to 7.4m customers, so growth just outran their controls

Monzo's own help page explains money muling really simply: someone gets recruited (often through social media), agrees to receive a payment into their account, quickly forwards it on to another account, and keeps a cut. No single transfer in that chain looks suspicious by itself. It's only obvious once you see it as a chain/network. That's basically the exact problem this project is trying to solve.

## Not all fraud is the same, and graphs don't catch all of it

This is important to keep in mind so I don't oversell what graphs can do:

| Fraud type | What's happening | Do graphs help? |
|---|---|---|
| Money muling / layering | Money passed through a chain of accounts to hide its origin | Yes, this is the ideal case, the whole signal is in the connections |
| Synthetic identity fraud | Fake identity built from bits of real + fake info | Yes, often reuses the same phone/device/address across "different" people |
| Bust-out fraud | Real person opens credit intending to max it and vanish | Partially, sometimes coordinated with others via shared address/device |
| Account takeover | Criminal steals someone's real login and drains it | Not really, the signal is that one account behaving weirdly, not its connections |
| APP scams (push payment scams) | Victim is tricked into sending money themselves | Half and half, the victim's side looks clean, but the receiving mule account is graph territory |
| Stolen card fraud | Stolen card used online | Not really, this is about velocity/geography/merchant checks on one card |

So I'm scoping this project specifically to muling and ring-style fraud, since that's where graphs genuinely add value. I want to be upfront about that rather than pretending graphs solve everything.

Also worth remembering: the type of fraud determines what data you need. Muling needs transaction + shared-identifier data. Synthetic identity needs identity attributes as graph nodes. Account takeover needs session/login behaviour, which barely needs a graph at all.

## How I'm actually going to build it

**Step 1: Make the data.**
Real fraud-ring data doesn't exist publicly (it's all private bank data), so I need to simulate it myself:
- Generate a bunch of "normal" accounts with realistic behaviour (salary in, bills out, normal spending)
- Inject some fraud rings, small clusters of accounts with tight, closed-loop transactions and shared devices/IPs
- Also inject a hierarchical case, a few small rings that all feed into one "collector" account, so I can test detecting structure at more than one level
- Important: don't make the fraud too obvious, or the project looks fake/trivial

**Step 2: Build the graph.**
Nodes = accounts. Edges = transactions, or shared device/IP/payee. Watch out for edges that are too common (like everyone using the same popular merchant), since that just creates a meaningless blob instead of a useful ring.

**Step 3: Engineer features.**
Two types:
- Normal transaction stuff: amount, frequency, time of day, how much this differs from the account's own usual pattern
- Graph stuff: degree (how connected), clustering coefficient (do my neighbours know each other), betweenness centrality, distance to nearest known-fraud account, which community it's in

Big thing to watch for: only use info that would have been available at that point in time. If I build the graph using future data, the model's basically cheating.

**Step 4: Find the rings, then find the rings of rings.**
Run Louvain community detection once, this finds the tight little clusters (the individual rings). Then collapse each cluster into a single "super node" and run community detection again on that smaller graph, this is what surfaces bigger structures, like three separate rings all connected to one collector account.

**Step 5: Model it.**
- Start simple: logistic regression on just transaction features, as a baseline
- Then XGBoost/LightGBM combining transaction + graph features, this is the main model
- Stretch goal if I have time: a Graph Neural Network (GraphSAGE via PyTorch Geometric) that learns straight from the graph structure instead of hand-built features

**Step 6: Check if it actually worked.**
- Precision/recall and PR-AUC (not ROC-AUC, misleading with imbalanced data)
- Ring-level check: did I actually catch the rings I injected? (compare detected communities to my known ground truth)
- Hierarchy check: did the second detection pass correctly find the collector structure?

**Step 7: Add a GenAI layer, natural-language summaries for analysts.**
Once the pipeline flags a ring (or a ring-of-rings), a real fraud analyst doesn't want to stare at a table of account IDs, feature values, and a community ID number, they need to quickly understand *why* something was flagged so they can decide what to do. So this step takes the actual evidence my pipeline already computed for a flagged case (which accounts are involved, what they share, i.e. device/IP, the transaction pattern between them, how fast money moved, whether it's a single ring or a hierarchy feeding a collector) and feeds that structured evidence into an LLM to generate a short, plain-English case summary an analyst could actually read and act on.

This isn't just a demo feature bolted on for show, it's a genuine part of the architecture, sitting right after ring detection and before the analyst review step, and it directly supports the precision@k / analyst review capacity framing I'm already using, since a good summary is what actually lets an analyst get through more cases per day, not just a nicer chatbot output.

Important guardrail: the LLM only summarises evidence my pipeline already extracted, it doesn't get to freely speculate about the ring or invent details. Grounding the summary strictly in computed features (not letting the model reason beyond what's actually there) matters a lot here, since hallucination in a fraud/compliance context is a real, serious failure mode, not just an annoyance. 

## Tools I'm using

- pandas, NumPy for data wrangling
- NetworkX for building/analysing the graph
- python-louvain for community detection
- scikit-learn, XGBoost/LightGBM for modelling
- PyTorch Geometric if I get to the GNN stretch goal
- SHAP for explainability
- matplotlib + NetworkX drawing for visuals
- An LLM API (OpenAI or Claude) for the analyst-summary layer, with a prompt template designed to only summarise pre-extracted evidence, not free-reason about the case

## Metrics

Don't just report accuracy. Translate everything into numbers a fraud/risk team would actually care about:
- % of injected rings correctly detected
- False positive rate at a fixed detection rate (since flagging too many normal accounts annoys real customers and costs money)
- Estimated £ fraud prevented vs £ cost of false declines
- Precision@k, since real fraud analysts can only manually review so many flagged accounts a day


## What I still need to remember / limitations

- It's synthetic data, so this is a proof of concept, not something production-ready.
- Real deployment would need ongoing monitoring for model drift (something like Population Stability Index), but that's out of scope here
- The GNN part is a stretch goal, fine to skip if I run out of time, the core project stands without it
- Graphs are good for muling/rings specifically, not a catch-all for fraud in general.
- The GenAI summary layer needs a hallucination guardrail (only summarising pre-extracted evidence, never letting the model invent or speculate).