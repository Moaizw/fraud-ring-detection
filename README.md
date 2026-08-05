# Hierarchical Fraud Ring Detection

A graph-based fraud detection project. Instead of classifying individual transactions
as fraud/not fraud, this project treats fraud as a network problem: money mule rings
and hierarchical fraud networks are surfaced through account connections (shared
devices, shared payees, transaction patterns), not single-transaction anomalies.

See [`docs/project_proposal.md`](docs/project_proposal.md) for the full aim,
methodology, and business framing.

## Status

🚧 In progress. Currently building the synthetic data generator.

- [x] Project scoping and methodology write-up
- [x] Reference data collection (ONS ASHE, LCFS)
- [x] Account archetypes: full-time employee
- [x] Account archetypes: part-time employee
- [ ] Income sampling: fit lognormal, Gamma/Weibull, and GB2 distributions
      to ONS percentile data via quantile matching (currently working on full-time and part-time only)
- [ ] Account archetypes: student
- [ ] Account archetypes: pensioner
- [ ] Fraud ring injection logic
- [ ] Graph construction
- [ ] Community detection (hierarchical)
- [ ] Modelling (XGBoost baseline, GNN stretch goal)
- [ ] Evaluation

## Project structure

```
fraud-ring-detection/
├── data/
│   └── reference/        # ONS reference tables used to calibrate archetypes
├── src/
│   ├── archetypes/        # one module per account archetype
│   ├── generation/        # income, spending, transaction generation logic
│   └── utils/
├── notebooks/              # exploration and sanity-checking only, not final logic
├── tests/
├── requirements.txt
└── README.md
```

## Data sources

All reference data used to calibrate the synthetic accounts is documented in
[`data/reference/README.md`](data/reference/README.md), including the exact ONS
table numbers and what's a real cited figure versus a documented modelling
assumption.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```
