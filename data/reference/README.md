# Reference data sources

Every file here is either a real cited source or clearly marked as a derived/documented
assumption. No numbers in this project are invented without a stated basis.

## `salary_lookup_age_occupation_fulltime_2025.csv`

- **Source**: ONS Annual Survey of Hours and Earnings (ASHE), Table 20.7a — "Annual pay,
  Gross — Age by Occupation (SOC2020)", Full-Time tab, 2025.
- **Contents**: median, mean, and 10th–90th percentile annual gross pay for the 9 major
  SOC2020 occupation groups, crossed with 6 age bands (18-21, 22-29, 30-39, 40-49,
  50-59, 60+).
- **Why major groups only**: the finer 2-digit SOC sub-groups have far more suppressed
  cells (small sample sizes) once crossed with age band. The 9 major groups stay
  statistically robust across every age band.
- **Known gaps**: a handful of percentile cells are still blank even at this level
  (e.g. 18-21 Managers has no 10th/20th/30th percentile). Fall back to the overall
  age-band figure (not occupation-specific) when a specific cell is missing.
- **This is gross pay**, not take-home. Convert using the gross-to-net function
  (2025/26 UK income tax + NI bands) before using as a transaction amount.

## `salary_lookup_age_occupation_parttime_2025.csv`

- **Source**: same ASHE table as above, Part-Time tab.
- **Important distinction**: ASHE only covers employees (PAYE), not the self-employed.
  This represents part-time *employees*, not freelance/gig workers.
- **Known gaps**: data completeness is worse than full-time. Some cells are entirely
  suppressed (e.g. 18-21 Managers, 18-21 Professional occupations have no data at all
  for this age/occupation combination). Fallback logic needs to handle whole missing
  rows, not just individual percentiles.
- Part-time pay does not scale linearly from full-time (e.g. part-time Managers earn
  proportionally much less than half of full-time Managers), so don't derive this by
  halving the full-time table — use these real figures directly.

## `spending_by_income_quintile_single_adult_2025.csv`

- **Source**: ONS Living Costs and Food Survey (LCFS) / Family Spending in the UK,
  "Workbook 2 — Expenditure by Income", Table A26 — "Expenditure of one adult
  non-retired households by gross income quintile group", FYE 2023–2025.
- **Why this table over the general population one (Table A4)**: A4 mixes households
  of different sizes (1.2 people/household in the lowest decile up to 3.0 in the
  highest), which would distort per-account modelling. A26 is fixed at exactly 1
  person per household across every quintile, a direct match for one simulated
  account's spending.
- **Units**: average weekly household expenditure (£), by category, per income
  quintile (5 groups, not deciles).
- **Known gaps**: Health and Education categories are suppressed ("..") in every
  quintile — likely because these are lumpy/infrequent expenses not well captured
  in a weekly diary survey, not a data quality problem specific to this project.
  Left blank/omitted rather than invented.
- **Not yet covering retirees**: an equivalent "1 adult retired household by income
  quintile" table exists in ONS's Family Spending series (referenced as Table A24/A25
  in the workbook contents) but isn't in this particular workbook. Needed later for
  the pensioner archetype.

## `age_band_distribution_by_archetype_2025.csv`

- **Source**: same ASHE Table 20.7a, the age-band total rows (before occupation
  breakdown) for the Full-Time and Part-Time tabs.
- **Contents**: real population share by age band, separately for full-time and
  part-time employees. Use this to weight age-band sampling per archetype, rather
  than sampling age uniformly.
- **Note the shape difference**: full-time is concentrated in prime working age
  (30-49 is over half the population); part-time is much flatter, with a notably
  larger 60+ share (19.5% vs 8.9% for full-time) and larger 18-21 share (7.3% vs
  1.8%). Don't reuse one age curve for both archetypes.
- **How this combines with the salary lookup tables**: sample age band first using
  this file's `population_share` column, then sample occupation *within* that age
  band using the `jobs_thousand` column already present in the salary lookup CSVs.
  This gives a fully joint, realistically-weighted (age, occupation) draw, both
  levels grounded in the same ONS source.

## Documented modelling assumptions (not directly sourced)

These are implementation choices, not claims about the real world, and don't need
a citation, but are listed here for transparency:

- Two-layer randomness (personal baseline drawn once per account, daily/weekly wobble
  around that baseline) — a standard simulation technique, not a specific data source.
- Distribution *family* choices (e.g. treating income as right-skewed and always
  positive) — a standard convention, superseded anyway by directly sampling from real
  ONS percentile breakpoints rather than assuming a parametric shape.
- Gig worker archetype — considered and excluded. See `docs/project_proposal.md`,
  Limitations section, for the reasoning.
