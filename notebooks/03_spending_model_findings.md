## Distribution fitting possible for consumer spending ?

Unlike income sampling, where my ONS data contained deciles (p10 -> p90) for income,
there was no spread data for consumer expenditure. Tried finding income spending by
category with each category including some form of spread data e.g. deciles, quintiles 
etc, however I could only find the average spending for a certain category for each
income quintile. 

Ultimately, because I only had average data for each category, I couldn't fit 
distributions (requires spread data) to spending categories. 

## My Approach - Dirichlet + Lognormal Distribution (2 layers)

My approach to bypass this limited data problem is using a Dirichlet + lognormal distribution. 

**WHY DIRICHLET?** -> Spending across multiple categories (let's say in a week) isn't just several seperate numbers, it's COMPOSITIONAL. So, whatever gets spent of Food, Housing, Transport etc sums up to an accounts total spent for that week. Not possible for an account to spend £320 when its categories sum up to £350. 

That's exactly why we don't randomly sample from each category, because the total of those sample WOULDN'T sum up to the total spend. In essence, independent draws have no built-in awareness of each other at all. 

Dirichlet, a distribution family, addresses this as it will generate samples where the proportion of each sample MUST sum up to 1. So, let's say we have 12 categories and using dirichlet, we draw samples, where one sample might return higher than normal so the next few samples will automatically be pulled down to ensure the proportions sum up to 1. 

**WHY LOGNORMAL?** -> The dirichlet will sample the proportions for us, so out of the total spend, how much does this person spend on food expressed as a proportion (0 - 1).
BUT what about the amount someone spends in a week ? This is what lognormal is used for. 
I've made an assumption, where total amount someone spends in a week will follow a lognormal distribution: always positive, typically right skewed, where most people cluster around a typical weekly total with some spending noticeably more. 

BOTH distributions are required -> one samples the total amount spent in a week while the other will sample split across category spend. Multiplying the two will give a realistic £ / category for each individual account. 

### Two-layer Structure

So, when modelling consumer spending, we need to consider two things: 

The first is person-to-person variation, meaning the amount spent genuinely varies from person to person. Say we have two simulated accounts in the SAME income quintile, both would look up the SAME average spend from A26 (reference data). If we just assigned that average directly to both, they'd have identical spending, which isn't realistic, real people in the same income bracket don't all spend exactly the same amount. So instead of
assigning the average directly, we draw a RANDOM value around that average for each account, giving each one their own personal, slightly different total, this is what actually creates the person-to-person variation.

The second is the week-to-week variation. Consumer spending week in, week out won't be exactly the same. Sometimes, we may spend more or less due to unforeseen circumstance e.g. birthday party, family gathering etc. This variation is captured in the SECOND layer, drawn fresh every week, around that account's OWN personal total/mix from LAYER ONE, not the quintile average again. 

**WHY 2 LAYER STRUCTURE?**
If we had let's say just 1 layer so we're extracting spending samples around the average of that quintile, we WOULD have person-to-person variation, HOWEVER, because we've ignored the second layer (week-to-week variation), each individuals weekly spend would be identical, which doesn't represent true consumer spending behaviour. 

It's also worth mentioning that person-to-person variation should be real and substantial, while week-to-week noise for that same person should be smaller but still present. Therefore, a wider spread should be used in layer 1 while layer 2 should have a tighter spread to make sure each individuals weekly spend should follow a general pattern around the average spend with a slight 'wobble' here and there. 

### How do we determine the spread/scale in layer 1 and 2 ?

Initially I was going to use a label (1 to 5) assigned to each income quintile, with the assumption that higher quintiles have a WIDER spread/scale. This is an ASSUMPTION (not backed by data) but makes sense, those that earn more have the luxury to overspend more from time to time. Issue: this treats every account in the SAME quintile identically, someone just earning barely enough to enter quintile 3 would get the same spread as someone sitting near the top of quintile 3's range.

**REFINEMENT** -> The reference data gives real income boundaries, the
LOWER boundary of each quintile determines which quintile an account
falls into, and the MEDIAN of each quintile is used as the anchor point
for interpolation. Using a simulated account's actual income position
between two medians (not just its quintile label), I can interpolate a
precise spread value rather than a flat per-quintile one. Assumption:
this relationship scales linearly with income.

**Note:** "wider spread with higher income" applies directly to the
LOGNORMAL spread (bigger number = more variation). For the DIRICHLET
concentration, the same underlying belief means the concentration
number should DECREASE as income rises (smaller concentration = more
variety), the two distributions express "more variation" in opposite
directions - easy to get backwards, worth double-checking sign later
when picking actual numbers.

## Why A26 table specifically ?

Other tables I found presented average spend across whole households
(not just single adults), and didn't distinguish whether occupants were
working, retired, or a mix, meaning household size and composition
varied across income groups in those tables. That's a real confound,
richer income groups in those tables also tend to contain more people
and more earners, so any spending difference partly reflects household
size, not income itself.

A26 avoids this entirely, confirmed directly in the table: "weighted
average number of persons per household = 1.0" across every single
quintile, single-adult, non-retired households only. This isolates
income as the only thing varying between quintiles.

This also matches the project directly: every simulated account is one
individual, not a pooled household income, so A26 is a genuine
one-to-one match, unlike a table describing multi-person households.

**NOTE**
Confirmed via ONS Family Spending FYE 2025 bulletin: health/education
suppression in A26 is a genuine, documented ONS decision (volatility in
small samples for those categories). Also confirmed A26 is unaffected by 
this year's equivalisation methodology change (that only affects 
3.1E/3.2E/A18Eq/A19Eq), so A26 still uses the original one-adult reference 
scale as assumed throughout.

## Gross-to-net boundary conversion

A26's boundaries are in GROSS terms, but accounts spend from NET income.
Fix: annualise each weekly boundary (×52), run through gross_to_net.

**NOTE:** not a perfect translation, gross_to_net was built for individual
PAYE salary income, A26's boundaries come from a general household
survey. Still the right call for internal consistency, same tax logic
used throughout the pipeline, just worth knowing it's an approximation.


## Understanding WHY alpha < 1 resulted in abnormal proportions

Initially my concentration/spread ranged (25 -> 12) for week-to-week
variation BUT spread was tighter compared to total weekly spend.
However, I completely overlooked how a smaller concentration would
affect the resulting alpha value (proportion * concentration). For
example, Alcohol spending roughly 1.9% of total spend and at
concentration 18, it gave an alpha value of ~0.35. The issue was I
missed a KEY characteristic of the Dirichlet distribution: it behaves
completely differently in terms of likeliness of drawing a sample of
THAT PROPORTION.

Consider the alcohol example (alpha = 0.35): Dirichlet is an extension
of multivariate distribution so looking at Beta density formula:
**f(x)∝ x^α_1−1 * (1−x)^α_2−1**

x -> alcohol proportion
1 - x -> proportion of everything else

when alpha = 0.35 -> we end up with -0.65 and we know that when taking
the negative exponent of a number, as x (the proportion drawn) gets
smaller, the DENSITY (how likely we are to draw that sample) gets
BIGGER. BUT this doesn't really explain why we saw the 19.6% sample for
alcohol? Yes the density for smaller proportions would be much higher
than for bigger proportions, HOWEVER, the curve's tail doesn't reach 0
but near zero SO this means most of our samples drawn will have a
similar proportion, but there's ALSO a small chance we get an abnormal
proportion.

**FIX** Increase numbers in concentration scale: (25 - 18) -> (150 -
80). This will guarantee alpha val for alcohol > 1 and more of a bell
shaped curve where density reaches a 'peak' and then falls.

## Results (04_spending_aggregate_check.py)

After increasing concentration scale, the proportions for each category
looked more reasonable (similar to real A26 proportions for that quintile
group). However, this single result doesn't prove simulation model has 
calibrated i.e. a model where the outputs, on avg, match the real world 
quantities. It doesn't mean that model predictions should EXACTLY match 
real world outputs, rather, the predictions should closely resemble the 
real world outputs BUT with believable random variation around it. 

Therefore, in order to confirm whether a model has calibrated, thousands
of samples required, where each individual draw would show noisiness BUT
taking the average of these random draws would result in overall sample
value converging towards the distributions true centre (law of large numbers).

Results between simulated avg vs real proportion data shows every category
sits just below a 2.1% error (difference between simulated and real / real)
so no abnormal simulated values. Also the errors don't point in just one
direction (+ve or -ve); 6 categories are slightly positive (simulated vals
came out slightly higher than actual) and 5 are slightly negative so model
not biased (good -> demonstrates real randomness). 

Also, some of the bigger categories e.g. 'Other', 'Housing Fuel', 'Transport' 
show errors proportionally in line with the smaller categories indicating 
there's no sign bigger categories are easier to calibrate than smaller ones.

Worth noting specifically: alcohol_tobacco (the category that broke
before the concentration fix, one draw hit 19.6% on a real target of
1.9%) now sits at 2.07% error, the LARGEST single error in the table,
but still small and unremarkable. This is the direct, aggregate-level
confirmation that the concentration fix genuinely resolved the bias,
not just avoided one unlucky individual draw.

## ERROR SPOTTED - unrealistic spending on all categories for all samples

Noticed A26's alcohol_tobacco average (£8-9/week) looked far too low
against real prices (cigarettes alone £14-19 /pack, alcohol ~£4/pub
drink). Turns out this isn't wrong, it's a population wide average
BLENDED across spenders and non-spenders. Confirmed via ONS: only 10.6%
of UK adults smoke. So the £8-9 average is mostly zeros, pulled up by
the minority who spend a lot more.

This exposes a REAL BUG: Dirichlet always produces a positive share for
every category, every draw, it can never output exactly 0. So every one
of the 50,000 simulated accounts currently gets SOME alcohol/tobacco
spend, every single week, when in reality a huge chunk of the real
population should show exactly £0 in that category, always.

Found the real fix data: ONS LCFS technical report, Table 14 (FYE 2025),
gives "% of households recording any expenditure" per category (whole
population, not confirmed single-adult specific like A26, but same
source/year, treated as reasonable approximation). Mapped onto
CATEGORY_COLS:

food_nonalcoholic 99%, alcohol_tobacco 56%, clothing_footwear 52%,
housing_fuel_power 100%, household_goods_services 92%, transport 89%,
communication 98%, recreation_culture 97.8%, restaurants_hotels 79%,
misc_goods_services 98%, other_expenditure_items 97%

Threshold: categories under 95% get a participation mechanism:
alcohol_tobacco, clothing_footwear, restaurants_hotels, transport,
household_goods_services. Everything else close enough to universal
that it's not worth the added complexity.

### Design: personal weekly probability, not a fixed lifetime flag

Considered a single permanent yes/no per account per category, but
rejected it, someone who rarely dines out still occasionally will (a
birthday, a last-minute decision), reasons that are genuinely
unmodellable and shouldn't be forced into a fixed lifetime flag. This
IS the noise in the model, honest acknowledgment that individual
causation is unknowable.

Design instead: each account gets its own personal weekly PROBABILITY
per flagged category, drawn ONCE (via Beta, since a probability must
stay between 0 and 1), centred on the real ONS rate. Averaging many 
accounts personal rates should converge back to the real population 
rate (e.g. alcohol_tobacco personal rates averaging to ~56%), same 
calibration logic as everywhere else in this project.

Each week: draw one random number 0-1, compare against the account's
own personal rate. If random draw <= personal rate, category is active
this week (normal Layer 2 draw happens). If random draw > personal
rate, category is inactive, £0 for that category, that week.

### Integration with existing Dirichlet weekly draw

Money that would've gone to an inactive category isn't redistributed 
to other categories, it's just not spent, week's TOTAL shrinks accordingly. 
Reasoning: someone skipping a restaurant visit didn't plan to overspend 
elsewhere that week to compensate, they just spend less overall.

Mechanism: for inactive categories, subtract their normal personal_mix
share from that week's total BEFORE running Dirichlet, then build a
SMALLER alpha vector using only the active categories' proportions, run
Dirichlet on that reduced set, multiply by the (now smaller) total.

## Investigated switching to Table 3.3 disposable income boundaries (BRANCH)

Found Table 3.3 (LCFS, disposable income quintiles, same 1-adult
non-retired population as A26). Compared against derived boundaries
(A26 gross -> net via gross_to_net) on 1000 real generated accounts.
Result: 54.2% of accounts got assigned a DIFFERENT quintile depending
on which boundary set was used, always shifting down by exactly 1
quintile under the real disposable-income figures. Significant, not
noise.

Investigated WHY: ONS disposable income = post-tax earnings PLUS cash
benefits (Universal Credit, Child Benefit, etc.) PLUS other income
(pensions, investments), MINUS Council Tax. gross_to_net only models
employment salary minus Income Tax and NI, no benefits, no other
income, no Council Tax.

DECISION: keep gross_to_net + A26 derived boundaries, not switch to
Table 3.3. Disposable income measures a genuinely different population
(real single-adult households, many receiving benefits) than what this
project simulates (pure employed individuals, no benefits modelled at
all). Switching would mean assigning benefit-free simulated accounts
against boundaries substantially shaped by benefit income they don't
have, a DIFFERENT mismatch, not a smaller one. Better to stay
internally consistent (same tax logic used throughout the whole
pipeline) than import a more "official" number measuring a different
thing than what's actually being simulated.

Real, quantified limitation worth keeping visible: the derived
boundaries are LOWER than real disposable-income boundaries at every
quintile, since benefits push the real boundaries upward, and this
project doesn't model benefits at all. This means it takes LESS
simulated net income to cross a derived boundary than it would take
real (benefit-inflated) income to cross the equivalent real boundary,
so simulated accounts get placed roughly one quintile HIGHER than a
real household with the same take-home pay would sit.

Worked example, to make the direction concrete: Q2's derived boundary
is £21,154, Q2's real (disposable) boundary is £26,884, higher.
A simulated account with net_income = £24,000 sits ABOVE the derived
boundary, so it's placed in Q2. But £24,000 sits BELOW the real
boundary, so against the real scale, that same £24,000 person would
actually still be in Q1. Same income, different quintile, purely
because the derived boundary is easier to clear than the real one.