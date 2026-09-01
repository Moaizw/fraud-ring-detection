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