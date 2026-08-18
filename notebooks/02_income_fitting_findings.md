# Income distribution fitting — Notes & Findings

Rough working notes as each distribution gets fitted and compared.
See src/generation/income.py for the actual pipeline code.

## Lognormal

### Findings:

Tested on 30-39 Professional (full-time): s=0.368, scale=49010, real median
was 48190 so ~1.7% off, fine.

Checked all 11 percentiles though and there's a pattern -> both tails
(p10, p90) undershoot by ~3%, p60/p70 overshoot by ~2-3%, p40 basically
bang on. Same tail curve behaviour I saw in the QQ plots earlier
(notebooks/01_income_distribution_exploration.py). So lognormal's good
but not perfect - need to actually check if gamma/GB2 do better via
AIC/BIC rather than just assuming lognormal's good enough.

Ran across all 54 rows (full-time). 44 rows have all 11 real points, rest
are lower (down to just 4 for 18-21 Managers, the one row we already knew
was sparse from the archetype work). Didn't interpolate missing points -
fit on whatever real ONS points are actually there, since interpolating
first would mean fitting the model to numbers I made up, not real data.

Added a confidence flag using degrees of freedom (n_points -> 2 params).
Rather than just using raw point count, since low dof = fit had barely any 
real data to prove itself wrong against, "small error" doesn't mean much 
with only 2 df. Only 18-21 Managers gets flagged low (df=2). Worth noting 
its fitted s (0.134) is notably smaller/tighter than every other 18-21 row 
-> could be real, could just be an artifact of only having 4 points to fit
against. Treat that one row's params with caution. 

## Gamma

### Notes:

First have to estimate the parameters (k -> shape, theta -> scale) as our
starting point for curve fitting. Unlike lognormal, exp(mu) != median
therefore method of moments required. For Gamma, mean = k * theta AND
variance = k * theta^2, which we can rearrange to:

**k = mean^2 / variance**

**theta = variance / mean**

So, mean is provided in the ONS data, however variance, I make an assumption,
where distribution is roughly spread out in a bell-ish way (N.B. this will
generate a parameter used as the initial val in fitting so OK to make this
assumption). Means gap between p10 & p90 corresponds to a known number of
standard deviations under normal distribution (2.5631). Equation for SD:

**sd = (P90 - P10) / 2.5631**

**variance = sd^2**

mean_value fallback chain added for rows where ONS 'mean' is missing
(didn't hit this for full-time, mean was present everywhere here, but
expect to need it for part-time): real mean -> median (if present in
row) -> plain average of whatever percentiles ARE present. Only ever
seeds curve_fit's starting guess so doesn't need to be exact.

### Findings:

Gamma - same row as lognormal (30-39 Professional, full-time):
a=7.52, scale=6906. Real mean 55821.

Same tail pattern as lognormal (tails undershoot, p60/p70 overshoot) but
every error is bigger - p10 -6.16% vs lognormal's -3.06%, roughly double.
So for THIS row, lognormal is the better fit so far. Not necessarily true
for every row/occupation though - need the full AIC/BIC comparison across
all 54 rows to know for sure, this is just one data point.

Ran across all 54 rows (full-time). Same n_points pattern as lognormal
(same rows sparse, same 4-point row for 18-21 Managers); makes sense,
it's the same underlying missing-data structure either way.

Interesting pattern in the fitted shape param (a) for Managers across age
bands: 30-39 a=3.21, 40-49 a=2.58, 50-59 a=2.47, 60+ a=2.22. Smaller a =
more spread out/skewed. So Managers' income spread gets WIDER with age -
makes sense, more senior/longer-tenured managers = more variation in how
senior/well-paid they've become. Didn't see this kind of interpretable
pattern as obviously in lognormal's s values.

## Weibull

### Notes:

Initially thought about skipping this (assumed weibull = gamma = thin-tailed,
not worth testing given gamma already lost to lognormal on tails). Wrong -
weibull's tail depends on its shape param k: k>=1 decays like exponential
(thin), k<1 decays slower ("stretched exponential", heavier than gamma).

Even best case though, weibull's tail (exp(-x^k)) still decays faster than
lognormal's (exp(-(ln x)^2)) for any k>0. So hypothesis: weibull might beat
gamma, probably still won't beat lognormal - but fitting it to check rather
than assuming.

Starting guess is messier than gamma's - mean/variance equations both involve
the Gamma function (Γ), no clean algebra:

**mean = lambda * Γ(1 + 1/k)**
**variance = lambda^2 * [Γ(1 + 2/k) - Γ(1 + 1/k)^2]**

Too messy to solve exactly just for a p0. Using an empirical CV-based shortcut
instead (CV = sd/mean, from wind-speed modelling, weibull's classic use case):
1. reuse the sd estimate from gamma (p10/p90 gap / 2.5631)
2. CV = sd / mean
3. plug into known CV-to-k approximation -> starting k

### Findings:

Ran across all 54 rows. k never dropped below 1 anywhere - min 1.54,
mean 3.84, max 8.30. So the k<1 possibility (heavier weibull tail) never
actually happened in practice for this data, hypothesis fully confirmed
at scale, not just on the one test row. Combined with the residual
comparison on 30-39 Professional (weibull's p10 error -12.8%, worse than
both gamma -6.16% and lognormal -3.06%), weibull is clearly the weakest
of the three fitted so far for this data. Keeping it in the final
AIC/BIC comparison anyway. 

## GB2

### Notes:

GB2 is a distribution with 4 parameters (a, b, p, q). As a generalised
distribution, it can **reduce** to several simpler distributions such as
Gamma or Weibull, **when specific parameters are fixed or taken to a limit**.
So, in other words, these 'simpler' distributions are nested within GB2.

Previous distributions used had two parameters (shape, scale) but GB2 has
two additional params (p -> shapes lower tail, q -> shapes upper tail) so
has much more shape flexibility. q is specifically important to us because
when q is small -> heavier tail, and from our findings, our data has heavier
tails (lognormal beat Gamma/Weibull).

For previous distribution fittings, I converted income to a probability
using ppf (reverse of CDF), HOWEVER, I had the luxury of the .ppf() method
already built into scipy.

scipy.stats doesn't have a GB2 object so there's no prebuilt gb2.ppf(). I
have to build that myself, which is why I'm going to explain how I intend
to do this.

This is a two-link chain, and working through it manually also helps me
understand what's 'under the hood' in scipy's .ppf() method for the other
distributions, since scipy is doing exactly this same kind of reversal,
just hidden from me.

CDF (income -> probability): start with x (income), compute a transformed
variable z from it, then compute u (probability) from z. x -> z & z -> u
is the two-link chain. However I WANT probability -> income (reverse
CDF/ppf), so I need to go u -> z, then z -> x instead. **N.B.** each link
has to be reversed separately, in reverse order (last forward step first),
since it's a chain of two dependent transformations, not one single step.

Why compute this transformed variable z from x at all? GB2's CDF requires
an integral that's very hard to solve directly on x, BUT substituting x
for z results in an integral that's already been solved, it defines the
Beta distribution's CDF, known as the incomplete beta function. The
substitution wasn't arbitrary, it was specifically chosen because it turns
GB2's hard problem into an already-solved Beta-distribution problem.

- **z = (x/b)^a / (1 + (x/b)^a)**   <- NOT a z-score, just the name for
  this transformed variable, bounded between 0 and 1, nothing to do with
  the normal distribution

- **u = I_z(p,q)**   <- z, plugged into the Beta distribution's CDF
  (the "incomplete beta function")

1. Reverse link 2 first (last forward step, first to reverse): given u,
   find the z that would have produced it. "I know the answer u, what z,
   plugged into the incomplete beta function, would have given me that
   answer?" That's exactly what betaincinv(p, q, u) computes.
2. Now that I have z, reverse link 1 using the original substitution
   equation, rearranged algebraically to solve for x:

   **x = b * (z / (1 - z)) ^ (1/a)**

### Hand-derived test case (for verifying _gb2_ppf once written)

a=2, b=40000, p=2, q=1 (q=1 collapses the incomplete beta to a plain power,
u = z^p, making this hand-computable without betaincinv):

- u=0.5 -> z=√0.5=0.7071 -> x ≈ £62,152
- u=0.1 -> z=√0.1=0.3162 -> x ≈ £27,205
- u=0.9 -> z=√0.9=0.9487 -> x ≈ £172,000

_gb2_ppf should reproduce these (via the general betaincinv path, not the
q=1 shortcut) before it's trusted on any real data.

Confirmed: _gb2_ppf(u, a=2, b=40000, p=2, q=1) reproduces the hand-derived
values almost exactly (u=0.1: 27202 vs hand-calc 27205, u=0.5: 62151 vs
62152, u=0.9: 171985 vs 172000 - tiny differences just from hand-rounding
to 4dp). Core GB2 maths confirmed correct before fitting to any real data.

### GB2 starting guess

Found a paper (Manandhar & Nandram 2025, hierarchical Bayesian GB2 for
small area estimation) that confirms two useful things:

1. GB2 is genuinely the standard choice for income data, they cite
   McDonald 1984 showing GB2 fits US family income better than every
   other distribution tested. Good citation that fitting GB2 here isn't
   overkill, it's normal practice.

2. There IS an actual formula for GB2's moments:
   E[X^k] = b^k * B(p + k/a, q - k/a) / B(p, q)
   (B = beta function). Set k=1 for mean, k=2 for variance, in theory
   could solve these for a starting guess instead of just winging it.

BUT, the catch: this only works if q > k/a, and more generally moments
can straight up NOT EXIST if the tail is heavy enough (q too small
relative to k/a). This is a real property of GB2, not a mistake, the
exact method-of-moments approach could break down in exactly the
heavy-tail region I'm most interested in testing for.

Worth understanding WHY moments can fail to exist, not just that they
can (had to go over this twice myself):

A mean is really just an integral -> sum every possible value, weighted
by how likely it is. For most distributions this settles down to a
normal finite number. But if a distribution's tail is heavy enough, that
sum can literally never settle down, it keeps growing as more extreme,
rare, very-high values get included. When that happens the honest answer
to "what's the average" is genuinely infinity/undefined, not a bug, just
what heavy tails actually do mathematically. Same idea as the classic
Pareto example (city sizes, wealth, insurance losses, fields with
genuinely extreme outliers often just don't have a clean average).

So: q small (heavy tail) is EXACTLY the region where GB2's moments are
most likely to break down. Not a coincidence, same underlying property
(heavy tail) showing up twice, once in the shape, once in whether the
moment formula even has a valid answer. Makes the exact method-of-moments
route risky specifically in the region I'm most trying to test (whether
the data needs a heavier tail than gamma/weibull could give it).

Given all that, sticking with a simpler pragmatic starting guess rather
than solving the moment equations exactly:
- a, b: start from already-fitted GAMMA params for that row (gamma is
  simpler/more stable to fit, reuse its answer as a rough starting point)
- p, q: start both at 1 (neutral, no assumption about tail shape)
- bounds: all 4 params > 0 in curve_fit, stops the optimizer wandering
  somewhere meaningless

Not the exact textbook approach, but genuinely reasonable given the
above.

### Findings:

GB2 -> same row as the others (30-39 Professional, full-time):
a=6.97, b=43953, p=0.726, **q=0.490**. Real mean 55821.

q=0.49, comfortably below 1, confirms the heavy tail hypothesis.

Errors vs the other three on the exact same row:
             p10       p90
Weibull    -12.83%   -7.58%
Gamma       -6.16%   -4.82%
Lognormal   -3.06%   -3.09%
GB2         +0.37%   +0.14%

Every percentile under 1% error for GB2, biggest is p60 at 0.78%. Roughly
10-20x better than lognormal specifically at the tails, which is the exact
gap I set out to close. Hypothesis confirmed, not just on theory now but
with real numbers.

Still need AIC/BIC before finalising, GB2 has 4 params
vs lognormal's 2 so some improvement is expected just from flexibility.
Given how big the gap is I'd expect GB2 to still win after the penalty,
but testing > assumption hence AIC/BIC.

**KEY LIMITATION:**

While GB2 fit better to (30-39 Professional, full-time), it performed poorly 
for (18 - 21 Managers, full-time). Reason -> GB2 NEEDS n_points >= 5 minimum 
(4 params + atleast 1 dof). curve_fit "fit" perfectly on paper but found a 
degenerate solution (p=492, q=0.039, nonsensical vs every other row's fitted 
values) that didn't even genuinely match the real points, plus a divide by zero 
warning from the optimizer wandering into invalid territory.

To address this limitation, I will add a constraint in fit_gb2_all_rows ->
skip rows where n_points < 5, not just < 2 like the simpler distributions.
Also when using AIC/BIC to compare the 4 distributions, this row and any others
will be excluded from the GB2 comparison, will remain in lognormal/gamma/weibull.

## How well do the chosen distributions fit to data ?

AIC/BIC used to assess how well candidate distribution fit to data.

AIC -> prioritises model accuracy (fixed penalty used for complexity)
BIC -> prioritises on finding the actual true model for a certain 
       underlying system. It will therefore impose a larger penalty
       on more complex models (more features) to prevent overfitting.

AIC/BIC formulas use the maximum likelihood i.e. 'given this distribution 
and its best-fit params, how probable was the data I actually observed ?'
This question is answered using MLE, which searches for params that make
your observed data as probable as possible under assumed distribution.

However, I used **curve_fit**, which uses least squares instead to find
the best income val by finding the min squared difference between predicted
and real income vals. So one technique (MLE) generates probability while the 
other (RSS) generates how far off a prediction is to the real val. Good thing
is that I can actually still use AIC/BIC to assess distribution fit, if I 
assume that the residuals (difference between vals) follows a normal distribution
with constant spread. Formula will be slightly different and derived from RSS. 

- **AIC = n * ln(RSS/n) + 2k**
- **BIC = n * ln(RSS/n) + k * ln(n)**

n -> number of data points
k -> number of params

### Comparing distributions using AIC/BIC

Computed AIC/BIC score for ONLY one row (30-39, Professionals) for all 4 distributions
just to confirm GB2 > lognormal > gamma > weibull. Made sure to select a row which had 
atleast 5 data points (so all 4 can be fairly compared). 

**RESULTS**
distribution  k         aic         bic
3          gb2  4  121.894552  123.486133
0    lognormal  2  160.235156  161.030947
1        gamma  2  168.676305  169.472095
2      weibull  2  178.895217  179.691008

The results confirm assumption I made from findings where gb2 outperforms the other 3 
distributions even with greater penalty (more params). 

### Final Comparison

**Higher paid/skilled occupations**:

Age band	Occupation	n	Winner	GB2 status
18-21	Managers	4	lognormal	insufficient data (n<5)
22-29	Managers	10	gb2	won
30-39	Managers	11	lognormal	no convergence
40-49	Managers	11	gb2	won
50-59	Managers	11	gb2	won
60+	Managers	11	lognormal	no convergence
18-21	Professional	9	gamma	no convergence
22-29	Professional	11	lognormal	no convergence
30-39	Professional	11	gb2	won
40-49	Professional	11	gb2	won
50-59	Professional	11	gb2	won
60+	Professional	11	gb2	won
18-21	Associate professional	10	weibull	converged, lost
22-29	Associate professional	11	lognormal	no convergence
30-39	Associate professional	11	lognormal	no convergence
40-49	Associate professional	11	gb2	won
50-59	Associate professional	11	gb2	won
60+	Associate professional	11	gb2	won

- Comparing how well the distributions fit to higher paid/skilled occupation,
  you can see the GB2 10/18 rows in this group. GB2 also seems to struggle more 
  in the younger age brackets (18-21, 22-29) and often wins from 40-49 onwards. 

**Admin/Sectretarial**:
Age band	Occupation	n	Winner	GB2 status
18-21	Administrative	10	weibull	converged, lost
22-29	Administrative	11	gb2	won
30-39	Administrative	11	gb2	won
40-49	Administrative	11	gb2	won
50-59	Administrative	11	lognormal	converged, lost (gb2_bic 227.8 vs lognormal 155.6)
60+	Administrative	11	lognormal	no convergence

- Mixed results seen here. One thing highlighting is the 50-59 row, where GB2 won
  but still lost by a significant margin.

**Lower paid/manual/service occupations**:
Age band	Occupation	n	Winner	GB2 status
18-21	Skilled trades	11	lognormal	no convergence
22-29	Skilled trades	11	lognormal	no convergence
30-39	Skilled trades	11	lognormal	no convergence
40-49	Skilled trades	11	lognormal	no convergence
50-59	Skilled trades	11	lognormal	no convergence
60+	Skilled trades	11	lognormal	no convergence
18-21	Caring/leisure	10	weibull	no convergence
22-29	Caring/leisure	11	lognormal	no convergence
30-39	Caring/leisure	11	gamma	no convergence
40-49	Caring/leisure	11	gamma	no convergence
50-59	Caring/leisure	11	lognormal	no convergence
60+	Caring/leisure	11	lognormal	no convergence
18-21	Sales	8	weibull	no convergence
22-29	Sales	11	lognormal	no convergence
30-39	Sales	11	lognormal	no convergence
40-49	Sales	11	lognormal	no convergence
50-59	Sales	11	lognormal	no convergence
60+	Sales	10	lognormal	no convergence
22-29	Process plant	11	lognormal	no convergence
30-39	Process plant	11	lognormal	no convergence
40-49	Process plant	11	lognormal	no convergence
50-59	Process plant	11	lognormal	no convergence
60+	Process plant	11	lognormal	no convergence
18-21	Elementary	10	weibull	no convergence
22-29	Elementary	11	lognormal	no convergence
30-39	Elementary	11	lognormal	no convergence
40-49	Elementary	11	lognormal	no convergence
50-59	Elementary	11	lognormal	no convergence
60+	Elementary	11	lognormal	no convergence

- These results are a bit unexpected. We saw in the high-paid table, GB2 loses
  due to insufficient data (higher params in GB2). However, in this table, GB2
  doesn't even converge let alone win, so that's something that I need to look
  into. 
- Other than that, we do see lognormal dominating for these occupations, which is
  expected from previous findings, where lognormal exhibits thicker tails (compared
  to weibull/gamma -> our data has heavy tails)

### Why GB2 no convergence ?

Carried out a small diagnostic check, where I computed the p90/p10 ratios and checked
to see if they lined up with the rows where GB2 didn't converge. The reason for this
check is to confirm something: GB2's params (p & q) shape the tails but when income
is tightly clustered (small p90/p10 ratio), tail 'shape' isn't really in the data.
Now because there is no tail behaviour, everything sits flat (incomes clusetered
together), so the predicted income vals barely change AND this is why the optimiser
fails to converge for GB2; every value nearby in the search space is as equally good.
This is exactly what the p90/p10 ratio diagnostic confirms:

p90/p10 ratio	Rows	Lognormal wins	GB2 wins	Gamma wins
< 2.0	9	8	1	0
2.0 - 2.3	23	18	2	2
2.3 - 3.1	10	3	7	0
> 3.1	4	2	2	0

You can see lognormal wins most of the time when p90/p10 ratio is small (< 2.3). 
On the other hand, GB2 wins most cases where this ratio is high i.e. when income 
spread much greater.

Two cases worth highlight:
- 22-29 Admin (1.85 ratio) & 40-49 Admin (2.15 ratio) both won by GB2 despite ratios
  being in the 'lognormal favour' zone.
- 30-39 Manager (3.82) & 60+ Managers (4.84), which were the two highest spread rows
  in my dataset and GB2 still couldn't converge on them. 

