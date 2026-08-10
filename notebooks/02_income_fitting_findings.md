# Income distribution fitting — findings

Rough working notes as each distribution gets fitted and compared.
See src/generation/income.py for the actual pipeline code.

## Lognormal

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

First have to estimate the parameters (k -> shape, theta -> scale) as our
starting point for curve fitting. Unlike lognormal, exp(mu) != median
therefore method of moments required. For Gamma, mean = k * theta AND
variance = k * theta^2, which we can rearrange to:

# k = mean^2 / variance

# theta = variance / mean

So, mean is provided in the ONS data, however variance, I make an assumption,
where distribution is roughly spread out in a bell-ish way (N.B. this will
generate a parameter used as the initial val in fitting so OK to make this
assumption). Means gap between p10 & p90 corresponds to a known number of
standard deviations under normal distribution (2.5631). Equation for SD:

# sd = (P90 - P10) / 2.5631

# variance = sd^2

mean_value fallback chain added for rows where ONS 'mean' is missing
(didn't hit this for full-time, mean was present everywhere here, but
expect to need it for part-time): real mean -> median (if present in
row) -> plain average of whatever percentiles ARE present. Only ever
seeds curve_fit's starting guess so doesn't need to be exact.

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

mean = lambda * Γ(1 + 1/k)
variance = lambda^2 * [Γ(1 + 2/k) - Γ(1 + 1/k)^2]

Too messy to solve exactly just for a p0. Using an empirical CV-based shortcut
instead (CV = sd/mean, from wind-speed modelling, weibull's classic use case):
1. reuse the sd estimate from gamma (p10/p90 gap / 2.5631)
2. CV = sd / mean
3. plug into known CV-to-k approximation -> starting k

# Findings:

Ran across all 54 rows. k never dropped below 1 anywhere - min 1.54,
mean 3.84, max 8.30. So the k<1 possibility (heavier weibull tail) never
actually happened in practice for this data, hypothesis fully confirmed
at scale, not just on the one test row. Combined with the residual
comparison on 30-39 Professional (weibull's p10 error -12.8%, worse than
both gamma -6.16% and lognormal -3.06%), weibull is clearly the weakest
of the three fitted so far for this data. Keeping it in the final
AIC/BIC comparison anyway. 