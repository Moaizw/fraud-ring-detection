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

