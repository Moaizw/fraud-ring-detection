"""
This file is ONLY for exploration purposes. The actual distribution fitting code belongs
in src/generation/income.py 

In this file I explore the ONS income percentile data for certain age_band & occupation pairs
before fitting any distributions. Aim -> check whether distribution looks lognormal (often used to model income)
or whether tail behaviour suggests something heavier (motivates GB2 comparison). 
"""

"""
FINDINGS:

Compared 3 representative (age_band, occupation) cells: 18-21 Elementary,
30-39 Associate Professional, 40-49 Managers.

- Raw log-scale plot (income vs raw percentile position) showed all three
  curving upward toward p90, however raw percentile positions aren't evenly 
  spaced in z-score terms, so even a genuinely lognormal distribution would 
  show some curve on that plot.

- Corrected version (log income vs actual z-score -> QQ-style check) 
  showed the curvature mostly disappearing for 18-21 Elementary and
  30-39 Associate Professional, both look reasonably close to straight
  lines, consistent with lognormal.

- 40-49 Managers still showed mild upward curvature at the top end (p75-p90)
  even after the z-score correction, suggesting a possibly heavier tail
  than lognormal predicts for this cell.

CONCLUSION: this is visual/qualitative evidence only, not proof, only
3 cells and 11 points each. It motivates actually running the formal
fitting + AIC/BIC comparison (rather than assuming lognormal without
checking), with Managers-type (senior/high-paid) occupations as the
cells most worth watching for whether GB2's extra complexity is genuinely
justified once fit quality is properly compared. Whether GB2 actually
wins anywhere is still an open question at this point.

Next step: src/generation/income.py, formal fitting.
"""

import matplotlib.pyplot as plt
import scipy
import pandas as pd 

#load FULL TIME salary CSV file 
df = pd.read_csv('/Users/moaizwazir/Downloads/fraud-ring-detection/data/reference/salary_lookup_age_occupation_fulltime_2025.csv')

#extract (age_band, occupation) rows
PERCENTILE_COLS = ['p10', 'p20', 'p25', 'p30', 'p40', 'median', 'p60', 'p70', 'p75', 'p80', 'p90']
PERCENTILE_POSITIONS = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
PERCENTILE_PROB = [x / 100 for x in PERCENTILE_POSITIONS]
Z_SCORE = scipy.stats.norm.ppf(PERCENTILE_PROB)

def get_percentile_values(df, age_band, occupation):
    row = df[(df['age_band'] == age_band) & (df['occupation'] == occupation)]
    if row.empty:
        raise ValueError(f"No data found for {age_band}, {occupation}")
    return row[PERCENTILE_COLS].values.flatten() #flattens to 1D -> for plotting later


#plot percentile pos vs income BUT also plot again with log income vs z-score (QQ plot)
def plot_income_curves(df, cells, log_scale=False, save_path=None):
    x_vals = Z_SCORE if log_scale else PERCENTILE_POSITIONS

    plt.figure()
    for label, (age_band, occupation) in cells.items():
        values = get_percentile_values(df, age_band, occupation)
        plt.plot(x_vals, values, marker='o', label=label)

    if log_scale:
        plt.yscale('log')
        plt.xlabel('Z-score')
        plt.ylabel('Log Income (£)')
        plt.title('Income percentiles (log scale, QQ style)')
    else:
        plt.xlabel('Percentile Position')
        plt.ylabel('Income (£)')
        plt.title('Income percentiles (linear scale)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


#intentionally picked 3 (age_band, occupation) rows to compare (these pairs should look meaningfully different -> eyeball data)
cells = {
    '18-21 & Elementary occupations': ('18-21', 'Elementary occupations'),
    '30-39 & Associate professional occupations': ('30-39', 'Associate professional occupations'),
    '40-49 & Managers': ('40-49', 'Managers directors and senior officials'),
}

plot_income_curves(df, cells, log_scale=False, save_path='notebooks/income_linear.png')
plot_income_curves(df, cells, log_scale=True, save_path='notebooks/income_log.png')