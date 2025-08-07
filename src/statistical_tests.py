import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, shapiro, levene
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("data/obesity_features.csv")

# chi-square test
contingency = pd.crosstab([df['low_veg_high_screen'], df['active_transport']], df['target'])
chi2, p, dof, expected = chi2_contingency(contingency)

print("###################### CHI-SQUARE TEST ######################")
print(f"chi2 value: {chi2:.4f}")
print(f"p-value: {p:.6f}")
print(f"degrees of freedom: {dof}")
print("reject null hypothesis:" if p < 0.05 else "fail to reject null hypothesis")
print()

# group data for anova
grouped = [df[df['target'] == i]['tue'] for i in sorted(df['target'].unique())]

print("###################### ANOVA ASSUMPTIONS ######################")
for i, group in enumerate(grouped):
    stat, pval = shapiro(group)
    print(f"Group {sorted(df['target'].unique())[i]}: Shapiro-Wilk p-value = {pval:.4f}")

levene_stat, levene_p = levene(*grouped)
print(f"Levene's test p-value: {levene_p:.4f}")
print()

anova_stat, anova_p = f_oneway(*grouped)
print("###################### ANOVA (TUE by OBESITY CLASS) ######################")
print(f"f-statistic: {anova_stat:.4f}")
print(f"p-value: {anova_p:.6f}")
print("reject null hypothesis:" if anova_p < 0.05 else "fail to reject null hypothesis")
print()

# post-hoc test ONLY IF NEEDED
if anova_p < 0.05:
    print("###################### POST-HOC: TUKEY'S HSD ######################")
    tukey = pairwise_tukeyhsd(df['tue'], df['target'])
    print(tukey)
    tukey = pairwise_tukeyhsd(df['tue'], df['target'])
    print(tukey)
