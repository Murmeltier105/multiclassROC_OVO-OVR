import scipy.stats
import pandas as pd
import numpy as np


# Confusion matrix
cmDF = pd.read_excel('/home/dr1/PycharmProjects/GraMa/trainedModelsNormalized/Ce2NoGroupAug+/Cmatrix.xlsx')
cm = cmDF[[0,1, 2]].to_numpy()

#
# Cohen's kappa
#
# Sample size
n = np.sum(cm)
# Number of classes
n_classes = cm.shape[0]
# Agreement
agreement = 0
for i in np.arange(n_classes):
    # Sum the diagonal values
    agreement += cm[i, i]
# Agreement due to chance
judge1_totals = np.sum(cm, axis=0)
judge2_totals = np.sum(cm, axis=1)
judge1_totals_prop = np.sum(cm, axis=0) / n
judge2_totals_prop = np.sum(cm, axis=1) / n
by_chance = np.sum(judge1_totals_prop * judge2_totals_prop * n)
# Calculate Cohen's kappa
kappa = (agreement - by_chance) / (n - by_chance)

#
# Confidence interval
#
# Expected matrix
sum0 = np.sum(cm, axis=0)
sum1 = np.sum(cm, axis=1)
expected = np.outer(sum0, sum1) / n
# Number of classes
n_classes = cm.shape[0]
# Calculate p_o (the observed proportionate agreement) and
# p_e (the probability of random agreement)
identity = np.identity(n_classes)
p_o = np.sum((identity * cm) / n)
p_e = np.sum((identity * expected) / n)
# Calculate a
ones = np.ones([n_classes, n_classes])
row_sums = np.inner(cm, ones)
col_sums = np.inner(cm.T, ones).T
sums = row_sums + col_sums
a_mat = cm / n * (1 - sums / n * (1 - kappa))**2
identity = np.identity(n_classes)
a = np.sum(identity * a_mat)
# Calculate b
b_mat = cm / n * (sums / n)**2
b_mat = b_mat * (ones - identity)
b = (1 - kappa)**2 * np.sum(b_mat)
# Calculate c
c = (kappa - p_e * (1 - kappa))**2
# Standard error
se = np.sqrt((a + b - c) / n) / (1 - p_e)
# Two-tailed statistical test
alpha = 0.05
z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
ci = se * z_crit * 2
lower = kappa - se * z_crit
upper = kappa + se * z_crit

print(
    f'kappa = {kappa}\n',
    f'a = {a:.3f}, b = {b:.3f}, c = {c:.3f}\n',
    f'standard error = {se:.3f}\n',
    f'lower confidence interval = {lower:.3f}\n',
    f'upper confidence interval = {upper:.3f}',
    sep=''
)