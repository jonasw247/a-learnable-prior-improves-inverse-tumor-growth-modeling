import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind_from_stats

def pairedTTest(values1, values2):
    
    tstat, pval = ttest_rel(values1, values2)

    #print results
    print("")
    print("Paired t-test")
    print("t-statistic:", tstat)
    print("p-value:", pval)

    # Calculate the differences
    differences = values1 - values2

    # Calculate the mean and standard deviation of the differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # Calculate Cohen's d for paired samples
    cohens_d = np.abs(mean_diff / std_diff)

    print(f"Effect size - Cohen's d: {cohens_d}")

def t_test_from_stats(mean1, std1, nobs1, mean2, std2, nobs2):
    # calculate the t test for two independent samples
    # compare samples
    t_stat, p = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1, mean2=mean2, std2=std2, nobs2=nobs2)
    print("t_stat: ", t_stat)
    print("p: ", p)
    print(f"cohen's d: {np.abs(mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)}")

    