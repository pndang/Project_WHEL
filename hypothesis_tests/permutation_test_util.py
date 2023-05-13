import pandas as pd
import numpy as np
import os


def tvd_of_groups(df, groups, cats, show_steps=False):

    '''
        groups: the binary column
        cats: the categorical column
    '''

    # Get count of each pair of catogory and group
    counts = df.pivot_table(index=cats, columns=groups, aggfunc='size')

    # Normalize each column
    distr = counts / counts.sum()

    if show_steps:
        print("\nSTEPS:")
        print(counts)
        print()
        print(distr)
        print()
        print(distr.diff(axis=1))
        print()
        print(distr.diff(axis=1).iloc[:, -1])
        print()
        print(distr.diff(axis=1).iloc[:, -1].abs())
        print()
        print(distr.diff(axis=1).iloc[:, -1].abs().sum() / 2)
        print("END\n")
    
    # Compute and return the total variation distance
    return distr.diff(axis=1).iloc[:, -1].abs().sum() / 2


def permutation_simulation(df, N, shuffle_column, cats_column, 
                           significance_level):

    '''
        Run N simulations under the null hypothesis (both distributions are
        from the same population, any observed difference is due to chance)

        Return:
        Test results (str): observed tvd, p-value, assessment of null hypothesis
        results (list): N simulated test statistics under the null hypothesis
    '''

    results = []
    for _ in range(N):
        # Shuffle group column (to_shuffle_column)
        with_shuffled = df.assign(
            shuffled=np.random.permutation(df[shuffle_column])
        )

        # Compute and store the TVD
        tvd = tvd_of_groups(with_shuffled, groups='shuffled', cats=cats_column,
                            show_steps=True if _ == 1 else False)
        results.append(tvd)

        if _ % 1000 == 0:
            print(tvd)
        
    # Calculate p-value and assess null hypothesis
    observed = tvd_of_groups(df, groups=shuffle_column, cats=cats_column)
    pval = (results >= observed).mean()
    if pval < significance_level:
        return f'Obsersed TVD = {np.round(observed, 3)}, P-value = {pval}, Reject null hypothesis at {significance_level}', results
    return f'Obsersed TVD = {np.round(observed, 3)}, P-value = {pval}, Fail to reject null hypothesis at {significance_level}', results
