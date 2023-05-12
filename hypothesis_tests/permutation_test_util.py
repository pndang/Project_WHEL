
def tvd_of_groups(df, groups, cats):

    '''
        groups: the binary column
        cats: the categorical column
    '''

    # Get count of each pair of catogory and group
    counts = df.pivot_table(index=cats, columns=groups, aggfunc='size')

    # Normalize each column
    distr = counts / counts.sum()

    # Compute and return the total variation distance
    return distr.diff(axis=1).iloc[:, -1].abs().sum() / 2

def permutation_simulation(df, N, shuffle_column, cats_column, 
                           significance_level):

    '''
        Run N simulations under the null hypothesis (both distributions are
        from the same population, any observed difference is due to chance)
    '''

    results = []
    for _ in range(N):
        # Shuffle group column (to_shuffle_column)
        with_shuffled = df.assign(
            shuffled=np.random.permutation(df[to_shuffle_column])
        )

        # Compute and store the TVD
        tvd = tvd_of_groups(df, groups='shuffled', cats=cats_column)
        results.append(tvd)

        # Calculate p-value and assess null hypothesis
        observed = tvd_of_groups(df, groups=shuffle_column, cats=cats_column)
        pval = (results >= observed).mean()
        if pval < significance_level:
            return f'P-value = {pval}, Reject null hypothesis at {significance_level}'
        return f'P-value = {pval}, Fail to reject null hypothesis at {significance_level}'


if __name__ == '__main__':
    import pandas as pd
    import numpy as np