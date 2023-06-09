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


def means_diff(df, groups, cats):

    """
        groups: the binary column
        cats: the categorical column
    """

    return df.groupby(groups)[cats].diff().abs().iloc[-1]


def permutation_simulation(df, N, shuffle_column, cats_column, 
                           significance_level, quantitative=False, 
                           pval_only=False):

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

        # Compute and store the test statistic
        if quantitative:
            test_stat = means_diff(with_shuffled, groups='shuffled', cats=cats_column)
        else:
            test_stat = tvd_of_groups(with_shuffled, groups='shuffled', cats=cats_column)#,
                            # show_steps=True if _ == 1 else False)

        results.append(test_stat)

        # if _ % 1000 == 0:
        #     print(test_stat)
        
    # Calculate p-value and assess null hypothesis
    if quantitative:
        observed = means_diff(df, groups=shuffle_column, cats=cats_column)
    else:
        observed = tvd_of_groups(df, groups=shuffle_column, cats=cats_column)

    pval = (results >= observed).mean()

    if pval_only:
        return pval

    if pval < significance_level:
        return f'Obsersed test statistic = {np.round(observed, 3)}, P-value = {pval}, Reject null hypothesis at {significance_level}', results
    return f'Obsersed test statistic = {np.round(observed, 3)}, P-value = {pval}, Fail to reject null hypothesis at {significance_level}', results



class FDRController():

    def __init__(self):
        pass

    def test(self, df, N, shuffle_column, quantitative_columns):
        
        p_values = []
        features = []

        for col in df.columns:
            if col == shuffle_column:
                continue

            p_value = permutation_simulation(
                df, N, shuffle_column, col, significance_level=None, 
                quantitative=True if col in quantitative_columns else False,
                pval_only=True
            )
         
            p_values.append(p_value)
            features.append(col)
        
        self.pvalues = p_values
        self.k = len(p_values)
        self.results = pd.DataFrame(data={'feature': features, 'p-values': p_values})
        
        return None
        

    def adjust(self):

        self.results.sort_values(by='p-values', ascending=True, inplace=True)
        sorted_p = self.results['p-values'].tolist()
        print(sorted_p)

        q_values = [(sorted_p[i]*self.k)/(i+1) for i in range(self.k)]

        # Ensure monotonicity for q-values
        adjusted_q = [min(q_values[i], q_values[i+1 if i < self.k-1 else i]) \
            for i in range(self.k)]

        self.qvalues = adjusted_q
        self.results['q-values'] = adjusted_q

        return None

    
    def get_results(self, fdr_threshold=0.05):

        critical_threshold = \
            self.results[self.results['q-values'] <= fdr_threshold]['q-values'].max()

        self.reject = self.results[self.results['q-values'] <= critical_threshold]
        self.critical_threshold = critical_threshold

        return None
