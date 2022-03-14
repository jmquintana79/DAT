import pandas as pd
import numpy as np
import eda.tools as tools
import eda.htest as htest
import itertools


## Describe relationship between numerical - numerical variables
def analysis_num_num(df:pd.DataFrame, 
                     only_dependent:bool = False, 
                     size_max_sample:int = None, 
                     is_remove_outliers:bool = True, 
                     alpha:float = 0.05, 
                     verbose:bool = False)->pd.DataFrame:
    """
    Describe relationship between numerical - numerical variables.
    df -- data to be analized.
    only_dependent -- only display relationships with dependeces (default, False).
    size_max_sample -- maximum sample size to apply analysis with whole sample. If this value
                       is not None are used random subsamples although it will not remove bivariate
                       outliers (default, None).
    is_remove_outliers -- Remove or not univariate outliers (default, True).
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- results in a table.
    """
    
    # get names of numeric columns
    cols_num = df.select_dtypes(include=['float64', 'int64']).columns.values
    # remove univariate outliers
    if is_remove_outliers:
        for col in cols_num:
            df[col] = tools.remove_outliers_IQR(df[col], verbose = verbose)    
    # all combinations between numerical variables
    combs_num = list(itertools.combinations(cols_num,r=2))
    # initialize
    num_num = list()
    # loop of combinations
    for cnum in combs_num[:]:
        # collect data
        temp = df[[cnum[0], cnum[1]]].copy()
        # number of rows
        nrows = len(temp)
        # if it is selected analysis by subsamples
        if not size_max_sample is None:

            # number of times to apply test in subsamples
            num_times = int(nrows / size_max_sample) + 10
            
            # most frequent result of independece test for random subsamples
            fsample = lambda X: htest.analysis_linear_correlation(X[cnum[0]].values, X[cnum[1]].values, alpha = alpha, return_corr = False, verbose = verbose)
            is_independent = tools.most_frequent([fsample(temp[[cnum[0], cnum[1]]].sample(size_max_sample).dropna()) for i in range(num_times)])
            
            # average of linear correlation results for random subsamples
            fsample = lambda X: htest.analysis_linear_correlation(X[cnum[0]].values, X[cnum[1]].values, alpha = alpha, return_corr = True, verbose = verbose)
            corr_linear = np.nanmean([fsample(temp[[cnum[0], cnum[1]]].sample(size_max_sample).dropna())[0] for i in range(num_times)])

            # average of MIC corr for random subsamples            
            fsample = lambda X: htest.correlation_mic(X[cnum[0]].values, X[cnum[1]].values)            
            corr_mic = np.nanmean([fsample(temp[[cnum[0], cnum[1]]].sample(size_max_sample).dropna()) for i in range(num_times)])           
        else:
            # remove bivariate outliers
            if is_remove_outliers:
                temp = tools.multivariate_outliers_detection(temp.dropna(), [cnum[0], cnum[1]], verbose = verbose)
            # remove nan values
            temp = temp.dropna()
            # collect data
            data1 = temp[cnum[0]].values
            data2 = temp[cnum[1]].values
            # independence test
            is_independent = htest.analysis_linear_correlation(data1, data2, alpha = alpha, return_corr = False, verbose = verbose)
            # linear correlation
            corr_linear, _ = htest.analysis_linear_correlation(data1, data2, alpha = alpha, return_corr = True, verbose = verbose)    
            # non linear correlation (Maximal Information Score)
            corr_mic = htest.correlation_mic(data1, data2)
        # append
        if only_dependent:
            if  not is_independent_spearman:
                num_num.append([cnum[0], cnum[1], not is_independent, corr_linear, corr_mic])
            else:
                pass
        else:
            num_num.append([cnum[0], cnum[1], not is_independent, corr_linear, corr_mic])
        # cleand
        del temp
    # store in df  
    cols_num_num = ['variable1', 'variable2', 'depend_corr_linear', 'corr_linear', 'corr_non_linear']
    dfnn = pd.DataFrame(num_num, columns = cols_num_num)
    # format 
    dfnn['corr_linear'] = dfnn['corr_linear'].values.round(decimals=2) 
    dfnn['corr_non_linear'] = dfnn['corr_non_linear'].values.round(decimals=2)
    # return
    return dfnn


## Describe relationship between categorical - categorical variables
def analysis_cat_cat(df:pd.DataFrame, 
                     only_dependent:bool = False,
                     alpha:float = 0.05, 
                     verbose:bool = False)->pd.DataFrame:                     
    """
    Describe relationship between categorical - categorical variables.
    df -- data to be analized.
    only_dependent -- only display relationships with dependeces (default, False).
    alpha -- Significance level (default, 0.05).
    is_remove_outliers -- Remove or not univariate outliers (default, True).
    verbose -- Display extra information (default, False).    
    return -- results in a table.
    """
    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'int64', 'category', 'bool']).columns.values
    # all combinations between numerical variables
    combs_cat = list(itertools.combinations(cols_cat,r=2))    
    # initialize
    cat_cat = list()
    # loop of combinations
    for ccat in combs_cat[:]:
        # independece test
        is_independent_chi2 = htest.chi_square(df[ccat[0]].values, df[ccat[1]].values, alpha = alpha, verbose = verbose)
        # append
        if only_dependent:
            if  not is_independent_chi2:
                cat_cat.append([ccat[0], ccat[1], not is_independent_chi2])
            else:
                pass
        else:
            cat_cat.append([ccat[0], ccat[1], not is_independent_chi2])  
    # store in df and return 
    cols_cat_cat = ['variable1', 'variable2', 'depend_chi2']
    return pd.DataFrame(cat_cat, columns = cols_cat_cat)


## Describe relationship between categorical - numerical variables
def analysis_cat_num(df:pd.DataFrame, 
                     only_dependent:bool = False,
                     alpha:float = 0.05, 
                     is_remove_outliers:bool = True, 
                     verbose:bool = False)->pd.DataFrame:
    """
    Describe relationship between categorical - numerical variables.
    df -- data to be analized.
    only_dependent -- only display relationships with dependeces (default, False).
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).    
    return -- results in a table.
    """    
    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'int64', 'category', 'bool']).columns.values
    # remove categorical variables with only one class
    cols_cat = [c for c in cols_cat if len(df[c].unique()) > 1]
    # get names of numeric columns
    cols_num = df.select_dtypes(include=['float64', 'int64']).columns.values    
    # remove outliers
    if is_remove_outliers:
        for col in cols_num:
            df[col] = tools.remove_outliers_IQR(df[col], verbose = verbose)  
    # combinations between categorical and numerical variables
    combs_cat_num = [[col_cat, col_num] for col_num in cols_num for col_cat in cols_cat]
    combs_cat_num = [[col_cat, col_num] for col_cat, col_num in combs_cat_num if col_cat != col_num]
    # initialize
    cat_num = list()
    # loop of combinations
    for comb_cat_num in combs_cat_num[:]:
        # collect pairs of data and drop nan values
        temp = df[comb_cat_num].dropna()
        # variance analysis
        is_samples_same_distribution = htest.analysis_variance(temp[comb_cat_num[0]].values,
                                                               temp[comb_cat_num[1]].values,
                                                               alpha = alpha, verbose = verbose)       
        # append
        if only_dependent:
            if  not is_samples_same_distribution:
                cat_num.append([comb_cat_num[0], comb_cat_num[1], not is_samples_same_distribution])
            else:
                pass
        else:
            cat_num.append([comb_cat_num[0], comb_cat_num[1], not is_samples_same_distribution])      
        # clean
        del temp#, groups, data_groups
    # store in df and return 
    cols_cat_num = ['variable1', 'variable2', 'subsamples_diff_dist']
    return pd.DataFrame(cat_num, columns = cols_cat_num).sort_values(['variable1', 'variable2']).reset_index(drop=True)
