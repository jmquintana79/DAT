import pandas as pd
import numpy as np
import DAT.funcs.eda.tools as tools
from DAT.funcs.eda.tools import validait
import DAT.funcs.eda.htest as htest
from DAT.funcs.eda.bhatta_dist import max_bhatta_dist
import itertools
import ppscore as pps
import logging


## Describe relationship between numerical - numerical variables
@validait
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
    # validate
    assert isinstance(df, pd.DataFrame)
    assert isinstance(only_dependent, bool)
    assert size_max_sample is None or (isinstance(size_max_sample, int) and size_max_sample > 0)
    assert isinstance(is_remove_outliers, bool)
    assert alpha > 0 and alpha < 1
    assert isinstance(verbose, bool)
    # initialize
    cols_num_num = ['variable1', 'variable2', 'depend_corr_linear', 'corr_linear', 'corr_non_linear', 'pps12', 'pps21']
    # get names of numeric columns
    cols_num = df.select_dtypes(include=['float64', 'int64']).columns.values
    # validate
    if len(cols_num) == 0 or len(df) == 0:
        return pd.DataFrame(columns = cols_num_num)
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
            try:
                fsample = lambda X: htest.analysis_linear_correlation(X[cnum[0]].values, X[cnum[1]].values, alpha = alpha, return_corr = False, verbose = verbose)
                is_independent = tools.most_frequent([fsample(temp[[cnum[0], cnum[1]]].sample(size_max_sample).dropna()) for i in range(num_times)])
            except:
                logging.warning(f"It was not possible to estimate if is independent or not for a sample of {cnum}.")
                is_independent = np.nan
            # average of linear correlation results for random subsamples
            try:
                fsample = lambda X: htest.analysis_linear_correlation(X[cnum[0]].values, X[cnum[1]].values, alpha = alpha, return_corr = True, verbose = verbose)
                corr_linear = np.nanmean([fsample(temp[[cnum[0], cnum[1]]].sample(size_max_sample).dropna())[0] for i in range(num_times)])
            except:
                logging.warning(f"It was not possible to estimate the linear correlation for a sample of {cnum}.")
                corr_linear = np.nan
            # average of MIC corr for random subsamples  
            try:          
                fsample = lambda X: htest.correlation_mic(X[cnum[0]].values, X[cnum[1]].values)            
                corr_mic = np.nanmean([fsample(temp[[cnum[0], cnum[1]]].sample(size_max_sample).dropna()) for i in range(num_times)])   
            except:
                logging.warning(f"It was not possible to estimate the MIC correlation for a sample of {cnum}.")
                corr_mic = np.nan        
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
            try:
                is_independent = htest.analysis_linear_correlation(data1, data2, alpha = alpha, return_corr = False, verbose = verbose)
            except:
                logging.warning(f"It was not possible to estimate if is independent or not of {cnum}.")
                is_independent = np.nan
            # linear correlation
            try:
                corr_linear, _ = htest.analysis_linear_correlation(data1, data2, alpha = alpha, return_corr = True, verbose = verbose)  
            except:
                logging.warning(f"It was not possible to estimate the linear correlation of {cnum}.")
                corr_linear = np.nan  
            # non linear correlation (Maximal Information Score)
            try:
                corr_mic = htest.correlation_mic(data1, data2)
            except:
                logging.warning(f"It was not possible to estimate the mic correlation of {cnum}.")
                corr_mic = np.nan
            # PPS
            try:
                pps12 = pps.score(temp, cnum[0], cnum[1])["ppscore"]
                pps21 = pps.score(temp, cnum[1], cnum[0])["ppscore"]
            except:
                logging.warning(f"It was not possible to do the pps analysis of {cnum}.")
                pps12 = np.nan
                pps21 = np.nan
        # append
        if only_dependent:
            if  not is_independent:
                num_num.append([cnum[0], cnum[1], not is_independent, corr_linear, corr_mic, pps12, pps21])
            else:
                pass
        else:
            num_num.append([cnum[0], cnum[1], not is_independent, corr_linear, corr_mic, pps12, pps21])
        # cleand
        del temp
    # store in df  
    dfnn = pd.DataFrame(num_num, columns = cols_num_num)
    # format 
    dfnn['corr_linear'] = dfnn['corr_linear'].values.round(decimals=2) 
    dfnn['corr_non_linear'] = dfnn['corr_non_linear'].values.round(decimals=2)
    dfnn['pps12'] = dfnn['pps12'].values.round(decimals=2)
    dfnn['pps21'] = dfnn['pps21'].values.round(decimals=2)
    # return
    return dfnn


## Describe relationship between categorical - categorical variables
@validait
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
    # validate
    assert isinstance(df, pd.DataFrame)
    assert isinstance(only_dependent, bool)
    assert alpha > 0 and alpha < 1
    assert isinstance(verbose, bool)
    # initialize
    cols_cat_cat = ['variable1', 'variable2', 'depend_chi2', 'pps12', 'pps21']
    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'int64', 'category', 'bool']).columns.values
    # validate
    if len(cols_cat) == 0 or len(df) == 0:
        return pd.DataFrame(columns = cols_cat_cat)
    # all combinations between numerical variables
    combs_cat = list(itertools.combinations(cols_cat,r=2))    
    # initialize
    cat_cat = list()
    # loop of combinations
    for ccat in combs_cat[:]:
        # independece test
        try:
            is_independent_chi2 = htest.chi_square(df[ccat[0]].values, df[ccat[1]].values, alpha = alpha, verbose = verbose)
        except:
            logging.warning(f"It was not possible to estimate the chi square test for {ccat}.")
            is_independent_chi2 = np.nan
        # PPS
        try:
            pps12 = pps.score(df, ccat[0], ccat[1])["ppscore"]
            pps21 = pps.score(df, ccat[1], ccat[0])["ppscore"]
        except:
            logging.warning(f"It was not possible to do the pps analysis for {ccat}.")
            pps12 = np.nan
            pps21 = np.nan
        # append
        if only_dependent:
            if  not is_independent_chi2:
                cat_cat.append([ccat[0], ccat[1], not is_independent_chi2, np.around(pps12, decimals=2), np.around(pps21, decimals=2)])
            else:
                pass
        else:
            cat_cat.append([ccat[0], ccat[1], not is_independent_chi2, np.around(pps12, decimals=2), np.around(pps21, decimals=2)])  
    # store in df and return 
    return pd.DataFrame(cat_cat, columns = cols_cat_cat)


## Describe relationship between categorical - numerical variables
@validait
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
    # validate
    assert isinstance(df, pd.DataFrame)
    assert isinstance(only_dependent, bool)
    assert isinstance(is_remove_outliers, bool)
    assert isinstance(verbose, bool)
    assert alpha > 0 and alpha < 1
    # initialize
    cols_cat_num = ['variable1', 'variable2', 'subsamples_diff_dist', 'pps12', 'pps21', 'max_bhatta_dist']
    # validate if empty df
    if len(df) == 0:
        return pd.DataFrame(columns = cols_cat_num)
    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'int64', 'category', 'bool']).columns.values
    # validate
    if len(cols_cat) == 0:
        return pd.DataFrame(columns = cols_cat_num)
    # remove categorical variables with only one class
    cols_cat = [c for c in cols_cat if len(df[c].unique()) > 1]
    # validate
    if len(cols_cat) == 0:
        return pd.DataFrame(columns = cols_cat_num)
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
    for comb_cat_num in combs_cat_num:
        # collect pairs of data and drop nan values
        temp = df[comb_cat_num].dropna()
        # variance analysis
        try:
            is_samples_same_distribution = htest.analysis_variance(temp[comb_cat_num[0]].values,
                                                                temp[comb_cat_num[1]].values,
                                                                alpha = alpha, verbose = verbose) 
        except:
            logging.warning(f"It was not possible to use variance analysis for {comb_cat_num}.")
            is_samples_same_distribution = np.nan
        # PPS
        try:
            pps12 = pps.score(temp, comb_cat_num[0], comb_cat_num[1])["ppscore"]
            pps21 = pps.score(temp, comb_cat_num[1], comb_cat_num[0])["ppscore"]  
        except:
            logging.warning(f"It was not possible to use PPS analysis for {comb_cat_num}.")
            pps12 = np.nan
            pps21 = np.nan
        # Maximum Bhattacharyya distance 
        try:
            max_dist = max_bhatta_dist(temp[comb_cat_num[1]].values, temp[comb_cat_num[0]].values, method='continuous')    
        except:
            logging.warning(f"It was not possible estimate Maximum Bhattacharyya distance for {comb_cat_num}.")
            max_dist = np.nan                                                      
        # append
        if only_dependent:
            if  not is_samples_same_distribution:
                cat_num.append([comb_cat_num[0], comb_cat_num[1], not is_samples_same_distribution, np.around(pps12, decimals=2), np.around(pps21, decimals=2), np.around(max_dist, decimals = 2)])
            else:
                pass
        else:
            cat_num.append([comb_cat_num[0], comb_cat_num[1], not is_samples_same_distribution, np.around(pps12, decimals=2), np.around(pps21, decimals=2), np.around(max_dist, decimals = 2)])      
        # clean
        del temp
    # store in df and return 
    return pd.DataFrame(cat_num, columns = cols_cat_num).sort_values(['variable1', 'variable2']).reset_index(drop=True)
