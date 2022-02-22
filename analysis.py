import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import tools
import htest
import itertools

 
## get basic information of df variables
def info(data:pd.DataFrame, decimals:int = 2)->pd.DataFrame:
    """
    Get basic information of df variables.
    data -- df to be analyzed.
    decimals -- precission to be returned (default, 2).
    return -- dataframe with the collected information.
    """
    # copy data
    df = data.copy()
    # get names of numeric columns
    cols_num = df.select_dtypes(include=['float64', 'int64']).columns.values
    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'category', 'bool', 'datetime64[ns]']).columns.values
    # get types information
    dfinfo = pd.DataFrame({'variable':df.dtypes.index, 'types':df.dtypes.values}).set_index('variable')
    # dtype to str
    dfinfo['types'] = dfinfo['types'].astype(str)
    # other types of data in each column
    dfinfo['mixed_types'] = np.ones(len(dfinfo)) * np.nan
    for col in df.columns:
        dfinfo.loc[col,'mixed_types'] = ','.join([str(x).replace('<class','').replace('>','').replace("'","").lstrip().rstrip() for x in df[col].dropna().apply(type).unique()])
    # estimate number of unique values for categorical and numerical variables
    dfinfo['unique'] = np.ones(len(dfinfo)) * np.nan
    for col in df.columns:
        dfinfo.loc[col,'unique'] = len(df[col].dropna().unique())
    dfinfo['unique'] = dfinfo['unique'].astype(int)
    # estimate order of magnitude for numerical variables
    dfinfo['magnitude'] = np.ones(len(dfinfo)) * np.nan
    for col in cols_num:
        # estimate magnitudes for values different to NaN
        magnitudes_values = [tools.magnitude(v) for v in data[col].dropna().values]
        # estimate the most frequent magnitude
        if len(magnitudes_values)>1:
            dfinfo.loc[col,'magnitude'] = tools.most_frequent(magnitudes_values)
        else:
            dfinfo.loc[col,'magnitude'] = np.nan
    # estimate percent of nan values
    dfinfo['%nan'] = (df.isnull().sum()*100 / len(df)).values.round(decimals=decimals)
    # estimate number of records without nan values
    nrecords = list()
    for col in df.columns.tolist():
        nrecords.append(len(df[col].dropna()))
    dfinfo['num_records'] = nrecords
    # return
    return dfinfo.sort_values('types', ascending = True)


## Missing values analysis
def describe_missing(data:pd.DataFrame):
    """
    ## Missing values analysis.
    data -- df to be analized.
    return -- None.
    """
    # estimate number of missing values
    ntotal_missing = data.isnull().sum().sum()
    # validate
    if ntotal_missing == 0:
        print('There are not any missing values.')
    else:
        import matplotlib.pyplot as plt
        import missingno as msno
        # missing values graph
        msno.matrix(data)
        plt.show()
        # counting non-missing values
        msno.bar(data)
        plt.show()
        # correlation between missing values
        msno.heatmap(data)
        plt.show()
    
    
## describe function for numeric data
def describe_numeric(df:pd.DataFrame, alpha:float = .05, decimals:int = 2, is_remove_outliers:bool = False)->pd.DataFrame:
    """
    Describe tool for numeric data.
    df -- dataframe with data to be described.
    alpha -- significance level (default, 0.05).
    decimals -- precission to be returned (default, 2).
    is_remove_outliers -- Removing univariate outliers or not (default, False).
    return -- describe df.
    """
    # get names of numeric columns
    cols_num = df.select_dtypes(include=['float64', 'int64']).columns.values
    # remove outliers
    if is_remove_outliers:
        for col in cols_num:
            df[col] = tools.remove_outliers_IQR(df[col], verbose = verbose)       
    # copy data
    data = df[cols_num].copy()
    # describe
    dfn = data[cols_num].describe(include = 'all', percentiles = [.05, .25, .5, .75, .95]).T
    # add percent of nan values
    #dfn['%nan'] = (data[cols_num].isnull().sum()*100 / len(data)).values
    # kurtosis
    dfn['kurtosis'] = kurtosis(data[cols_num], nan_policy = 'omit')
    # skew
    dfn['skew'] = skew(data[cols_num], nan_policy = 'omit')
    # test if it is uniform
    dfn['uniform'] = [htest.test_uniform_num(data[c], alpha = alpha) for c in cols_num]
    # test if it is gaussian
    if len(data) >= 5000:
        dfn['gaussian'] = [htest.test_anderson(data[c], alpha = alpha) for c in cols_num]
    else:
        dfn['gaussian'] = [htest.test_shapiro(data[c], alpha = alpha) for c in cols_num]
    # test if it is gaussian
    dfn['unimodal'] = [htest.test_dip(data[c], alpha = alpha) for c in cols_num]
    # format
    dfn['count'] = dfn['count'].astype(int)
    for col in ['mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'kurtosis', 'skew']:
        dfn[col] = dfn[col].values.round(decimals=decimals)
    # return
    return dfn


## describe function for categorical data
def describe_categorical(df:pd.DataFrame, max_size_cats:int = 5, alpha:float = .05, decimals:int = 2)->pd.DataFrame:
    """
    Describe tool for categorical data.
    df -- dataframe with data to be described.
    max_size_cats -- maximum number of categories to be showed.
    alpha -- significance level (default, 0.05).
    decimals -- precission to be returned (default, 2).
    return -- describe df.
    """
    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'int64', 'category', 'bool']).columns.values
    # copy data
    data = df[cols_cat].copy()
    # integer to categorical
    for col in data.select_dtypes(include=['int64']).columns.values:
        data[col] = pd.Categorical(data[col])
    # describe
    dfc = data[cols_cat].describe(include = 'all').T[['count', 'unique']]
    # add percent of nan values
    #dfc['%nan'] = (data[cols_cat].isnull().sum()*100 / len(data)).values.round(decimals=decimals)
    # test if it is uniform
    dfc['uniform'] = [htest.test_uniform_cat(data[c].dropna().values, alpha = alpha) for c in cols_cat]
    
    
    ## add categories percents

    # set columns
    col_temp = ['var'] + ['value{}'.format(i) for i in range(max_size_cats)] + ['%value{}'.format(i) for i in range(max_size_cats)]
    # initialize
    values_temp = list()
    # loop of variables
    for col in cols_cat:
        # count categories
        temp = data[col].value_counts(normalize=True,sort=True,ascending=False,dropna=True)*100.
        # collect values and names
        c = temp.index.values
        v = temp.values.round(decimals=decimals)   
        # resize
        if len(v) > max_size_cats:
            v = np.append(v[:max_size_cats-1], np.sum(v[-(max_size_cats):]).round(decimals=decimals))
            c = np.append(c[:max_size_cats-1], 'others')
        else:
            v = np.pad(v.astype(str),(0, max_size_cats-len(v)), 'empty')
            c = np.pad(c.astype(str),(0, max_size_cats-len(c)), 'empty')
        # append    
        values_temp.append([col] + list(np.append(c,v)))
    # add new information
    dfc = pd.concat([dfc, pd.DataFrame(values_temp, columns = col_temp).set_index('var')], axis = 1)
    # percent of values to str
    cols_per = [c for c in dfc.columns if "%value" in c]
    for col in cols_per:
        dfc[col] = dfc[col].astype(str)    
    # return
    return dfc    


## describe function for datetime data
def describe_datetime(df:pd.DataFrame, decimals:int = 2)->pd.DataFrame:
    """
    Describe tool for datetime data.
    df -- dataframe with data to be described.
    decimals -- precission to be returned (default, 2).
    return -- describe df.
    """    
    # get names of categorical columns
    cols_dt = df.select_dtypes(include=['datetime64[ns]']).columns.values
    # copy data
    data = df[cols_dt].copy()
    # main description
    dfdt = data[cols_dt].describe().T
    # rename columns
    dfdt.rename(columns = {'top':'most_frequent', 'freq':'num_most_frequent'}, inplace = True)
    # remove most frequent information when count and unique are equal
    dfdt.loc[dfdt[dfdt['count'] == dfdt['unique']].index.tolist(), ['most_frequent', 'num_most_frequent']] = np.nan
    ## timedelta analysis
    # initialize
    td_cols = list()
    per_td_cols = list()
    num_td_cols = list()
    # loop of dt columms
    for col in dfdt.index.tolist():
        # counting timedelta
        temp = data[[col]].diff().dropna()[[col]].value_counts(normalize=True,sort=True,ascending=False)*100
        # get tds
        td = temp.index
        # percent of tds
        per_td = temp.values
        # num of tds
        num_td = len(td)
        # append
        td_cols.append(td[0])
        per_td_cols.append(per_td[0])
        num_td_cols.append(num_td)
        # clean
        del temp
    # store
    dfdt['most_frequent_td'] = td_cols
    dfdt['%most_frequent_td'] = per_td_cols
    dfdt['num_td'] = num_td_cols
    # format float
    dfdt['%most_frequent_td'] = dfdt['%most_frequent_td'].values.round(decimals=decimals) 
    # return
    return dfdt


## Describe relationship between numerical - numerical variables
def describe_num_num(df:pd.DataFrame, 
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
def describe_cat_cat(df:pd.DataFrame, 
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
def describe_cat_num(df:pd.DataFrame, 
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


## Describe bivariate relationships
def describe_bivariate(data:pd.DataFrame, 
                     only_dependent:bool = False, 
                     size_max_sample:int = None, 
                     is_remove_outliers:bool = True,
                     alpha:float = 0.05, 
                     verbose:bool = False)->pd.DataFrame:                       
    """
    Describe bivariate relationships.
    df -- data to be analized.
    only_dependent -- only display relationships with dependeces (default, False).
    size_max_sample -- maximum sample size to apply analysis with whole sample. If this value
                       is not None are used random subsamples although it will not remove bivariate
                       outliers (default, None).
    is_remove_outliers -- Remove or not univariate outliers (default, True).
    return -- results in a table.
    """ 
    # copy data
    df = data.copy()
    # relationship num - num
    dfnn = describe_num_num(df, only_dependent = only_dependent, size_max_sample = size_max_sample,
                            is_remove_outliers = is_remove_outliers, alpha = alpha, verbose = verbose)                                                    
    # relationship cat - cat
    dfcc = describe_cat_cat(df, only_dependent = only_dependent, alpha = alpha, verbose = verbose)
    # relationship cat - num
    dfcn = describe_cat_num(df, only_dependent = only_dependent, alpha = alpha, 
                            is_remove_outliers = is_remove_outliers, verbose = verbose)
    # append results
    dfbiv = dfnn.copy()
    dfbiv = dfbiv.append(dfcc)
    dfbiv = dfbiv.append(dfcn)
    # return
    return dfbiv