import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import eda.tools as tools
from eda.tools import timeit, validait, preparation
import eda.htest as htest
from eda.analysis import analysis_cat_cat, analysis_num_num, analysis_cat_num
import itertools

 
## get basic information of df variables
@validait
def describe_info(data:pd.DataFrame, decimals:int = 2)->pd.DataFrame:
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
@validait
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
@timeit
@validait
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
    # mode
    lmode = list()
    for col in cols_num:
        # collect data
        v = data[col].values
        # remove nans
        v = list(v[~(np.isnan(v))])
        # estimate mode
        if len(v)>1:
            imode = max(v, key=v.count)
        else:
            imode = np.nan
        # append
        lmode.append(imode)
    dfn['mode'] = lmode
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
    # inter-quantil range
    dfn['iqr'] = dfn['75%'].values - dfn['25%'].values
    # format
    dfn['count'] = dfn['count'].astype(int)
    for col in ['mode', 'mean', 'std', 'iqr', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'kurtosis', 'skew']:
        dfn[col] = dfn[col].values.round(decimals=decimals)
    # return
    return dfn[['count', 'mode', 'mean', 'std', 'iqr', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'kurtosis', 'skew', 'uniform','gaussian', 'unimodal']]


## describe function for categorical data
@timeit
@validait
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
@validait
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


## Describe bivariate relationships
@timeit
@validait
def describe_bivariate(data:pd.DataFrame, 
                     only_dependent:bool = False, 
                     size_max_sample:int = None, 
                     is_remove_outliers:bool = True,
                     alpha:float = 0.05, 
                     max_num_rows:int = 5000, 
                     max_size_cats:int = 5,
                     verbose:bool = False)->pd.DataFrame:                       
    """
    Describe bivariate relationships.
    df -- data to be analized.
    only_dependent -- only display relationships with dependeces (default, False).
    size_max_sample -- maximum sample size to apply analysis with whole sample. If this value
                       is not None are used random subsamples although it will not remove bivariate
                       outliers (default, None).
    is_remove_outliers -- Remove or not univariate outliers (default, True).
    alpha -- significance level (default, 0.05).
    max_num_rows -- maximum number of rows allowed without considering a sample (default, 5000).
    max_size_cats -- maximum number of possible values in a categorical variable to be allowed (default, 5).
    return -- results in a table.
    """ 
    # data preparation
    df = preparation(data, max_num_rows, max_size_cats, verbose = True)
    # relationship num - num
    dfnn = analysis_num_num(df, only_dependent = only_dependent, size_max_sample = size_max_sample,
                            is_remove_outliers = is_remove_outliers, alpha = alpha, verbose = verbose)                                                    
    # relationship cat - cat
    dfcc = analysis_cat_cat(df, only_dependent = only_dependent, alpha = alpha, verbose = verbose)
    # relationship cat - num
    dfcn = analysis_cat_num(df, only_dependent = only_dependent, alpha = alpha, 
                            is_remove_outliers = is_remove_outliers, verbose = verbose)
    # append results
    dfbiv = dfnn.copy()
    dfbiv = dfbiv.append(dfcc)
    dfbiv = dfbiv.append(dfcn)
    # return
    return dfbiv


## Describe duplicates for all combinations of columns
@timeit
@validait
def describe_duplicates(data:pd.DataFrame, max_num_rows:int = 5000, max_size_cats:int = 5)->pd.DataFrame:
    """
    Describe duplicates for all combinations of columns.
    data -- data to be analized.
    max_num_rows -- maximum number of rows allowed without considering a sample (default, 5000).
    max_size_cats -- maximum number of possible values in a categorical variable to be allowed (default, 5).
    return -- percent of duplicated records per columns combinations.
    """
    # data preparation
    df = preparation(data, max_num_rows, max_size_cats, verbose = True)
    
    ## all possible combinations of column names

    # list of columns
    columns = df.columns.tolist()
    # possible number of elements for combinations
    rs = np.arange(1,len(columns)+1, 1)
    # initialize
    combs_columns = list()
    # collect all possible combinations for all possible sizes
    for r in rs:
        combs_columns += list(itertools.combinations(columns,r=r))


    ## estimate percent of duplicated records

    # initialize
    records = list()
    # loop of combinations
    for comb in combs_columns:
        # list of columns with a fixed size
        list_columns = list(comb) + ['' for i in range(len(columns) - len(comb))]
        # append
        records.append(list_columns + [100 * (1 - (df.drop_duplicates(list(comb)).shape[0] / df.shape[0]))] )
    # list to df and retunr
    return pd.DataFrame(records, columns = [f'col{i}' for i in range(len(columns))] + ["percent_dupli"]).sort_values("percent_dupli", ascending = False)