import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import DAT.funcs.eda.tools as tools
from DAT.funcs.eda.tools import timeit, validait, preparation, cat_encoding
import DAT.funcs.eda.htest as htest

class EDA():
    
    def __init__(self):
        pass    


    ## get basic information of df variables
    @validait
    def info(self, data:pd.DataFrame, decimals:int = 2)->pd.DataFrame:
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
    

    ## Skimpy Summary
    @validait
    def summary(self, data:pd.DataFrame):
        """
        Skimpy (library) summary.
        data -- Data to be summarized.
        """
        from skimpy import skim
        skim(data)


    ## Missing values analysis
    @validait
    def missing(self, data:pd.DataFrame):
        """
        Missing values analysis.
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


    ## Outliers values analysis
    @validait
    def outliers(self, df:pd.DataFrame, num_iqr:float = 1.5):
        """
        Outlier values analysis. The outlier detection tecnique used is with IQR distances.
        df -- df to be analyzed.
        num_iqr -- Number of IQR's to estimate outliers threshold using quantiles (default, 1.5).
        return -- None.
        """
        # get names of numeric columns
        cols_num = df.select_dtypes(include=['float64', 'int64']).columns.values
        # validate
        if len(cols_num) == 0:
            # display
            print('There are not any numerical columns in this dataframe.')
            # return
            return None
        # initialize output df
        temp = pd.DataFrame(np.zeros(df[cols_num].shape, dtype = bool), columns = cols_num)
        # loop of columns
        for c in cols_num:
            # collect data
            v = df[c].values
            # mark outliers
            v = tools.mark_outliers_IQR(v, num_iqr = num_iqr, verbose = False)
            # include in marks in output df
            temp[c] = (v == np.inf)
        # number of outliers found
        num_outliers = temp.sum().sum()
        # validate if there are or not outliers
        if num_outliers == 0:
            print("There are not any outlier in numerical columns.")
        else:
            # replace True values with NaN to be detected as a missing value
            temp.replace(True, np.nan, inplace = True)    
            # launch missing analysis
            self.missing(temp)
        # return
        return None

            
    ## describe function for numeric data
    @timeit
    @validait
    def numeric(self, df:pd.DataFrame, alpha:float = .05, decimals:int = 2, is_remove_outliers:bool = False)->pd.DataFrame:
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
                df[col] = tools.remove_outliers_IQR(df[col], verbose = False)       
        # copy data
        data = df[cols_num].copy()
        # describe
        dfn = data[cols_num].describe(include = 'all', percentiles = [.05, .25, .5, .75, .95]).T
        # add percent of nan values
        #dfn['%nan'] = (data[cols_num].isnull().sum()*100 / len(data)).values
        # mode / mode percent
        lmode = list()
        lmode_percent = list()
        for col in cols_num:
            # collect data
            v = data[col].values
            # remove nans
            v = list(v[~(np.isnan(v))])
            # estimate mode
            if len(v)>1:
                imode = max(v, key=v.count)
                imode_percent = len([iv for iv in v if iv == imode]) * 100. / len(v)
            else:
                imode = np.nan
                imode_percent = np.nan
            # append
            lmode.append(imode)
            lmode_percent.append(imode_percent)
        dfn['mode'] = lmode
        dfn['mode_per'] = lmode_percent
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
        # test if it is unimodal
        dfn['unimodal'] = [htest.test_dip(data[c], alpha = alpha) for c in cols_num]
        # inter-quantil range
        dfn['iqr'] = dfn['75%'].values - dfn['25%'].values
        # normalized std
        dfn['std_norm'] = dfn['std'].values / dfn['mean'].values
        # format
        dfn['count'] = dfn['count'].astype(int)
        for col in ['mode', 'mode_per', 'mean', 'std', 'std_norm', 'iqr', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'kurtosis', 'skew']:
            dfn[col] = dfn[col].values.round(decimals=decimals)
        # return
        return dfn[['count', 'mode', 'mode_per', 'mean', 'std', 'std_norm', 'iqr', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'kurtosis', 'skew', 'uniform','gaussian', 'unimodal']]


    ## describe function for categorical data
    @timeit
    @validait
    def categorical(self, df:pd.DataFrame, max_size_cats:int = 5, alpha:float = .05, decimals:int = 2)->pd.DataFrame:
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