import pandas as pd
import numpy as np
import DAT.funcs.eda.tools as tools
from DAT.funcs.eda.tools import timeit, validait, preparation, cat_encoding


class EDA():
    def __init__(self):
        pass    

    ## get basic information of df variables
    @validait
    def describe_info(self, data:pd.DataFrame, decimals:int = 2)->pd.DataFrame:
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