import pandas as pd
import numpy as np
import math
from DAT.funcs.eda.tools import timeit, validait
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
# logging
import logging
FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(
    level=logging.INFO, 
    format=FORMAT, 
)

class PLOTS():

    def __init__(self):
        pass

    ## validate if a df is emtpy        
    def _validate_is_empty_df(self, df:pd.DataFrame)->bool:
        if len(df) == 0:
            logging.warning("Input df is emtpy.")
            return True
        else:
            return False

    ## collect columns of numerical variables
    def columns_numerical(self, df:pd.DataFrame)->list:
        return df.select_dtypes(include=['float64']).columns.tolist()  

    ## collect columns of categorical variables
    def columns_categorical(self, df:pd.DataFrame)->list:
        return df.select_dtypes(include=['int64', 'object', 'category', 'bool']).columns.tolist() 
    

    """ 1 NUMERICAL VARIABLES """

    ## violin plot
    @staticmethod
    def _dist(df:pd.DataFrame, column:str, ax:"matplotlib axis")->"matplotlib axis":
        # validation
        assert isinstance(df, pd.DataFrame)
        assert column in df.columns.tolist(), f"column '{column}' is not available."
        # plot
        ax = sns.stripplot(x = column, data = df, color = "red", alpha = .35, ax = ax)
        ax = sns.violinplot(x = column, data = df, scale="count", inner="quartile", scale_hue=False, bw=.2, ax = ax)
        ax = sns.boxplot(x = column, data = df, showfliers=True, showbox=False, ax = ax)
        # return axis
        return ax


    ## plot histogram of selected column
    @validait
    ## plot distribution of a column
    def hitogram(self, df:pd.DataFrame, column_num:str, figsize:tuple = (5, 5)):
        # validate
        assert isinstance(df, pd.DataFrame)
        assert isinstance(column_num, str)
        assert column_num in df.columns.tolist()
        if self._validate_is_empty_df(df):
            return None
        # plot
        _, ax = plt.subplots(ncols = 1, nrows = 1, figsize = figsize)
        ax = self._dist(df, column_num, ax)
        plt.show()
    
    
    ## descriptive charts for numerical variables (histograms)
    @validait
    def nums(self, df:pd.DataFrame, 
                   columns:"list or str" = [], 
                   num_plots_per_row:int = 3, 
                   figsize:tuple = (0,0)):
        # validate arguments
        assert isinstance(df, pd.DataFrame)
        if self._validate_is_empty_df(df):
            return None
        assert isinstance(columns, list) or isinstance(columns, str)
        assert isinstance(num_plots_per_row, int) and num_plots_per_row > 0
        assert isinstance(figsize, tuple)
        assert isinstance(figsize[0], int) and isinstance(figsize[1], int)
        assert len(figsize) == 2
        if columns == []:
            columns = self.columns_numerical(df)
        # if columns is only one string    
        if isinstance(columns, str):
            # figsize
            figsize_final = (10, 10) if figsize == (0,0) else figsize
            # number of plots in rows / columns
            nrs = ncs = 1
        # if is a list
        elif isinstance(columns, list):
            # number of columns to be ploted
            ncolumns = len(columns)
            # number of plots in rows / columns
            nrs = math.ceil(ncolumns / num_plots_per_row)
            ncs = num_plots_per_row if ncolumns >= num_plots_per_row else ncolumns 
            # figsize
            if figsize == (0,0):
                figsize_final = (ncs*5, nrs*5)    
            else:
                figsize_final = (figsize[0]*ncs, figsize[1]*nrs)
        else:
            raise
        # create figure and axis    
        _, ax = plt.subplots(ncols = ncs, nrows = nrs, figsize = figsize_final)
        # if axis is an array
        if isinstance(ax, np.ndarray):
            # reshape
            ax = ax.ravel()
            # loop of axis
            for ii, c in enumerate(columns):
                # validate
                assert c in df.columns.tolist(), f"Column {c} is not available."
                # plot in cells
                _ = self._dist(df, c, ax[ii])
        # if axis is only one
        else:
            # plots only one
            _ = self._dist(df, columns[0] if len(columns) == 1 else columns, ax)
        # display plot
        plt.show()
        # return 
        return None