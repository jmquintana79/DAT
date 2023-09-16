import pandas as pd
import numpy as np
import math
from DAT.funcs.eda.tools import timeit, validait
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt


class PLOTS():

    def __init__(self):
        pass

    @property
    def columns_numerical(self, df:pd.DataFrame)->list:
        return df.select_dtypes(include=['float64']).columns.tolist()  


    @property
    def columns_categorical(self, df:pd.DataFrame)->list:
        return df.select_dtypes(include=['int64', 'object', 'category', 'bool']).columns.tolist() 
    

    @staticmethod
    def _dist(df:pd.DataFrame, column:str, ax:"matplotlib axis")->"matplotlib axis":
        # validation
        assert column in df.columns.tolist(), f"column '{column}' is not available."
        # plot
        ax = sns.stripplot(x = column, data = df, color = "red", alpha = .35, ax = ax)
        ax = sns.violinplot(x = column, data = df, scale="count", inner="quartile", scale_hue=False, bw=.2, ax = ax)
        ax = sns.boxplot(x = column, data = df, showfliers=True, showbox=False, ax = ax)
        # return axis
        return ax


    """ 1 NUMERICAL VARIABLEV"""

    ## plot histogram of selected column
    @validait
    ## plot distribution of a column
    def hitogram(self, df:pd.DataFrame, column_num:str, figsize:tuple = (5, 5)):
        assert column_num in df.columns.tolist()
        _, ax = plt.subplots(ncols = 1, nrows = 1, figsize = figsize)
        ax = self._dist(df, column_num, ax)
        plt.show()
    
    
    # plot histogram of selected columns
    @validait
    def histograms(self, df:pd.DataFrame, columns:"list or str", num_plots_per_row:int = 3):
        # if columns is only one string    
        if isinstance(columns, str):
            # figsize
            figsize = (10, 10)
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
            figsize = (ncs*5, nrs*5)    
        else:
            raise
        # create figure and axis    
        _, ax = plt.subplots(ncols = ncs, nrows = nrs, figsize = figsize)
        # if axis is an array
        if isinstance(ax, np.ndarray):
            # reshape
            ax = ax.ravel()
            # loop of axis
            for ii, c in enumerate(columns):
                # plot in cells
                _ = self._dist(df, c, ax[ii])
        # if axis is only one
        else:
            # plots only one
            _ = self._dist(df, columns[0] if len(columns) == 1 else columns, ax)
        # display plot
        plt.show()