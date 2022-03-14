import math
import numpy as np
import pandas as pd


## estimate order of magnitude
def magnitude(value:float)->int:
    """
    Estimate order of magnitude.
    value -- value to be analized.
    return -- order of magnitude.
    """
    if (value == 0): return 0
    try:
        return int(math.floor(math.log10(abs(value))))
    except:
        return np.nan

    
## estimate most frequent value in a list
def most_frequent(List:list)->"element":
    """
    Estimate most frequent value in a list.
    List -- list to be analized.
    return -- most freq element
    """
    return max(set(List), key = List.count)


## remove outliers of a 1D array according to the Inter Quartile Range (IQR)
def remove_outliers_IQR(v:np.array, verbose:bool = False)->np.array:
    """
    Remove outliers of a 1D array according to the Inter Quartile Range (IQR).
    v -- array of values to be analyzed.
    verbose -- display extra information (default, False).
    return -- array of values after removing outliers.
    """
    # estimate boundary thresholds
    Q1 = np.nanquantile(v,0.25)
    Q3 = np.nanquantile(v,0.75)
    IQR = Q3 - Q1
    t_lower = Q1 - 1.5*IQR
    t_upper = Q3 + 1.5*IQR
    # display
    if verbose:
        print('Thresholds: lower = %.5f / upper = %.5f'%(t_lower, t_upper))
    # remove values outside of these thresholds and return
    v[v < t_lower] = np.nan
    v[v > t_upper] = np.nan
    # return
    return v


## detect multivariate outliers using Local Outlier Factor
def detect_outliers_LOF(X:np.array, n_neighbors:int = 25, n_jobs:int = 2, verbose:bool = False)->np.array:
    """
    Detect multivariate outliers using Local Outlier Factor.
    X -- array of values to be analyzed.
    n_neighbors -- number of neighbors to be used by the algorithm (default, 25).
    n_jobs -- number of jobs to be used (default, 2).
    verbose -- display extra information (default, False).
    return -- array of tagged samples (-1: it is outlier, 1: it is not).
    """
    from sklearn.neighbors import LocalOutlierFactor
    # initialize
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, 
                             algorithm = "auto",
                             metric = 'minkowski',
                             p = 2,
                             contamination="auto",
                             novelty=False,
                             n_jobs = n_jobs
                            )  
    # estimate
    try:
        y_pred = clf.fit(X).predict(X)
    except:
        y_pred = clf.fit_predict(X)
    # display
    if verbose:
        print(f'There are {y_pred[y_pred == -1].shape[0]} outliers from {y_pred.shape[0]}.') 
    # return
    return y_pred


## multivariante outliers detection for a given column of a df
def multivariate_outliers_detection(data:pd.DataFrame, 
                                  col_names: list,
                                  is_remove:bool = True,  
                                  methodology:'function' = detect_outliers_LOF, 
                                  verbose:bool = False)->pd.DataFrame:
    """
    Multivariante outliers detection for a given column of a df.
    data -- dataframe to be analyzed.
    col_names -- columns to be used.
    is_remove -- if removing outliers or just detect (default, True).
    methodology -- function of method to be used to remove / detect outliers (default, detect_outliers_LOF()).
    verbose -- display extra information (default, False).
    return -- df of values without outliers or a mask with detected outliers.
    """
    # validate
    for col in col_names:
        assert col in data.columns.tolist()
    # initialize if just detection
    if not is_remove:
        df_mask = pd.DataFrame(np.zeros(data[col_names].shape, dtype=bool), columns = col_names)
    # outliers detection
    y_pred = methodology(data[col_names].values, n_neighbors = 25, verbose = verbose)  
    # number of outliers
    num_outliers = len(y_pred[y_pred == -1])
    # validate
    if num_outliers > 0:
        # if removing
        if is_remove:
            # add labels
            data['label'] = y_pred
            # filter
            ni = len(data)
            data = data[data.label == 1]
            nf = len(data)
            # display
            if verbose:
                print(f'It was removed {ni - nf} records.')
            # remove unnecessary column
            data.drop('label', axis = 1, inplace = True)
        # if just detection
        else:
            # loop of columns
            for col in col_names:
                # inpute True if it is an outlier
                df_mask[col] = [True if v == -1 else False for v in y_pred]
    # return 
    if is_remove:
        return data
    else:
        return df_mask