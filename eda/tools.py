import math
import numpy as np
import pandas as pd
from functools import wraps
import time


## decorator: time spent estimation
def timeit(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('[info] Function "{}()" finished in {:.2f} {}.'.format(
            func.__name__, elapsed_time if elapsed_time < 60 else elapsed_time / 60., "seconds" if elapsed_time < 60 else "minutes"))
        return result
    return new_func


## decorator: handle errors (no impact on help() function)
def validait(func):
    @wraps(func)
    def handle_error(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            descr = '[error] Function "%s()": %s'%(func.__name__,str(e))
            print(descr)
    return handle_error


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
@validait
def most_frequent(List:list)->"element":
    """
    Estimate most frequent value in a list.
    List -- list to be analized.
    return -- most freq element
    """
    return max(set(List), key = List.count)


## remove outliers of a 1D array according to the Inter Quartile Range (IQR)
@validait
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


## remove outliers of a 1D array according to the Inter Quartile Range (IQR)
def mark_outliers_IQR(v:np.array, verbose:bool = False)->np.array:
    """
    Remove outliers of a 1D array according to the Inter Quartile Range (IQR).
    v -- array of values to be analyzed.
    verbose -- display extra information (default, False).
    return -- array of values after removing outliers.
    """
    # map infinite value
    v[v==np.inf] = np.nan
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
    v[v < t_lower] = np.inf
    v[v > t_upper] = np.inf
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
@validait
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
    
    
## data preparation previous to be analized
@validait
def preparation(df:pd.DataFrame, max_num_rows:int = 5000, max_size_cats:int = 5, verbose:bool = True)->pd.DataFrame:
    """
    Data preparation previous to be analized.
    df -- data to be prepared.
    max_num_rows -- maximum number of rows allowed without considering a sample (default, 5000).
    max_size_cats -- maximum number of possible values in a categorical variable to be allowed (default, 5).
    verbose -- display extra information (default, True).
    return -- processed data.
    """

    ## get random sample if there are too much data

    # validate
    if len(df) > max_num_rows:
        # get a random sample
        df = df.sample(max_num_rows, random_state = 8)
        # display
        if verbose:
            print(f"[warning] It has taken a random sample with {len(df)} records.")


    ## get simplified categorical columns reducing the number of possible values

    # get names of categorical columns
    cols_cat = df.select_dtypes(include=['object', 'int64', 'category', 'bool']).columns.values
    # validate
    if len(cols_cat) > 0:
        # loop of variables
        for col in cols_cat:
            # count categories
            temp = df[col].value_counts(normalize=True,sort=True,ascending=False,dropna=True)
            # collect names order by frequency
            c = temp.index.values
            # resize
            if len(c) > max_size_cats:
                # get columns to be replace by "other"
                #cols_to_keep = list(c[:max_size_cats-1])
                cols_to_replace = list(c[max_size_cats-1:])    
                # replace less frequent columns by "others"
                df[col] = df[col].apply(lambda x: "other" if x in cols_to_replace else x)
                # validate
                if len(df[col].dropna().unique()) != max_size_cats:
                    print(f"[error] something was wrong in column '{col}' reducing its possible values.")
                else:
                    if verbose:
                        print(f"[info] it was simplified the categorical variable '{col}'.")
            else:
                pass
            # clean
            del temp
    # return 
    return df


## categorical variables encoding with numerical values, included NaN values (with -1)
def cat_encoding(data):
    """
    Categorical variables encoding with numerical values, included NaN values (with -1).
    data -- input dataframe.
    return -- encoded input dataframe.
    """
    # copy 
    df = data.copy()
    # loop of transformation
    for c in df.columns.tolist():
        df[c] = df[c].astype('category').cat.codes.values
    # return
    return df