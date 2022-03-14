import numpy as np
import pandas as pd
from minepy import MINE


## test if it is an array is a uniform distribution for numeric data
def test_uniform_num(data:np.array, alpha:float = .05, verbose:bool = False)->bool:
    """
    Test if it is an array is a uniform distribution for numeric data (Kolmogorov–Smirnov test).
    data -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import uniform, ks_2samp
    # remove nan values
    data = data[~(np.isnan(data))]
    # get extremes
    dismin=np.amin(data)
    dismax=np.amax(data)
    try:
        # build an uniform distribution
        T = uniform(dismin,dismax-dismin).rvs(data.shape[0])
        # test if both have same distribution
        stat, p = ks_2samp(data, T)
    except:
        # manage exception
        if verbose:
            print('[error-Uniform num] It was not possible get a result.')
        return np.nan
            
    # display
    if verbose:
        print('stat=%.3f, p=%.3f' % (stat, p))
    # results
    if p > alpha:
        # display
        if verbose:
            print('Probably is Uniform')
        # return
        return True
    else:
        # display
        if verbose:
            print('Probably is not Uniform') 
        # return
        return False

    
## test if it is an array is a uniform distribution for categorial data    
def test_uniform_cat(data:np.array, alpha:float = .05, verbose:bool = False)->bool:
    """
    Test if it is an array is a uniform distribution for categorial data (Kolmogorov–Smirnov test).
    data -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import ks_2samp
    try:
        # number of categories
        cats = np.unique(data)
        # resize if data is too large
        if len(data)>1000 and len(cats)*1000 < len(data):
            data = np.random.choice(data, size = len(cats)*1000)    
        # create artificial data with uniform distribution
        data_uniform = np.random.choice(cats, size = len(data), p = np.ones(len(cats)) / len(cats))
        # cat to num of input data
        temp = list()
        for ii, ic in enumerate(cats):
            temp += list(np.ones(len(data[data==ic])) * ii)
        data_modif = np.array(temp)
        # cat to num of artificial data
        temp = list()
        for ii, ic in enumerate(cats):
            temp += list(np.ones(len(data_uniform[data_uniform==ic])) * ii)
        data_uniform_modif = np.array(temp)
        # test
        stat, p = ks_2samp(data, data_uniform)
    except:
        # manage exception
        if verbose:
            print('[error-Uniform cat] It was not possible get a result.')
        return np.nan
    if verbose:
        print('stat=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        if verbose:
            print('Probably is Uniform')
        return True
    else:
        if verbose:
            print('Probably is not Uniform')   
        return False

    
## test if distribution is Gaussian (for N < 5000)
def test_shapiro(data:np.array, alpha:float = .05, verbose:bool = False)->bool:
    """
    Test if distribution is Gaussian (Shapiro–Wilk test).
    data -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import shapiro
    # remove nan values
    data = data[~(np.isnan(data))]
    try:
        # test
        stat, p = shapiro(data)
    except:
        # manage exception
        if verbose:
            print('[error-Shapiro] It was not possible get a result.')
        return np.nan        
    # display
    if verbose:
        print('stat=%.3f, p=%.3f' % (stat, p))
    # results
    if p > alpha:
        # display
        if verbose:
            print('Probably Gaussian')
        # return
        return True
    else:
        # display
        if verbose:
            print('Probably not Gaussian')
        # return
        return False

    
## test if distribution is Gaussian (large sample size) 
def test_anderson(data:np.array, alpha:float = .05, verbose:bool = False)->bool:
    """
    Test if distribution is Gaussian (Agostino and Pearson’s test).
    data -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import normaltest
    # remove nan values
    data = data[~(np.isnan(data))]
    try:
        # test
        stat, p = normaltest(data)
    except:
        # manage exception
        if verbose:
            print('[error-Anderson] It was not possible get a result.')
        return np.nan        
    # display
    if verbose:
        print('stat=%.3f, p=%.3f' % (stat, p))
    # results
    if p > alpha:
        # display
        if verbose:
            print('Probably Gaussian')
        # return
        return True
    else:
        # display
        if verbose:
            print('Probably not Gaussian')
        # return
        return False
            
            
## test if is unimodal
def test_dip(data:np.array, alpha:float = 0.05, verbose:bool = False)->bool:
    """
    Test if is unimodal (Hartigan’s test).
    data -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    import unidip.dip as dip
    data = data[~(np.isnan(data))]
    try:
        # sort data
        data = np.msort(data)
        # test
        stat, p, _ = dip.diptst(data)
    except:
        # manage exception
        if verbose:
            print('[error-Unimodal] It was not possible get a result.')
        return np.nan        
    if p is None:
        return np.nan
    # display
    if verbose:
        print('stat=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        if verbose:
            print('Probably unimodal')
        return True
    else:
        if verbose:
            print('Probably not unimodal.')
        return False


## Test if two numerical variables are independents (Pearson's)
def correlation_pearson(data1:np.array, 
                        data2:np.array, 
                        alpha:float = 0.05, 
                        return_corr:bool = False, 
                        verbose:bool = False)->bool:
    """
    Test if two numerical variables are independents (Pearson's).
    
    data1, date2 -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    return_corr -- If is True, return correlation value and his p-value (default, False).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import pearsonr
    try:
        # calculate Pearson's correlation
        corr, p = pearsonr(data1, data2)
    except:
        # manage exception
        if verbose:
            print('[error-Pearson] It was not possible get a result.')
        if return_corr:
            return np.nan, np.nan
        else:
            return np.nan    
    # display
    if verbose:
        print('corr=%.3f, p=%.3f' % (corr, p))
    # check result and return
    if p > alpha:
        # display
        if verbose:
            print('Probably independent')
        # return
        if return_corr:
            return corr, p
        else:
            return True
    else:
        # display
        if verbose:
            print('Probably dependent')
        # return
        if return_corr:
            return corr, p
        else:
            return False
    
    
## Test if two numerical variables are independents (Spearman's)
def correlation_spearman(data1:np.array, 
                         data2:np.array, 
                         alpha:float = 0.05, 
                         return_corr:bool = False, 
                         verbose:bool = False)->bool:
    """
    Test if two numerical variables are independents (Spearman's).
    
    data1, date2 -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    return_corr -- If is True, return correlation value and his p-value (default, False).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import spearmanr
    try:
        # calculate Pearson's correlation
        corr, p = spearmanr(data1, data2)
    except:
        # manage exception
        if verbose:
            print('[error-Spearman] It was not possible get a result.')
        if return_corr:
            return np.nan, np.nan
        else:
            return np.nan        
    # display
    if verbose:
        print('corr=%.3f, p=%.3f' % (corr, p))
    # check result and return
    if p > alpha:
        # display
        if verbose:
            print('Probably independent')
        # return
        if return_corr:
            return corr, p
        else:
            return True
    else:
        # display
        if verbose:
            print('Probably dependent')
        # return
        if return_corr:
            return corr, p
        else:
            return False

        
## Calculate Kendall’s tau, a correlation measure for ordinal data
def correlation_kendalltau(data1:np.array, 
                           data2:np.array, 
                           alpha:float = 0.05, 
                           return_corr:bool = False, 
                           verbose:bool = False)->float:
    """
    Test if two numerical/ordinal variables are independents (Kendall’s tau).
    
    data1, date2 -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    return_corr -- If is True, return correlation value and his p-value (default, False).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import kendalltau
    # test
    try:
        corr, p = kendalltau(data1, data2)
    except:
        # manage exception
        if verbose:
            print('[error-Kendall Tau] It was not possible get a result.')
        if return_corr:
            return np.nan, np.nan
        else:
            return np.nan            
    # visualize
    if verbose:
        print('corr=%.3f, p=%.5f' % (corr, p))
    # check result and return
    if p > alpha:
        # display
        if verbose:
            print('Probably independent')
        # return
        if return_corr:
            return corr, p
        else:
            return True
    else:
        # display
        if verbose:
            print('Probably dependent')
        # return
        if return_corr:
            return corr, p
        else:
            return False

        
## Linear correlation analysis to test independence for numerical / ordinal variables
def analysis_linear_correlation(data1:np.array, 
                                data2:np.array,
                                alpha:float = .05, 
                                return_corr:bool = True, 
                                verbose:bool = False)->bool:
    """
    ## Linear correlation analysis to test independence for numerical / ordinal variables.
    
    data1, date2 -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    return_corr -- If is True, return correlation value and his p-value (default, False).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """

    # get types
    type1 = data1.dtype
    type2 = data2.dtype
    # get size
    n = len(data1)

    # ord - ord
    if type1 == "int64" and type2 == "int64":
        # number of categories
        ncat1 = len(np.unique(data1))
        ncat2 = len(np.unique(data2))
        # analysis
        if ncat1 >= 5 and ncat2 >= 5:
            result = correlation_spearman(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)
        else:
            result = correlation_kendalltau(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)

    # num - num
    if type1 == "float64" and type2 == "float64":
        # test if variables are gaussian
        if n >= 5000:
            is_normal1 = test_anderson(data1, alpha = alpha)
            is_normal2 = test_anderson(data2, alpha = alpha)
        else:
            is_normal1 = test_shapiro(data1, alpha = alpha)
            is_normal2 = test_shapiro(data2, alpha = alpha)
        # analysis
        if n >= 100:
            result = correlation_pearson(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)
        else:
            if is_normal1 and is_normal2:
                result = correlation_pearson(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)
            else:
                result = correlation_spearman(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)

    # num - ord
    if (type1 == "float64" and type2 == "int64") or (type1 == "int64" and type2 == "float64"):
        # number of categories
        if type1 == "int64":
            ncat = len(np.unique(data1))
        else:
            ncat = len(np.unique(data2))
        # analysis
        if ncat < 5:
            result = correlation_kendalltau(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)
        else:
            if n >= 100:
                result = correlation_pearson(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)
            else:
                result = correlation_spearman(data1, data2, alpha = alpha, return_corr = return_corr, verbose = verbose)

    # return
    return result

    
## Maximal Information Score to estimate non-linear correlation
def correlation_mic(x:np.array, y:np.array)->float:
    """
    Maximal Information Score to estimate non-linear correlation.
    x -- first array to be analyzed.
    y -- second array to be analyzed.
    return -- MIC.
    
    > DOC: https://minepy.readthedocs.io/en/latest/python.html
    """
    try:
        mine = MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(x, y)
        return mine.mic()
    except:
        # manage exception
        if verbose:
            print('[error-MIC] It was not possible get a result.')
        return np.nan        


## Test if two categorical variables are independents (Chi-Squared Test)
def chi_square(data1:np.array, data2:np.array, alpha:float = 0.05, verbose:bool = False)->bool:
    """
    Test if two categorical variables are independents (Chi-Squared Test).
    data1, date2 -- 1D data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    from scipy.stats import chi2_contingency
    try:
        # contingence table
        table = pd.crosstab(data1, data2, margins = False).values
        # test
        stat, p, dof, expected = chi2_contingency(table)
    except:
        # manage exception
        if verbose:
            print('[error-chi square] It was not possible get a result.')
        return np.nan            
    # display
    if verbose:
        print('stat=%.3f, p=%.3f' % (stat, p))
    # results
    if p > alpha:
        # display
        if verbose:
            print('Probably independent')
        # return 
        return True
    else:
        # display
        if verbose:
            print('Probably dependent')
        # return
        return False
    

## Leneve test (test if samples have same variance = Homoscedasticity)
def test_leneve(*args, alpha:float = 0.05, verbose:bool = False)->bool:
    """
    Leneve test (test if samples have same variance = Homoscedasticity).
    *args -- n groups of samples.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- True / False, samples have same distribution.
    """
    from scipy.stats import levene
    # test
    stat, p = levene(*args)
    # interpret
    if p > alpha:
        # display
        if verbose:
            print(f'Probably all samples with equal variances (fail to reject H0 with alpha = {alpha})')
        # return
        return True
    else:
        # display
        if verbose:
            print(f'Probably all samples with different variances (reject H0 with alpha = {alpha})')
        # return
        return False
    
    
## ANOVA test
def ANOVA(*args, alpha:float = 0.05, verbose:bool = False)->bool:
    """
    The one-way ANOVA variance test (parametric) tests the null hypothesis that two or more groups 
    have the same population mean. The test is applied to samples from two or more groups, possibly 
    with differing sizes.
    *args -- n groups of samples.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- True / False, samples have same distribution.
    """
    from scipy.stats import f_oneway
    try:
        # test
        stat, p = f_oneway(*args)
    except:
        # manage exception
        if verbose:
            print('[error-ANOVA] It was not possible get a result.')
        return np.nan           
    # display
    if verbose:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    if p > alpha:
        # display
        if verbose:
            print(f'Same distributions (fail to reject H0 with alpha = {alpha})')
        # return
        return True
    else:
        # display
        if verbose:
            print(f'Different distributions (reject H0 with alpha = {alpha})')
        # return
        return False
    
    
## Kruskal-Wallis test
def test_kruskal(*args, alpha:float = 0.05, verbose:bool = False)->bool:
    """
    The Kruskal-Wallis variance test (non-parametric) tests the null hypothesis that two or more groups 
    have the same population mean. The test is applied to samples from two or more groups, possibly 
    with differing sizes.
    *args -- n groups of samples.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- True / False, samples have same distribution.
    """
    from scipy import stats
    try:
        # test
        stat, p = stats.kruskal(*args)
    except:
        # manage exception
        if verbose:
            print('[error-Kruskal] It was not possible get a result.')
        return np.nan           
    # display
    if verbose:
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    if p > alpha:
        # display
        if verbose:
            print(f'Same distributions (fail to reject H0 with alpha = {alpha})')
        # return
        return True
    else:
        # display
        if verbose:
            print(f'Different distributions (reject H0 with alpha = {alpha})')
        # return
        return False
    
    
## Variance analysis: subsamples of num variable created by a cat variable are the same original sample
def analysis_variance(data_cat:np.array, 
                      data_num:np.array,
                      alpha:float = .05, 
                      verbose:bool = False)->bool:
    """
    ## Linear correlation analysis to test independence for numerical / ordinal variables.
    
    data_cat -- 1D categorical data to be tested. 
    date_num -- 1D numerical data to be tested.
    alpha -- Significance level (default, 0.05).
    verbose -- Display extra information (default, False).
    return -- boolean according test result.
    """
    # validate
    assert len(data_cat) == len(data_num), "both data arrays must to have the same lenght."
    # store in df and remove nan values
    temp = pd.DataFrame({'vcat':data_cat, 'vnum':data_num}).dropna()
    # validate
    if len(temp) == 0:
        return np.nan
    # number of records
    n = len(temp)
    # test if numerical variable is gaussian
    if n >= 5000:
        is_normal = test_anderson(temp.vnum.values, alpha = alpha)
    else:
        is_normal = test_shapiro(temp.vnum.values, alpha = alpha)      
    # collect groups data according categorical variable
    groups = temp.groupby("vcat")["vnum"]
    data_groups = [groups.get_group(c).values for c in temp.vcat.dropna().unique()]
    # validate number of groups
    if len(data_groups) == 1:
        return True
    # test Homoscedasticity inter-groups
    is_same_variance = test_leneve(*data_groups, alpha = alpha, verbose = verbose)
    # test if samples of numerical variable by categorical variable have same distribution
    if n >= 50:
        if is_same_variance:
            is_samples_same_distribution = ANOVA(*data_groups, alpha = alpha, verbose = verbose)
        else:
            is_samples_same_distribution = test_kruskal(*data_groups, alpha = alpha, verbose = verbose) 
            
    else:
        if is_normal:
            if is_same_variance:
                is_samples_same_distribution = ANOVA(*data_groups, alpha = alpha, verbose = verbose)
            else:
                is_samples_same_distribution = test_kruskal(*data_groups, alpha = alpha, verbose = verbose) 
        else:
            is_samples_same_distribution = test_kruskal(*data_groups, alpha = alpha, verbose = verbose) 
            
    # return
    return is_samples_same_distribution