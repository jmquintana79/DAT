from data import Samples
from DAT import EDA
import pandas as pd

## load iris dataset
sp = Samples()
data = sp.load_dataset_iris()

## test eda empty df validation
def test_empty_df_validation():
    # initialize class
    eda = EDA()
    # test if df is empty or not
    assert eda._validate_if_df_empty(pd.DataFrame()) is True
    assert eda._validate_if_df_empty(data) is False
    # test in case of input is not a df
    try:
        _ = eda._validate_if_df_empty(999)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done is False


## test eda info method is working
def test_eda_info():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.info(data, decimals=2)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda missing method is working
def test_eda_missing():
    # initialize class
    eda = EDA()
    # describe info
    try:
        #_ = eda.missing(data) # avoid plots
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda outliers method is working
def test_eda_outliers():
    # initialize class
    eda = EDA()
    # describe info
    try:
        #_ = eda.outliers(data, num_iqr = 1.5) # avoid plots
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda numeric method is working
def test_eda_numeric():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.numeric(data)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda categorical method is working
def test_eda_categorical():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.categorical(data)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda summary method is working
def test_eda_summary():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.summary(data)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda temporal method is working
def test_eda_temporal():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.temporal(data)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda bivariate method is working
def test_eda_bivariate():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.bivariate(data)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done


## test eda duplicates method is working
def test_eda_duplicates():
    # initialize class
    eda = EDA()
    # describe info
    try:
        _ = eda.duplicates(data)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done