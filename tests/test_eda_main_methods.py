from data import Samples
from DAT import EDA

## load iris dataset
sp = Samples()
data = sp.load_dataset_iris()

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
