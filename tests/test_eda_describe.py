from data import Samples
from DAT import EDA

## load iris dataset
sp = Samples()
data = sp.load_dataset_iris()

## test eda describe method is working
def test_eda_describe():
    # initialize class
    eda = EDA()
    # describe info
    try:
        df = eda.describe_info(data, decimals=2)
        is_done = True
    except:
        is_done = False
    # test
    assert is_done
    

