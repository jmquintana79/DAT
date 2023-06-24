import pandas as pd
from sklearn.datasets import load_iris

class Samples():
    def __init__(self):
        pass

    ## load iris dataset
    def load_dataset_iris(self)->pd.DataFrame:
        # load dataset
        dataset = load_iris()
        dataset.keys()
        # dataset to df
        data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        data['class'] = dataset.target
        dclass = dict()
        for i, ic in enumerate(dataset.target_names):
            dclass[i] = ic
        data['class'] = data['class'].map(dclass)
        # return
        return data

