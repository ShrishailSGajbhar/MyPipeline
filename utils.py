import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse

class PandasSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, x, y = None):
        return self
    
    def transform(self, x):
        return x.loc[:,self.columns]

class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

class PandasStandardScalar(StandardScaler):

    def transform(self, X, y=None):
        z = super(PandasStandardScalar, self).transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)

class PandasOneHotEncoder(OneHotEncoder):

    def __init__(self, sparse=True, handle_unknown='error'):
        super(PandasOneHotEncoder, self).__init__(sparse=False, handle_unknown='ignore')

    def transform(self, X, y=None):
        z = super(PandasOneHotEncoder, self).transform(X.values)
        #Create a Pandas DataFrame of the hot encoded column
        ohe_df = pd.DataFrame(z, index = X.index, columns=super(PandasOneHotEncoder, self).get_feature_names())
        #concat with original data
        data = pd.concat([X, ohe_df], axis=1).drop(X.columns.tolist(), axis=1)
        return data 

