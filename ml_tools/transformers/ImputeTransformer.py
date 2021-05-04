from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
import pandas as pd
import numpy as np


class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type: str = 'KNN',
                        by_cols: dict = None,
                        **params):
        super().__init__()
        self.type = type
        self.by_cols = by_cols
        self.imputers = {
            'KNN': KNNImputer,
            'iterative': IterativeImputer,
            'simple': SimpleImputer,
        }
        self.imputer = self.imputers[type](**params)

    def _impute_by_cols(self, X, by_cols):
        if by_cols['func'] == 'count':
            impute_data = X[by_cols['by'] + [by_cols['target']]].groupby(by=by_cols['by'])[by_cols['target']].count().reset_index()
        elif by_cols['func'] == 'median':
            impute_data = X[by_cols['by'] + [by_cols['target']]].groupby(by=by_cols['by'])[by_cols['target']].median().reset_index()
        X_ = X.merge(impute_data, on=by_cols['by'])
        X_[by_cols['target']] = np.where(X_[by_cols['target'] + '_x'].isnull(), X_[by_cols['target'] + '_y'], X_[by_cols['target'] + '_x'])
        X_.drop([by_cols['target'] + '_x', by_cols['target'] + '_y'], axis=1, inplace=True)
        return X


    def fit(self, X, y=None):
        print(self.imputer)
        print(X.isnull().sum())
        print('Imputing begins...')

        for by_col in self.by_cols:
            X = self._impute_by_cols(X, by_col)

        self.imputer.fit(X, y)
        print('Imputing ended...')
        print('')
        return self

    def transform(self, X, y=None):
        columns = X.columns
        imputed_data = self.imputer.transform(X)
        return pd.DataFrame(imputed_data, columns=columns)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
