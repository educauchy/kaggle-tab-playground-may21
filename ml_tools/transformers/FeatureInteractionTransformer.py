from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np


class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree: int = 1,
                        interaction_only: bool = True,
                        include_bias: bool = False,
                        exclude_cols: bool = None):
        super().__init__()
        self.exclude_cols = exclude_cols
        self.creator = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)

    def fit(self, X, y=None):
        self.creator.fit(X)
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        new_values = np.nan_to_num((self.creator.transform(self.X).astype(np.float32)))
        d = pd.DataFrame(data=new_values, columns=self.creator.get_feature_names())
        self.X = self.X.join(d)
        return self.X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
