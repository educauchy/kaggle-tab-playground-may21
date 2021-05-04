from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd


class EncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type: str = 'label',
                        column: str = '',
                        out_column: str = '',
                        data: pd.Series = None):
        super().__init__()
        self.type = type
        self.column = column
        self.out_column = out_column
        self.data = data
        self.encoders = {
            'label': LabelEncoder(),
            'ordinal': OrdinalEncoder(),
            'onehot': OneHotEncoder(categories='auto', handle_unknown='ignore')
        }
        self.encoder = self.encoders[type]

    def fit(self, X, y=None):
        self.X = X.copy()
        if self.data is None:
            if self.type == 'onehot':
                # print(np.array(self.X[self.column].astype(str)).reshape(-1, 1))
                self.encoder.fit( np.array(self.X[self.column].astype(str)).reshape(-1, 1) )
            else:
                self.encoder.fit(self.X[self.column].astype(str))
        else:
            self.encoder.fit(self.data.astype(str))
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        if self.type in ['label', 'ordinal']:
            self.X[self.out_column] = self.encoder.transform(self.X[self.column].astype(str))
            self.X.loc[self.X[self.column].isnull(), self.out_column] = np.nan
        elif self.type == 'onehot':
            enc_df = pd.DataFrame(self.encoder.transform(np.array(self.X[self.column].astype(str)).reshape(-1, 1)))
            self.X = self.X.join(enc_df, rsuffix='_onehot')

        return self.X

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
