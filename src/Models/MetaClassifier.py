from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
                                StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier



class MetaClassifier(ClassifierMixin):
    def __init__(self, model: str = 'RF', **params):
        super().__init__()
        self.models = {
            'RF': RandomForestClassifier,
            'LogReg': LogisticRegression,
            'SVM': LinearSVC,
            'AdaBoost': AdaBoostClassifier,
            'GBM': GradientBoostingClassifier,
            'Tree': DecisionTreeClassifier,
            'Stacking': StackingClassifier,
            'Bagging': BaggingClassifier,
            'LGBM': LGBMClassifier,
        }
        self.model = self.models[model](**params)
        print('Model:')
        print(self.model)
        print('-----------------------')

    def fit(self, X, y=None):
        print('Fitting model begins...')
        self.X = X.copy()
        self.y = y.copy()
        self.y.reset_index(drop=True, inplace=True)
        if 'Is_Anomaly' in self.X.columns:
            non_nan_anomaly = self.X[self.X.Is_Anomaly == 1].index
            self.X = self.X[self.X.Is_Anomaly == 1]
            self.y = self.y.reindex(non_nan_anomaly)
            self.X.drop(['Is_Anomaly'], axis=1, inplace=True)
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        self.model.fit(self.X, self.y)
        print('Fitting model ended...')
        print('')
        return self

    def predict(self, X, y=None):
        self.X = X.copy()
        if 'Is_Anomaly' in self.X.columns:
            self.X.drop(['Is_Anomaly'], axis=1, inplace=True)
        predict = self.model.predict(self.X)
        return predict

    def score(self, X, y=None):
        score = self.model.score(X, y)
        return score

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
