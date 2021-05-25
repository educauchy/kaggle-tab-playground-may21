from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
                                StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from ml_tools.helpers import Logging


class MetaClassifier(ClassifierMixin, Logging):
    def __init__(self, model: str = 'RF', verbose=False, random_state=None, params=None):
        super().__init__()
        self.verbose = verbose
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

        if model == 'Stacking':
            estimators = []
            for param in params:
                model_params = param['params'] if param['params'] is not None else {}
                estimators.append( (param['name'], self.models[param['name']](**model_params)) )
            self.model = self.models[model](estimators=estimators, final_estimator = LogisticRegression(max_iter=10000))
        else:
            self.model = self.models[model](random_state=random_state, **params)

        if verbose:
            logging.info(self.model)

    @Logging.logging_output('model')
    def fit(self, X, y=None):
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
        return self

    def predict(self, X, y=None):
        self.X = X.copy()
        if 'Is_Anomaly' in self.X.columns:
            self.X.drop(['Is_Anomaly'], axis=1, inplace=True)
        predict = self.model.predict(self.X)
        return predict

    def predict_proba(self, X, y=None):
        self.X = X.copy()
        if 'Is_Anomaly' in self.X.columns:
            self.X.drop(['Is_Anomaly'], axis=1, inplace=True)
        predict = self.model.predict_proba(self.X)
        return predict

    def score(self, X, y=None):
        score = self.model.score(X, y)
        return score

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
