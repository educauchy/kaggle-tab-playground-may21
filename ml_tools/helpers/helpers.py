import random


short_names = {
    'isoforest': 'if',
    'lof': 'lof',
    'onesvm': 'os',
    'iterative': 'iter',
    'KNN': 'knn',
    'simple': 'smpl',
    'DBSCAN': 'dbscn',
    'AdaBoost': 'ada',
    'GBM': 'GBM',
    'RF': 'RF',
    'LGBM': 'LGBM',
}


def gen_submit(config: dict, score_test: float) -> str:
    output = str(score_test) + '_' + str(random.randint(1, 100000000))
    return output

def save_submit() -> None:
    pass
