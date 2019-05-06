from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
import pandas as pd

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 0.2,
                                                    random_state=41,
                                                    stratify = y,
                                                    shuffle = True)

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)


def get_f1_score(model):
    predict = model.predict(x_test, num_iteration=model.best_iteration)
    predict_label = np.round(predict)

    return f1_score(test_data.get_label(), predict_label, average='micro')

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics':'binary_error',
    'max_depth': 3,
    'num_leaves': 3,
    'max_bin':7,
    'min_data_in_leaf': 20,
    'learning_rate': 0.1,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'feature_fraction': 0.2,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'min_gain_to_split': 0
}


gbm = lgb.train(params,
                train_data,
                num_boost_round=10,
                valid_sets=test_data,
                early_stopping_rounds=5)

print(get_f1_score(gbm))

