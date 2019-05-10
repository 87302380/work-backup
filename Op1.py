from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GridSearchCV

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 0.2,
                                                    random_state=35,
                                                    stratify = y,
                                                    shuffle = True)



params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
}

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=35,
                                                    stratify=y,
                                                    shuffle=True)

lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)

min_merror = float('Inf')
best_params = {}

# params_test1={'max_depth': range(3,8,1), 'num_leaves':range(3, 20, 1)}
#
# gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(
#                                     boosting_type='gbdt',
#                                     objective='binary',
#                                     #metrics='l2',
#                                     max_depth=4,
#                                     num_leaves=10,
#                                     max_bin= 7,
#                                     min_data_in_leaf=20,
#                                     learning_rate=0.1,
#                                     bagging_fraction=0.5,
#                                     bagging_freq=20,
#                                     feature_fraction=0.2,
#                                     ),
#                                     param_grid = params_test1,
#                                     scoring="accuracy",
#                                     cv=10)
# gsearch1.fit(x,y)
# print(gsearch1.score, gsearch1.best_params_, gsearch1.best_score_)


params_test2={'max_bin': range(3,40,1), 'min_data_in_leaf':range(2,50,1)}


gsearch2 = GridSearchCV(estimator = lgb.LGBMClassifier(
                                    boosting_type='gbdt',
                                    objective='binary',
                                    #metrics='binary log loss',
                                    max_depth=4,
                                    num_leaves=10,
                                    learning_rate=0.1,
                                    # bagging_fraction=0.1,
                                    # bagging_freq=0,
                                    # feature_fraction=1,
                                    ),
                                    param_grid = params_test2,
                                    scoring="accuracy",
                                    cv=10
                                    )
gsearch2.fit(x,y)
print(gsearch2.score, gsearch2.best_params_, gsearch2.best_score_)

# params_test3={'feature_fraction': [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#               'bagging_fraction': [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#               'bagging_freq': range(0,50,1)
# }
#
# gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(
#                                     boosting_type='gbdt',
#                                     objective='binary',
#                                     #metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=3,
#                                     max_bin= 7,
#                                     min_data_in_leaf=20,
#                                     learning_rate=0.1,
#                                     bagging_fraction=0.5,
#                                     bagging_freq=20,
#                                     feature_fraction=0.2,
#                                     ),
#                                     param_grid = params_test3,
#                                     scoring="f1",
#                                     cv=3,
#                                     n_jobs=4)
# gsearch3.fit(x_train,y_train)
# print(gsearch3.score, gsearch3.best_params_, gsearch3.best_score_)

# params_test4={'lambda_l1': [0.0,0.1,0.3,0.5,0.7,0.9,1.0],
#               'lambda_l2': [0.0,0.1,0.3,0.5,0.7,0.9,1.0]
# }
#
# gsearch4 = GridSearchCV(estimator = lgb.LGBMClassifier(
#                                     boosting_type='gbdt',
#                                     objective='binary',
#                                     #metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=3,
#                                     max_bin= 7,
#                                     min_data_in_leaf=20,
#                                     learning_rate=0.1,
#                                     bagging_fraction=0.5,
#                                     bagging_freq=20,
#                                     feature_fraction=0.2,
#                                     ),
#                                     param_grid = params_test4,
#                                     scoring="f1",
#                                     cv=3,
#                                     n_jobs=4)
# gsearch4.fit(x_train,y_train)
# print(gsearch4.score, gsearch4.best_params_, gsearch4.best_score_)

# params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
#
# gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(
#                                     boosting_type='gbdt',
#                                     objective='binary',
#                                     #metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=3,
#                                     max_bin= 7,
#                                     min_data_in_leaf=20,
#                                     learning_rate=0.1,
#                                     bagging_fraction=0.5,
#                                     bagging_freq=20,
#                                     feature_fraction=0.2,
#                                     lambda_l1=0.0,
#                                     lambda_l2=0.0
#                                     ),
#                                     param_grid = params_test5,
#                                     scoring="f1",
#                                     cv=3,
#                                     n_jobs=4)
# gsearch5.fit(x_train,y_train)
# print(gsearch5.score, gsearch5.best_params_, gsearch5.best_score_)