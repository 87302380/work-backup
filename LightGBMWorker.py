from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np

from hpbandster.core.worker import Worker
import ConfigSpace as CS


class LightGBMWorker(Worker):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        dataset = load_breast_cancer()

        x = dataset.data
        y = dataset.target

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=y,
                                                            shuffle=True)

        self.train_loader = lgb.Dataset(x_train, label=y_train)
        self.test_loader = lgb.Dataset(x_test, label=y_test)


    def compute(self, config, budget, *args, **kwargs):
        max_depth = int(config['max_depth'])
        num_leaves = int(config['num_leaves'])
        max_bin = int(config['max_bin'])
        min_data_in_leaf = int(config['min_data_in_leaf'])
        # num_trees = int(config['num_trees'])
        bagging_fraction = config['bagging_fraction']
        bagging_freq = int(config['bagging_freq'])
        feature_fraction = config['feature_fraction']
        # lambda_l2 = params['lambda_l2'],
        # min_gain_to_split = params['min_gain_to_split']

        param = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.1,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            #'num_trees': num_trees,
            'max_bin': max_bin,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'feature_fraction': feature_fraction,
            # 'verbose': -1,
            # 'lambda_l2': lambda_l2,
            # 'min_gain_to_split': min_gain_to_split
        }

        cv_results = lgb.cv(
            param,
            self.train_loader,
            nfold=10,
            metrics='binary_error',
            early_stopping_rounds=10
        )

        l2_mean = min(cv_results['binary_error-mean'])

        return ({
            'loss': float(l2_mean),  # this is the a mandatory field to run hyperband
            'info': l2_mean  # can be used for any user-defined information - also mandatory
        })



    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', lower=3, upper=8))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('num_leaves', lower=3, upper=50))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('min_data_in_leaf', lower=3, upper=50))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_bin', lower=3, upper=50))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('bagging_fraction', lower=0, upper=1))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('bagging_freq', lower=0, upper=50))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('feature_fraction', lower=0, upper=1))

        return (config_space)