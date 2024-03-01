from typing import Callable
from typing import Optional
from typing import cast

import numpy as np
from scipy.special import xlogy

from src.operations import EqualOperation
from src.operations import LessOrEqualOperation
from src.operations import MoreOrEqualOperation


def entropy(y_values: np.ndarray) -> float:
    # count occurencies of each class
    count = np.unique(y_values, return_counts=True)[1]
    # calculate probabilities of each class
    p = count / y_values.shape
    # calculate entropy
    entropy = -(xlogy(p, p) / np.log(2)).sum()
    return entropy


class SplitDetective:
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, random_state: Optional[int] = None):
        self.x_values = x_values
        self.y_values = y_values

        self.feature_ids = np.arange(self.x_values.shape[1])
        self.cur_entropy = entropy(self.y_values)

        self.random_state = random_state
        np.random.seed(seed=random_state)

    def get_best_feature(self):
        # get information gains of all features
        best_operations, best_igs = map(list, zip(*[self._info_gain(feature_id) for feature_id in self.feature_ids]))

        best_feature_idx = np.argmax(best_igs)

        return self.feature_ids[best_feature_idx], best_operations[best_feature_idx]

    def _info_gain(self, feature_id: int):
        # calculate information gain if split target feature
        best_operation, best_entropy_cond = self._cond_entropy(feature_id)

        best_ig = self.cur_entropy - best_entropy_cond
        return best_operation, best_ig

    def _cond_entropy(self, feature_id: int):
        # get feature values to search the split from
        feature_vals = np.unique(self.x_values[:, feature_id])
        if len(feature_vals) > 100:
            feature_vals = np.random.uniform(low=np.min(feature_vals), high=np.max(feature_vals), size=100)
            feature_vals = np.sort(feature_vals)

        true_idx = []
        false_idx = []
        probs = []

        operations_fun_array = [EqualOperation, MoreOrEqualOperation, LessOrEqualOperation]
        for operation_fun in operations_fun_array:
            for condition in feature_vals:
                operation = operation_fun(condition=condition)
                operation = cast(Callable, operation)

                true_idx.append([operation(val) for val in self.x_values[:, feature_id]])
                false_idx.append([~operation(val) for val in self.x_values[:, feature_id]])
                probs.append(np.sum([operation(val) for val in self.x_values[:, feature_id]]) / self.x_values.shape[0])

        H_cond = [
            entropy(self.y_values[eq_idx]) * p + entropy(self.y_values[neq_idx]) * (1 - p)
            for (eq_idx, neq_idx, p) in zip(true_idx, false_idx, probs)
        ]

        # WHY WAS HERE ARGMAX
        best_id = np.argmin(H_cond)

        best_val = feature_vals[int(best_id % (len(H_cond) / len(operations_fun_array)))]
        best_operation = operations_fun_array[int(best_id // (len(H_cond) / len(operations_fun_array)))]

        return best_operation(condition=best_val), H_cond[best_id]
