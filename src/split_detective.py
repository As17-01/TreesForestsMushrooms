import numpy as np
from scipy.special import xlogy


def entropy(y_values: np.ndarray) -> float:
    # count occurencies of each class
    count = np.unique(y_values, return_counts=True)[1]
    # calculate probabilities of each class
    p = count / y_values.shape
    # calculate entropy
    entropy = -(xlogy(p, p) / np.log(2)).sum()
    return entropy


class SplitDetective:
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray):
        self.x_values = x_values
        self.y_values = y_values

        self.feature_ids = np.arange(self.x_values.shape[1])
        self.cur_entropy = entropy(self.y_values)

    def get_best_feature(self):
        # get information gains of all features
        best_split_vals, best_igs = map(list, zip(*[self._info_gain(feature_id) for feature_id in self.feature_ids]))

        best_feature_idx = np.argmax(best_igs)

        return self.feature_ids[best_feature_idx], best_split_vals[best_feature_idx]

    def _info_gain(self, feature_id: int):
        # calculate information gain if split target feature
        best_split_val, best_entropy_cond = self._cond_entropy(feature_id)

        best_ig = self.cur_entropy - best_entropy_cond
        return best_split_val, best_ig

    def _cond_entropy(self, feature_id: int):
        # get feature values and corresponding counts of target feature
        feature_vals, val_counts = np.unique(self.x_values[:, feature_id], return_counts=True)
        all_probabilities = val_counts / self.x_values.shape[0]
        # calculate probabilities of each feature value of target feature

        # indices, where target feature is equal to each of the feature values
        equal_idx = [self.x_values[:, feature_id] == val for val in feature_vals]
        # indices, where target feature is not equal to each of the feature values
        n_equal_idx = [self.x_values[:, feature_id] != val for val in feature_vals]

        # conditional entropies of each value of each feature value
        H_cond = [
            entropy(self.y_values[eq_idx]) * p + entropy(self.y_values[neq_idx]) * (1 - p)
            for (eq_idx, neq_idx, p) in zip(equal_idx, n_equal_idx, all_probabilities)
        ]

        # get best feature value
        best_id = np.argmax(H_cond)
        return feature_vals[best_id], H_cond[best_id]
