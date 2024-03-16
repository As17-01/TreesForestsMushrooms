from typing import Callable
from typing import Optional
from typing import Sequence
from typing import cast

import numpy as np

# from src.operations import LessOrEqualOperation
from src.operations import EqualOperation
from src.operations import MoreOrEqualOperation

# import time

# from loguru import logger


def entropy(y_values: np.ndarray) -> float:
    # count occurencies of each class
    count = np.unique(y_values, return_counts=True)[1]
    # calculate probabilities of each class
    p = count / y_values.shape
    # calculate entropy
    entropy = -(p * np.log(p) / np.log(2)).sum()
    return entropy


def gini(y_values: np.ndarray) -> float:
    # count occurencies of each class
    count = np.unique(y_values, return_counts=True)[1]
    # calculate probabilities of each class
    p = count / y_values.shape
    # calculate gini
    entropy = 1 - (p**2).sum()
    return entropy


class SplitDetective:
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, criterion: str, random_state: Optional[int] = None):
        self.x_values = x_values
        self.y_values = y_values

        self.criterion = criterion
        if criterion == "gini":
            self._criterion = gini
        elif criterion == "entropy":
            self._criterion = entropy
        else:
            raise ValueError("Only `gini` and `entropy` criterions are allowed!")

        self.feature_ids = np.arange(self.x_values.shape[1])
        self.cur_entropy = entropy(self.y_values)

        self.random_state = random_state
        np.random.seed(seed=random_state)

    def get_best_feature(self, cat_features_idx: Sequence[int]):
        # start = time.time()
        # get information gains of all features
        best_operations_list = []
        best_igs_list = []
        for feature_id in self.feature_ids:
            is_cat_feature = feature_id in cat_features_idx

            best_operation, best_entropy_cond = self._get_best_operation(feature_id, is_cat_feature)
            best_ig = self.cur_entropy - best_entropy_cond

            best_operations_list.append(best_operation)
            best_igs_list.append(best_ig)

        best_feature_idx = np.argmax(best_igs_list)

        # end = time.time()
        # logger.debug(f"SEARCHING TIME: {end - start}")
        return self.feature_ids[best_feature_idx], best_operations_list[best_feature_idx]

    def _get_best_operation(self, feature_id: int, is_categorical: bool):
        # Create space to search the split from
        if is_categorical:
            feature_vals = np.unique(self.x_values[:, feature_id])

            operations_fun_array = [EqualOperation]  # type: ignore
        else:
            feature_vals = np.sort(
                np.random.uniform(
                    np.min(self.x_values[:, feature_id]),
                    np.max(self.x_values[:, feature_id]),
                    20,
                )
            )
            # feature_vals = np.sort(
            #     np.random.normal(
            #         np.mean(self.x_values[:, feature_id]),
            #         np.std(self.x_values[:, feature_id]),
            #         20,
            #     )
            # )

            # operations_fun_array = [MoreOrEqualOperation, LessOrEqualOperation]  # type: ignore
            operations_fun_array = [MoreOrEqualOperation]  # type: ignore

        H_cond = self._entr_over_space(feature_id, operations_fun_array, feature_vals)
        # WHY WAS HERE ARGMAX
        best_id = np.argmin(H_cond)

        best_val = feature_vals[int(best_id % (len(H_cond) / len(operations_fun_array)))]
        best_operation = operations_fun_array[int(best_id // (len(H_cond) / len(operations_fun_array)))]

        return best_operation(condition=best_val), H_cond[best_id]

    def _entr_over_space(self, feature_id: int, operations_fun_array: Sequence[Callable], feature_vals: np.ndarray):
        H_cond = []
        for operation_fun in operations_fun_array:
            for condition in feature_vals:
                operation = operation_fun(condition=condition)
                operation = cast(Callable, operation)

                # MAKE SURE OPERATION IS VALID OVER ARRAYS
                condition_result = operation(self.x_values[:, feature_id])
                true_idx = condition_result.copy()
                false_idx = ~condition_result.copy()
                p = np.sum(condition_result) / self.x_values.shape[0]

                true_idx_y_values = self.y_values[true_idx]
                false_idx_y_values = self.y_values[false_idx]

                H_cond.append(self._criterion(true_idx_y_values) * p + self._criterion(false_idx_y_values) * (1 - p))
        return H_cond
