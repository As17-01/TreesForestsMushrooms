from typing import Any, Union
import numpy as np


class EqualOperation:
    def __init__(self, condition: Any):
        self.condition = condition

    def __call__(self, a: Any) -> Union[bool, np.ndarray]:
        return a == self.condition

    def __repr__(self) -> str:
        return f" == {self.condition}"


class MoreOrEqualOperation:
    def __init__(self, condition: Any):
        self.condition = condition

    def __call__(self, a: Any) -> Union[bool, np.ndarray]:
        return a >= self.condition

    def __repr__(self) -> str:
        return f" >= {self.condition}"


class LessOrEqualOperation:
    def __init__(self, condition: Any):
        self.condition = condition

    def __call__(self, a: Any) -> Union[bool, np.ndarray]:
        return a <= self.condition

    def __repr__(self) -> str:
        return f" <= {self.condition}"
