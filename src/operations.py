from typing import Any


class EqualOperation:
    def __init__(self, condition: Any):
        self.condition = condition

    def __call__(self, a: Any) -> bool:
        return a == self.condition
    
    def __repr__(self) -> str:
        return f"=={self.condition}"
    
class MoreOrEqualOperation:
    def __init__(self, condition: Any):
        self.condition = condition

    def __call__(self, a: Any) -> bool:
        return a >= self.condition
    
    def __repr__(self) -> str:
        return f">={self.condition}"
    
class LessOrEqualOperation:
    def __init__(self, condition: Any):
        self.condition = condition

    def __call__(self, a: Any) -> bool:
        return a <= self.condition
    
    def __repr__(self) -> str:
        return f"<={self.condition}"
