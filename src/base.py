from abc import ABC
from abc import abstractmethod

import pandas as pd

from src.encoder import LabelEncoder


class BaseModel(ABC):
    def _process_categorical(self, x: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        if is_train:
            self.encoders = {}
            for col_name in x.select_dtypes(include=["object"]):
                encoder = LabelEncoder()

                encoder.fit(x[col_name].values)
                self.encoders[col_name] = encoder

                x[col_name] = encoder.encode(x[col_name].values)
        else:
            for col_name, encoder in self.encoders.items():
                x[col_name] = encoder.encode(x[col_name].values)
        return x

    @abstractmethod
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test: pd.DataFrame):
        raise NotImplementedError
