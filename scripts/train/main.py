import pathlib
import sys

import hydra
import numpy as np
import omegaconf
import pandas as pd
from hydra_slayer import Registry
from loguru import logger

sys.path.append("../../")

import src

FEATURES = ["does-bruise-or-bleed", "habitat", "season", "cap-diameter", "stem-height", "stem-width"]
TARGET = "class"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)

    f1_score = 0
    for cl in classes:
        tp = np.sum(y_true[y_true == cl] == y_pred[y_true == cl])
        fp = np.sum(y_true[y_true == cl] != y_pred[y_true == cl])
        fn = np.sum(y_true[y_true != cl] != y_pred[y_true != cl])

        f1_score += (tp / (tp + (fp + fn) / 2)) / len(classes)
    return f1_score


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    train_path = pathlib.Path(cfg.data.train_key)
    test_path = pathlib.Path(cfg.data.test_key)
    save_path = pathlib.Path(cfg.data.save_key)

    train_data = pd.read_csv(train_path, index_col="Id")
    test_data = pd.read_csv(test_path, index_col="Id")

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src, prefix="src.")
    algorithm = registry.get_from_params(**cfg_dct["algorithm"])

    train_data.reset_index(drop=True, inplace=True)
    train_index = np.random.choice(np.array(train_data.index), size=int(0.8 * len(train_data)), replace=False)
    is_train = train_data.index.isin(train_index)
    
    train = train_data.iloc[is_train]
    val = train_data.iloc[~is_train]

    logger.info(f"DATA SIZE: {len(train_data)}")
    logger.info(f"TRAIN SIZE: {len(train)}")
    logger.info(f"VAL SIZE: {len(val)}")

    logger.info("Training...")

    algorithm.fit(train[FEATURES], train[TARGET])

    predictions_train = algorithm.predict(train[FEATURES])
    predictions_val = algorithm.predict(val[FEATURES])

    score_f1_train = macro_f1(y_true=train[TARGET], y_pred=np.where(predictions_train >= 0.5, 1, 0))
    score_f1_val = macro_f1(y_true=val[TARGET], y_pred=np.where(predictions_val >= 0.5, 1, 0))

    logger.info(f"Train F1: {score_f1_train}")
    logger.info(f"Val F1: {score_f1_val}")

    logger.info("Predicting...")
    test_predictions = test_data.reset_index()[["Id"]].copy()
    test_predictions["class"] = np.where(algorithm.predict(test_data[FEATURES]) >= 0.5, 1, 0)
    test_predictions.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
