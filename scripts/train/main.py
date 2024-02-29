import pathlib
import sys

import hydra
import numpy as np
import omegaconf
import pandas as pd
from hydra_slayer import Registry
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("../../")

import src

FEATURES = ["does-bruise-or-bleed", "habitat", "season", "cap-diameter", "stem-height", "stem-width"]
TARGET = "class"


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    train_path = pathlib.Path(cfg.data.train_key)
    test_path = pathlib.Path(cfg.data.test_key)
    save_path = pathlib.Path(cfg.data.save_key)

    data = pd.read_csv(train_path, index_col="Id").reset_index(drop=True)
    test_data = pd.read_csv(test_path, index_col="Id")

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src, prefix="src.")
    algorithm = registry.get_from_params(**cfg_dct["algorithm"])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=200)

    logger.info("Training...")
    metric_history = []
    for i, (train_index, val_index) in enumerate(kf.split(data, data[TARGET])):
        logger.info(f"Fold {i}")

        train = data.iloc[train_index]
        val = data.iloc[val_index]

        algorithm.fit(train[FEATURES], train[TARGET])

        predictions = algorithm.predict(val[FEATURES])
        score = roc_auc_score(y_true=val[TARGET], y_score=predictions)
        logger.info(f"Fold {i} val ROC AUC: {score}")

        predictions_train = algorithm.predict(train[FEATURES])
        score_train = roc_auc_score(y_true=train[TARGET], y_score=predictions_train)
        logger.info(f"Fold {i} train ROC AUC: {score_train}")

        metric_history.append(score)
        break

    logger.info(f"ROC AUC: {sum(metric_history) / len(metric_history)}")
    
    best_thr = 0.0
    best_f1 = 0.0
    for thr in np.arange(0.0, 1.0, 0.01):
        cur_f1 = f1_score(y_true=val[TARGET], y_pred=np.where(predictions >= thr, 1, 0), average="macro")
        if cur_f1 > best_f1:
            best_thr = thr
            best_f1 = cur_f1
    logger.info(f"Best threshold: {best_thr} with F1: {best_f1}")

    logger.info("Predicting...")
    test_predictions = test_data.reset_index()[["Id"]].copy()
    test_predictions["class"] = np.where(algorithm.predict(test_data[FEATURES]) >= best_thr, 1, 0)
    test_predictions.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
