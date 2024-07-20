import os
import sys
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from sympy import check_assumptions

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("End of testing!")
    # for predictions use trainer.predict(...)
    # log.info("Starting predicting!")
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
    # log.info("End of predicting!")

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    # look at the args. If a path is given, then use it to extract the config and check point
    # print current working directory
    if len(sys.argv) > 1:
        training_dir = sys.argv[1]
    else:
        raise ValueError("Please provide a training result directory")
    assert os.path.isdir(
        training_dir
    ), "Please provide a valid training directory"
    assert os.path.isfile(
        os.path.join(training_dir, ".hydra", "config.yaml")
    ), "No config.yaml found in training directory"
    assert os.path.isfile(
        os.path.join(training_dir, "checkpoints", "last.ckpt")
    ), "No last.ckpt found in training directory"

    # path to config.yaml
    config_dir = os.path.join(
        os.getcwd(), training_dir, ".hydra"
    )  # hydra look for config relative to script

    # path to checkpoint
    if len(sys.argv) > 2:
        checkpoint_type = sys.argv[2]
    else:
        checkpoint_type = "-best"
    assert checkpoint_type in [
        "-last",
        "-best",
    ], "Please provide a valid checkpoint type"
    if checkpoint_type == "-last":
        checkpoint_path = os.path.join(
            training_dir, "checkpoints", "last.ckpt"
        )
    else:
        # look into /checkpoints/ for other file
        files_list = os.listdir(os.path.join(training_dir, "checkpoints"))
        files_list = [
            file_name
            for file_name in files_list
            if file_name.startswith("epoch_") and file_name.endswith(".ckpt")
        ]
        if len(files_list) == 0:
            raise ValueError(
                "No best epoch checkpoint found in training directory"
            )
        elif len(files_list) > 1:
            raise ValueError(
                "Multiple best epoch checkpoints found in training directory"
            )
        checkpoint_path = os.path.join(
            training_dir, "checkpoints", files_list[0]
        )

    # generate cmd line for calling hydra
    cmd = [
        "cmd",  # this is ignored
        "-cn",
        "config",
        "-cd",
        config_dir,
        "ckpt_path=" + checkpoint_path,
        "+trainer=gpu",
    ]

    # override cmd line args
    sys.argv = cmd
    main()
