from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    non_trainable_param = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams["model/params/total"] = total_param
    hparams["model/params/trainable"] = trainable_param
    hparams["model/params/non_trainable"] = non_trainable_param
    log.info(
        f"Total params: {total_param}, Trainable params: {trainable_param}, Non-trainable params: {non_trainable_param}"
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
