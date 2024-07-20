import glob
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.preprocessing.utils.utils import (
    check_for_corrupted_files,
    create_virtual_dataset,
)


@hydra.main(version_base=None, config_path="../config", config_name="create_vrt_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dir_path = os.path.expanduser(cfg.dir_path)
    full_pattern = f"{dir_path}/**/{cfg.file_pattern}"
    files_list = glob.glob(full_pattern, recursive=cfg.recursive)
    # For LiDAR, also create a vrt for classification files
    classif_files_list = [x for x in files_list if "classif" in x]
    files_list = [x for x in files_list if ("classif" not in x) and ("ISSUE" not in x)]
    if cfg.check_invalid_files:
        files_list = check_for_corrupted_files(files_list)
        if len(classif_files_list):
            classif_files_list = check_for_corrupted_files(classif_files_list)

    if cfg.vrt_path is None:
        vrt_path = os.path.join(dir_path, "full.vrt")
    else:
        vrt_path = cfg.vrt_path
    create_virtual_dataset(
        files_list, vrt_path, crs=cfg.crs, noDataValue=cfg.noDataValue
    )
    if len(classif_files_list):
        classif_vrt_path = os.path.join(dir_path, "full_classification_mask.vrt")
        create_virtual_dataset(
            classif_files_list,
            classif_vrt_path,
            crs=cfg.crs,
            noDataValue=cfg.noDataValue,
        )


if __name__ == "__main__":
    main()
