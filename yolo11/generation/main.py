# This file is the entrance of calibration set generation step.
#
# It parses commandline arguments, then starts generation.
#

from generation.config import get_config, GenerationConfig
from generation.distill_trainer import build_distill_trainer  # type: ignore
from pathlib import Path

from ultralytics.utils.files import increment_path  # type: ignore


def save_config(work_dir: Path, config: GenerationConfig):
    config_path = work_dir / "config.json"
    with config_path.open("w") as f:
        f.write(config.model_dump_json(indent=2))


if __name__ == "__main__":
    config = get_config()
    save_dir = increment_path(
        Path(config.project) / config.name, exist_ok=False, mkdir=True
    )
    save_config(save_dir, config)  # for reproduction of experiment results.
    distill_trainer = build_distill_trainer(save_dir, config)
    distill_trainer.train_all()
