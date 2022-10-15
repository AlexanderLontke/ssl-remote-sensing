from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load


# Project Directories
TASK_ROOT = Path("pretext_tasks/gan")
ROOT = Path()
CONFIG_FILE_PATH = TASK_ROOT / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = ROOT / "models"

class Config(BaseModel):
    """Master config object."""
    eurosat_data_dir: str
    image_height: int
    image_width: int
    num_channels: int
    size_latent: int
    g_featuremaps: int
    d_featuremaps: int
    num_gpus: int
    max_epochs: int
    batch_size: int
    num_workers: int
    lr: float
    b1: float
    b2: float

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(**parsed_config.data)

    return _config


config = create_and_validate_config()