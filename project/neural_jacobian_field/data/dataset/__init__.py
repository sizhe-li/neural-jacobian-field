from .config_parser import DNeRFDataParserConfig
from .dataset_allegro import DatasetAllegroCfg, DatasetAllegro
from .dataset_hsa import DatasetHsaCfg, DatasetHsa
from .dataset_pneumatic import DatasetPneumaticHandOnlyCfg, DatasetPneumaticHandOnly
from .dataset_toy_arm import DatasetToyArmCfg, DatasetToyArm

DatasetCfg = (
    DatasetAllegroCfg | DatasetHsaCfg | DatasetToyArmCfg | DatasetPneumaticHandOnlyCfg
)
DatasetType = DatasetAllegro | DatasetHsa | DatasetToyArm | DatasetPneumaticHandOnly
