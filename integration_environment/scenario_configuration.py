from dataclasses import dataclass
from enum import Enum


class PayloadSizeConfig(Enum):
    SMALL = 8
    MEDIUM = 100
    LARGE = 200


@dataclass
class ScenarioConfiguration:
    scenario_id: str
    payload_size: PayloadSizeConfig
    num_devices: int
