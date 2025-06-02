from dataclasses import dataclass
from enum import Enum


class PayloadSizeConfig(Enum):
    small = 8
    medium = 100
    large = 200


class ScenarioDuration(Enum):
    one_min = 60 * 1000
    five_min = 5 * 60 * 1000
    one_hour = 60 * 60 * 1000


class ModelType(Enum):
    ideal = "ideal"
    channel = "channel"
    static_graph = "static_graph"
    detailed = "detailed"
    meta_model = "meta_model"


class NumDevices(Enum):
    two = 2
    ten = 10
    fifty = 50
    hundred = 100
    thousand = 1000


@dataclass
class ScenarioConfiguration:
    payload_size: PayloadSizeConfig
    num_devices: NumDevices
    model_type: ModelType
    scenario_duration: ScenarioDuration

    @property
    def scenario_id(self):
        """Create a scenario ID that includes all configuration parameters."""
        return f"{self.model_type.name}-{self.num_devices.name}-{self.payload_size.name}-{self.scenario_duration.name}"
