from dataclasses import dataclass
from enum import Enum


class NetworkModelType(Enum):
    simbench_ethernet = 'Ethernet'
    simbench_lte = 'LTE'
    simbench_lte450 = 'LTE450'
    simbench_5g = 'Net5G'
    none = ''


class PayloadSizeConfig(Enum):
    small = 8
    medium = 100
    large = 200
    none = 0


class ScenarioDuration(Enum):
    one_min = 60 * 1000
    five_min = 5 * 60 * 1000
    one_hour = 60 * 60 * 1000
    none = 0


class ModelType(Enum):
    ideal = "ideal"
    channel = "channel"
    static_graph = "static_graph"
    detailed = "detailed"
    meta_model = "meta_model"
    none = ""


class NumDevices(Enum):
    two = 2
    ten = 10
    fifty = 50
    hundred = 100
    none = 0


class TrafficConfig(Enum):
    none = 0
    cbr_broadcast_1_mps = 1  # one message per second
    cbr_broadcast_1_mpm = 2  # one message per minute
    cbr_broadcast_4_mph = 3  # four messages per hour
    poisson_broadcast_1_mps = 4  # one message per second
    poisson_broadcast_1_mpm = 5  # one message per minute
    poisson_broadcast_4_mph = 6  # four messages per hour
    unicast_1s_delay = 7
    unicast_5s_delay = 8
    unicast_10s_delay = 9
    deer_use_case = 10



@dataclass
class ScenarioConfiguration:
    payload_size: PayloadSizeConfig = PayloadSizeConfig.none
    num_devices: NumDevices = NumDevices.none
    model_type: ModelType = ModelType.none
    scenario_duration: ScenarioDuration = ScenarioDuration.none
    traffic_configuration: TrafficConfig = TrafficConfig.none
    network_type: NetworkModelType = NetworkModelType.none

    @property
    def scenario_id(self):
        """Create a scenario ID that includes all configuration parameters."""
        return (f"{self.model_type.name}-{self.num_devices.name}-{self.payload_size.name}-{self.scenario_duration.name}"
                f"-{self.traffic_configuration.name}-{self.network_type.name}")
