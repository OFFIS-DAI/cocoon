from dataclasses import dataclass
from enum import Enum


class NetworkModelType(Enum):
    simbench_ethernet = 'Ethernet'
    simbench_lte = 'LTE'
    simbench_lte450 = 'LTE450'
    simbench_5g = 'Net5G'


class PayloadSizeConfig(Enum):
    small = 8
    medium = 100
    large = 200
    none = 0


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


class TrafficConfig(Enum):
    cbr_broadcast_1_mps = 0  # one message per second
    cbr_broadcast_1_mpm = 1  # one message per minute
    cbr_broadcast_4_mph = 2  # four messages per hour
    poisson_broadcast_1_mps = 3  # one message per second
    poisson_broadcast_1_mpm = 4  # one message per minute
    poisson_broadcast_4_mph = 5  # four messages per hour
    unicast_1s_delay = 6
    unicast_5s_delay = 7
    unicast_10s_delay = 8
    deer_use_case = 9


@dataclass
class ScenarioConfiguration:
    payload_size: PayloadSizeConfig = PayloadSizeConfig.small
    num_devices: NumDevices = NumDevices.two
    model_type: ModelType = ModelType.ideal
    scenario_duration: ScenarioDuration = ScenarioDuration.one_min
    traffic_configuration: TrafficConfig = TrafficConfig.cbr_broadcast_1_mps
    network_type: NetworkModelType = NetworkModelType.simbench_ethernet

    @property
    def scenario_id(self):
        """Create a scenario ID that includes all configuration parameters."""
        return (f"{self.model_type.name}-{self.num_devices.name}-{self.payload_size.name}-{self.scenario_duration.name}"
                f"-{self.traffic_configuration.name}-{self.network_type.name}")
