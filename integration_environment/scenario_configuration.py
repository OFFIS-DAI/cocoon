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
    one_day = 60 * 60 * 1000 * 24
    none = 0


class ModelType(Enum):
    ideal = "ideal"
    channel = "channel"
    static_graph = "static_graph"
    detailed = "detailed"
    meta_model = "meta_model"
    meta_model_training = "meta_model_training"
    none = ""


class NumDevices(Enum):
    two = 2
    ten = 10
    fifty = 50
    hundred = 100
    thousand = 1000
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


class ClusterDistanceThreshold(Enum):
    none = 0
    half = 0.5
    one = 1
    three = 3
    five = 5


class BatchSizeIPupa(Enum):
    none = 0
    ten = 10
    fifty = 50
    hundred = 1000


class LearningRateWeighting(Enum):
    none = 0
    small = 0.7
    medium = 0.7
    large = 0.9


class ButterflyThresholdValue(Enum):
    none = 0
    small = 0.5
    small_medium = 0.75
    medium = 0.8
    large = 0.9


class SubstitutionPriority(Enum):
    none = 'none'
    error_trend = 'error_trend'
    error_level = 'error_level'
    cluster_distance = 'cluster_distance'
    topology_stability = 'topology_stability'


@dataclass
class ScenarioConfiguration:
    payload_size: PayloadSizeConfig = PayloadSizeConfig.none
    num_devices: NumDevices = NumDevices.none
    model_type: ModelType = ModelType.none
    scenario_duration: ScenarioDuration = ScenarioDuration.none
    traffic_configuration: TrafficConfig = TrafficConfig.none
    network_type: NetworkModelType = NetworkModelType.none

    # specific for meta-model
    i_pupa: BatchSizeIPupa = BatchSizeIPupa.none
    cluster_distance_threshold: ClusterDistanceThreshold = ClusterDistanceThreshold.none
    learning_rate_weighting: LearningRateWeighting = LearningRateWeighting.none
    butterfly_threshold_value: ButterflyThresholdValue = ButterflyThresholdValue.none
    substitution_priority: SubstitutionPriority = SubstitutionPriority.none

    run: int = 0

    @property
    def scenario_id(self):
        """Create a scenario ID that includes all configuration parameters."""
        return (f"{self.model_type.name}-{self.num_devices.name}-{self.payload_size.name}-{self.scenario_duration.name}"
                f"-{self.traffic_configuration.name}-{self.network_type.name}-{self.cluster_distance_threshold.name}-"
                f"{self.i_pupa.name}-{self.learning_rate_weighting.name}-{self.butterfly_threshold_value.name}"
                f"-{self.substitution_priority.name}-{self.run}")

    @classmethod
    def from_scenario_id(cls, scenario_id: str) -> 'ScenarioConfiguration':
        try:
            (model_str, devices_str, payload_str, duration_str, traffic_str, network_str, cl_thr_str, i_pupa,
             learning_rate, butterfly_threshold_value, substitution_priority, run) = scenario_id.split('-')
            return cls(
                model_type=ModelType[model_str],
                num_devices=NumDevices[devices_str],
                payload_size=PayloadSizeConfig[payload_str],
                scenario_duration=ScenarioDuration[duration_str],
                traffic_configuration=TrafficConfig[traffic_str],
                network_type=NetworkModelType[network_str],
                cluster_distance_threshold=ClusterDistanceThreshold[cl_thr_str],
                i_pupa=BatchSizeIPupa[i_pupa],
                learning_rate_weighting=learning_rate,
                butterfly_threshold_value=butterfly_threshold_value,
                substitution_priority=substitution_priority,
                run=run
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid scenario_id format or value: {scenario_id}") from e
