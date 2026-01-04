from dataclasses import dataclass
from enum import Enum


class NetworkModelType(Enum):
    simbench_ethernet = 'Ethernet'
    simbench_lte = 'LTE'
    simbench_lte450 = 'LTE450'
    simbench_5g = 'Net5G'
    evaluation_ethernet = 'EvaluationNetworkEthernet'
    evaluation_lte = 'EvaluationNetworkLTE'
    evaluation_lte450 = 'EvaluationNetworkLTE450'
    evaluation_5g = 'EvaluationNetwork5G'
    none = ''


class PayloadSizeConfig(Enum):
    small = 8
    medium = 100
    large = 200
    none = 0


class ScenarioDuration(Enum):
    one_min = 60 * 1000
    five_min = 5 * 60 * 1000
    ten_min = 10 * 60 * 1000
    thirty_min = 30 * 60 * 1000
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
    five = 5
    ten = 10
    twenty = 20
    fifty = 50
    hundred = 100
    thousand = 1000
    none = 0


class TrafficConfig(Enum):
    none = 0

    cbr_broadcast_1_mps = 1  # one message per second
    cbr_broadcast_1_mpm = 2  # one message per minute
    cbr_broadcast_4_mph = 3  # four messages per hour

    # poisson_broadcast_[frequency]_[seed]
    poisson_broadcast_1_mps_1 = 4  # one message per second
    poisson_broadcast_1_mpm_1 = 5  # one message per minute

    poisson_broadcast_1_mps_2 = 6  # one message per second
    poisson_broadcast_1_mpm_2 = 7  # one message per minute

    unicast_1s_delay = 8
    unicast_5s_delay = 9
    unicast_10s_delay = 10

    # central_dsb_[frequency]_[processing_delay]_[response_ratio]
    central_dsb_1mpm_5s_50p = 11
    central_dsb_5mpm_30s_75 = 12
    central_dsb_10mph_60s_25p = 13

    deer_use_case = 14


class Substitution(Enum):
    disabled = 0
    enabled = 1


class TestTrainSplit(Enum):
    none = 0
    parametrization_split = 1
    traffic_load_split = 2
    technology_split = 3
    scale_split = 4
    traffic_model_split = 5


class AmountOfScenariosTrainingData(Enum):
    one = 1
    ten = 10
    all = 1000000


class ClusterDistanceThreshold(Enum):
    none = 0
    zero_one = 0.1
    half = 0.5
    one = 1
    two = 2
    three = 3
    five = 5


class BatchSizeIPupa(Enum):
    none = 0
    ten = 10
    fifty = 50
    hundred = 100
    hundred_fifty = 150
    two_hundred = 200


class LearningRateWeighting(Enum):
    none = 0
    small = 0.1
    small_medium = 0.4
    center = 0.5
    large_medium = 0.6
    large = 0.9


class ButterflyThresholdValue(Enum):
    none = 0
    small = 0.1
    small_medium = 0.4
    center = 0.5
    large_medium = 0.6
    large = 0.9


class SubstitutionPriority(Enum):
    none = 'none'
    error_trend = 'error_trend'
    error_level = 'error_level'
    cluster_distance = 'cluster_distance'
    topology_stability = 'topology_stability'


class PredictionModelType(Enum):
    none = 'none'
    decision_tree_regressor = 'decision_tree_regressor'
    random_forest_regressor = 'random_forest_regressor'


@dataclass
class ScenarioConfiguration:
    payload_size: PayloadSizeConfig = PayloadSizeConfig.none
    num_devices: NumDevices = NumDevices.none
    model_type: ModelType = ModelType.none
    scenario_duration: ScenarioDuration = ScenarioDuration.none
    traffic_configuration: TrafficConfig = TrafficConfig.none
    network_type: NetworkModelType = NetworkModelType.none

    # specific for meta-model
    substitution: Substitution = Substitution.disabled
    amount_of_scenarios_in_training_data: AmountOfScenariosTrainingData = AmountOfScenariosTrainingData.all
    test_train_split: TestTrainSplit = TestTrainSplit.none
    i_pupa: BatchSizeIPupa = BatchSizeIPupa.none
    cluster_distance_threshold: ClusterDistanceThreshold = ClusterDistanceThreshold.none
    learning_rate_weighting: LearningRateWeighting = LearningRateWeighting.none
    butterfly_threshold_value: ButterflyThresholdValue = ButterflyThresholdValue.none
    substitution_priority: SubstitutionPriority = SubstitutionPriority.none

    prediction_model_type: PredictionModelType = PredictionModelType.none

    run: int = 0

    @property
    def scenario_id(self):
        """Create a scenario ID that includes all configuration parameters."""
        return (f"{self.model_type.name}-{self.num_devices.name}-{self.payload_size.name}-{self.scenario_duration.name}"
                f"-{self.traffic_configuration.name}-{self.network_type.name}-{self.cluster_distance_threshold.name}-"
                f"{self.substitution.name}-{self.amount_of_scenarios_in_training_data.name}-{self.test_train_split.name}-"
                f"{self.i_pupa.name}-{self.learning_rate_weighting.name}-{self.butterfly_threshold_value.name}"
                f"-{self.substitution_priority.name}-{self.prediction_model_type.name}-{self.run}")

    @property
    def omnet_config(self):
        return f"{self.network_type.value}_{self.num_devices.value}"

    @classmethod
    def from_scenario_id(cls, scenario_id: str) -> 'ScenarioConfiguration':
        try:
            split_id = scenario_id.split('-')
            if len(split_id) == 15:
                (model_str, devices_str, payload_str, duration_str, traffic_str, network_str, cl_thr_str,
                 substitution_str,
                 amount_of_scen_str, test_train_split_str,
                 i_pupa, learning_rate, butterfly_threshold_value, substitution_priority, run) = split_id
                prediction_model_type_str = PredictionModelType.none.name
            else:
                (model_str, devices_str, payload_str, duration_str, traffic_str, network_str, cl_thr_str,
                 substitution_str,
                 amount_of_scen_str, test_train_split_str,
                 i_pupa, learning_rate, butterfly_threshold_value,
                 substitution_priority, prediction_model_type_str, run) = split_id
            return cls(
                model_type=ModelType[model_str],
                num_devices=NumDevices[devices_str],
                payload_size=PayloadSizeConfig[payload_str],
                scenario_duration=ScenarioDuration[duration_str],
                traffic_configuration=TrafficConfig[traffic_str],
                network_type=NetworkModelType[network_str],
                cluster_distance_threshold=ClusterDistanceThreshold[cl_thr_str],
                substitution=Substitution[substitution_str],
                amount_of_scenarios_in_training_data=AmountOfScenariosTrainingData[amount_of_scen_str],
                test_train_split=TestTrainSplit[test_train_split_str],
                i_pupa=BatchSizeIPupa[i_pupa],
                learning_rate_weighting=LearningRateWeighting[learning_rate],
                butterfly_threshold_value=ButterflyThresholdValue[butterfly_threshold_value],
                substitution_priority=SubstitutionPriority[substitution_priority],
                prediction_model_type=PredictionModelType[prediction_model_type_str],
                run=run
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid scenario_id format or value: {scenario_id}") from e
