import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
import pandas as pd
import psutil
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.external_coupling import ExternalSchedulingContainer
from mango.container.factory import create_external_coupling
from pyDOE3 import *

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler, CommunicationScheduler
from integration_environment.model_comparison.network_definitions.channel_model_network_generator import \
    parse_ned_to_channel_topology
from integration_environment.roles import *
from integration_environment.scenario_configuration import *

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())
for deer_message_class in deer_message_classes:
    my_codec.add_serializer(*deer_message_class.__serializer__())

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_training_df(scenario_configuration: ScenarioConfiguration):
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(current_dir, 'cocoon_training_data')

    existing_configurations = [ScenarioConfiguration.from_scenario_id(f.split('.')[0])
                               for f in os.listdir(training_data_path)]
    # get string of traffic model
    search_str = ''
    if 'cbr' in scenario_configuration.traffic_configuration.name:
        search_str = 'cbr'
    if 'poisson' in scenario_configuration.traffic_configuration.name:
        search_str = 'poisson'
    if 'unicast' in scenario_configuration.traffic_configuration.name:
        search_str = 'unicast'
    if 'central_dsb' in scenario_configuration.traffic_configuration.name:
        search_str = 'central_dsb'

    selected_traffic_configs_for_training = []
    if scenario_configuration.test_train_split == TestTrainSplit.parametrization_split:
        # use same base traffic model but with different params (+ same technology)
        selected_traffic_configs_for_training = [c for c in existing_configurations
                                                 if (search_str in c.traffic_configuration.name and
                                                     c.network_type == scenario_configuration.network_type and
                                                     c is not scenario_configuration)]
    elif scenario_configuration.test_train_split == TestTrainSplit.traffic_load_split:
        # use same base traffic model but with different message volume (+ same technology)
        selected_traffic_configs_for_training = [c for c in existing_configurations
                                                 if (search_str in c.traffic_configuration.name and
                                                     c.network_type == scenario_configuration.network_type and
                                                     c.traffic_configuration is not
                                                     scenario_configuration.traffic_configuration)]
    elif scenario_configuration.test_train_split == TestTrainSplit.technology_split:
        # use same base traffic model (with all possible parameters) but with different network technology
        selected_traffic_configs_for_training = [c for c in existing_configurations
                                                 if (search_str in c.traffic_configuration.name and
                                                     c.network_type is not scenario_configuration.network_type)]
    elif scenario_configuration.test_train_split == TestTrainSplit.scale_split:
        # use same technology and same base traffic model, but different amount of devices
        selected_traffic_configs_for_training = [c for c in existing_configurations
                                                 if (search_str in c.traffic_configuration.name and
                                                     c.network_type == scenario_configuration.network_type and
                                                     c.num_devices is not scenario_configuration.num_devices)]
    elif scenario_configuration.test_train_split == TestTrainSplit.traffic_model_split:
        # use same technology, but different traffic model
        selected_traffic_configs_for_training = [c for c in existing_configurations
                                                 if (search_str not in c.traffic_configuration.name and
                                                     c.network_type == scenario_configuration.network_type)]
    if len(selected_traffic_configs_for_training) > scenario_configuration.amount_of_scenarios_in_training_data.value:
        selected_traffic_configs_for_training = random.sample(selected_traffic_configs_for_training,
                                                              scenario_configuration.amount_of_scenarios_in_training_data.value)
    dataframes = []
    for c in selected_traffic_configs_for_training:
        try:
            csv_path = os.path.join(training_data_path, f'{c.scenario_id}.csv')
            df = pd.read_csv(csv_path)
        except Exception as e:
            continue
        dataframes.append(df)
    if len(dataframes) == 0:
        return pd.DataFrame.empty
    complete_df = pd.concat(dataframes)
    complete_df.dropna(subset=['actual_delay_ms'], inplace=True)
    return complete_df


def get_scenario_configurations_for_phase_0():
    if not os.path.exists('cocoon_training_data'):
        os.makedirs('cocoon_training_data')
    existing_configuration_ids = [f.split('.')[0] for f in os.listdir('cocoon_training_data')]
    scenario_configurations = []
    for network in [NetworkModelType.evaluation_ethernet,
                    NetworkModelType.evaluation_5g,
                    NetworkModelType.evaluation_lte,
                    NetworkModelType.evaluation_lte450]:
        payload_size = PayloadSizeConfig.medium
        for n_devices in [NumDevices.five, NumDevices.ten, NumDevices.fifty]:
            for scenario_duration, traffic_config in get_duration_traffic_list_meta_model_training():
                config = ScenarioConfiguration(payload_size=payload_size,
                                               num_devices=n_devices,
                                               model_type=ModelType.meta_model_training,
                                               scenario_duration=scenario_duration,
                                               traffic_configuration=traffic_config,
                                               network_type=network)
                if config.scenario_id not in existing_configuration_ids:
                    scenario_configurations.append(config)
    return scenario_configurations


def get_duration_traffic_list_meta_model_training():
    return [
        (ScenarioDuration.one_min, TrafficConfig.cbr_broadcast_1_mps),
        (ScenarioDuration.thirty_min, TrafficConfig.cbr_broadcast_1_mpm),
        (ScenarioDuration.thirty_min, TrafficConfig.cbr_broadcast_4_mph),
        (ScenarioDuration.one_min, TrafficConfig.poisson_broadcast_1_mps_1),
        (ScenarioDuration.thirty_min, TrafficConfig.poisson_broadcast_1_mpm_1),
        (ScenarioDuration.one_min, TrafficConfig.poisson_broadcast_1_mps_2),
        (ScenarioDuration.thirty_min, TrafficConfig.poisson_broadcast_1_mpm_2),
        (ScenarioDuration.one_min, TrafficConfig.unicast_1s_delay),
        (ScenarioDuration.thirty_min, TrafficConfig.unicast_5s_delay),
        (ScenarioDuration.thirty_min, TrafficConfig.unicast_10s_delay),
        (ScenarioDuration.thirty_min, TrafficConfig.central_dsb_1mpm_5s_50p),
        (ScenarioDuration.thirty_min, TrafficConfig.central_dsb_5mpm_30s_75),
        (ScenarioDuration.one_day, TrafficConfig.central_dsb_10mph_60s_25p)
    ]


def get_duration_traffic_list_for_screening_design():
    return [
        (ScenarioDuration.one_min, TrafficConfig.cbr_broadcast_1_mps),
        (ScenarioDuration.one_min, TrafficConfig.poisson_broadcast_1_mps_1),
        (ScenarioDuration.thirty_min, TrafficConfig.central_dsb_1mpm_5s_50p)
    ]


def get_scenario_configurations_for_phase_2():
    scenario_configurations = []
    payload_size = PayloadSizeConfig.medium
    for model_type in [ModelType.detailed,
                       ModelType.ideal,
                       ModelType.meta_model,
                       ModelType.channel,
                       ModelType.static_graph,
                       ]:
        if not model_type == ModelType.ideal:
            networks = [NetworkModelType.evaluation_ethernet,
                        NetworkModelType.evaluation_5g,
                        NetworkModelType.evaluation_lte,
                        NetworkModelType.evaluation_lte450]
        else:
            networks = [NetworkModelType.none]
        for network in networks:
            for n_devices in [
                NumDevices.five,
                NumDevices.ten,
                NumDevices.twenty,
                NumDevices.fifty
            ]:
                for traffic_config in [TrafficConfig.cbr_broadcast_1_mps,
                                       TrafficConfig.cbr_broadcast_1_mpm,
                                       TrafficConfig.poisson_broadcast_1_mps_1,
                                       TrafficConfig.poisson_broadcast_1_mpm_1,
                                       TrafficConfig.central_dsb_1mpm_5s_50p,
                                       TrafficConfig.central_dsb_5mpm_30s_75
                                       ]:
                    for scenario_duration in [
                        ScenarioDuration.one_min,
                        ScenarioDuration.thirty_min
                    ]:
                        if model_type == ModelType.meta_model:
                            for tts in [TestTrainSplit.parametrization_split,
                                        TestTrainSplit.traffic_load_split,
                                        TestTrainSplit.technology_split,
                                        TestTrainSplit.scale_split,
                                        TestTrainSplit.traffic_model_split]:
                                scenario_configurations.append(
                                    ScenarioConfiguration(payload_size=payload_size,
                                                          num_devices=n_devices,
                                                          model_type=model_type,
                                                          scenario_duration=scenario_duration,
                                                          traffic_configuration=traffic_config,
                                                          network_type=network,
                                                          cluster_distance_threshold=ClusterDistanceThreshold.five,
                                                          i_pupa=BatchSizeIPupa.fifty,
                                                          learning_rate_weighting=LearningRateWeighting.small,
                                                          butterfly_threshold_value=ButterflyThresholdValue.small,
                                                          substitution_priority=SubstitutionPriority.error_trend,
                                                          test_train_split=tts))
                        else:
                            scenario_configurations.append(
                                ScenarioConfiguration(payload_size=payload_size,
                                                      num_devices=n_devices,
                                                      model_type=model_type,
                                                      scenario_duration=scenario_duration,
                                                      traffic_configuration=traffic_config,
                                                      network_type=network))
    return scenario_configurations


def get_central_composite_design(substitution: Substitution = Substitution.enabled):
    # using a face-centered central composite design
    if substitution == Substitution.enabled:
        central_composite_design = pd.DataFrame(ccdesign(n=6,
                                                         face='ccf'))
        central_composite_design.rename(columns={0: 'C-DT',
                                                 1: 'C-IP',
                                                 2: 'C-LR',
                                                 3: 'C-BT',
                                                 4: 'C-SP',
                                                 5: 'C-TTS'},
                                        inplace=True)
    else:
        central_composite_design = pd.DataFrame(ccdesign(n=5,
                                                         face='ccf'))
        central_composite_design.rename(columns={0: 'C-DT',
                                                 1: 'C-IP',
                                                 2: 'C-LR',
                                                 3: 'C-TTS',
                                                 4: 'C-AT'},
                                        inplace=True)
    factor_mappings = {
        'C-DT': {
            -1: ClusterDistanceThreshold.one,  # 1 (cube point low)
            0: ClusterDistanceThreshold.three,  # 3 (center point)
            1: ClusterDistanceThreshold.five,  # 5 (cube point high)
        },
        'C-IP': {
            -1: BatchSizeIPupa.fifty,  # 50 (cube point low)
            0: BatchSizeIPupa.hundred,  # 100 (center point)
            1: BatchSizeIPupa.hundred_fifty,  # 150 (cube point high)
        },
        'C-LR': {
            -1: LearningRateWeighting.small,  # 0.1 (cube point low)
            0: LearningRateWeighting.center,  # 0.5 (center point)
            1: LearningRateWeighting.large,  # 0.9 (cube point high)
        },
        'C-TTS': {
            -1: TestTrainSplit.scale_split,
            0: TestTrainSplit.parametrization_split,
            1: TestTrainSplit.technology_split,
        }
    }
    if substitution == Substitution.enabled:
        factor_mappings['C-SP'] = {
            -1: SubstitutionPriority.error_level,  # error_level (cube point low)
            0: SubstitutionPriority.none,  # none (center point)
            1: SubstitutionPriority.error_trend,  # error_trend (cube point high)
        }
        factor_mappings['C-BT'] = {
            -1: ButterflyThresholdValue.small,  # 0.1 (cube point low)
            0: ButterflyThresholdValue.center,  # 0.5 (center point)
            1: ButterflyThresholdValue.large,  # 0.9 (cube point high)
        }
    else:
        factor_mappings['C-AT'] = {
            -1: AmountOfScenariosTrainingData.one,
            0: AmountOfScenariosTrainingData.ten,
            1: AmountOfScenariosTrainingData.all
        }
    for col in central_composite_design.columns:
        # Map to enum values
        central_composite_design[col] = central_composite_design[col].map(factor_mappings[col])
    return central_composite_design


def get_scenario_configurations_for_phase_1():
    scenario_configurations = []
    existing_configuration_ids = [f.split('messages_')[1].split('.')[0] for f in os.listdir('results/phase1') if
                                  'messages_' in f]
    payload_size = PayloadSizeConfig.medium
    n_devices = NumDevices.five
    networks = [NetworkModelType.evaluation_5g, NetworkModelType.evaluation_ethernet]

    central_composite_design = get_central_composite_design()
    central_composite_design_without_substitution = get_central_composite_design(substitution=Substitution.disabled)

    for scenario_duration, traffic_config in get_duration_traffic_list_for_screening_design():
        for network in networks:
            # detailed model
            config = ScenarioConfiguration(payload_size=payload_size,
                                           num_devices=n_devices,
                                           model_type=ModelType.detailed,
                                           scenario_duration=scenario_duration,
                                           traffic_configuration=traffic_config,
                                           network_type=network)
            if config.scenario_id not in existing_configuration_ids:
                scenario_configurations.append(config)
            # meta-model
            for i, row in central_composite_design.iterrows():
                config = ScenarioConfiguration(payload_size=payload_size,
                                               num_devices=n_devices,
                                               model_type=ModelType.meta_model,
                                               scenario_duration=scenario_duration,
                                               traffic_configuration=traffic_config,
                                               network_type=network,
                                               cluster_distance_threshold=row['C-DT'],
                                               i_pupa=row['C-IP'],
                                               learning_rate_weighting=row['C-LR'],
                                               butterfly_threshold_value=row['C-BT'],
                                               substitution_priority=row['C-SP'],
                                               test_train_split=row['C-TTS'],
                                               substitution=Substitution.enabled)
                if config.scenario_id not in existing_configuration_ids:
                    scenario_configurations.append(config)
            for i, row in central_composite_design_without_substitution.iterrows():
                config = ScenarioConfiguration(payload_size=payload_size,
                                               num_devices=n_devices,
                                               model_type=ModelType.meta_model,
                                               scenario_duration=scenario_duration,
                                               traffic_configuration=traffic_config,
                                               network_type=network,
                                               cluster_distance_threshold=row['C-DT'],
                                               i_pupa=row['C-IP'],
                                               learning_rate_weighting=row['C-LR'],
                                               butterfly_threshold_value=ButterflyThresholdValue.none,
                                               substitution_priority=SubstitutionPriority.none,
                                               amount_of_scenarios_in_training_data=row['C-AT'],
                                               test_train_split=row['C-TTS'],
                                               substitution=Substitution.disabled)
                if config.scenario_id not in existing_configuration_ids:
                    scenario_configurations.append(config)

    return scenario_configurations


def get_scheduler(scenario_configuration: ScenarioConfiguration,
                  container_mapping: Dict[str, ExternalSchedulingContainer],
                  phase: int) -> Optional[CommunicationScheduler]:
    if scenario_configuration.model_type == ModelType.ideal:
        return IdealCommunicationScheduler(container_mapping=container_mapping,
                                           scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.channel:
        if 'Ethernet' in scenario_configuration.omnet_config:
            ned_name = 'EvaluationNetwork_Ethernet.ned'
            propagation_speed_mps = 2e8
            processing_delay_ms = random.uniform(0.02, 0.12)
        elif '5G' in scenario_configuration.omnet_config:
            ned_name = 'EvaluationNetwork_5G.ned'
            propagation_speed_mps = 3e8
            processing_delay_ms = 2
        elif 'LTE' in scenario_configuration.omnet_config:
            ned_name = 'EvaluationNetwork_LTE.ned'
            propagation_speed_mps = 3e8
            processing_delay_ms = 5
        else:
            raise ValueError(f'Unknown network for configuration {scenario_configuration.omnet_config}. ')
        ned_file = Path(f'../../cocoon_omnet_project/networks/{ned_name}').read_text(encoding="utf-8")
        topology_dict = parse_ned_to_channel_topology(ned_file,
                                                      default_propagation_speed_mps=int(propagation_speed_mps),
                                                      default_processing_delay_ms=processing_delay_ms)
        return ChannelModelScheduler(container_mapping=container_mapping,
                                     scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                     topology_dict=topology_dict)
    elif scenario_configuration.model_type == ModelType.static_graph:
        return StaticDelayGraphModelScheduler(container_mapping=container_mapping,
                                              scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                              topology_file_name=f'network_definitions/static_delay_graph_'
                                                                 f'{scenario_configuration.network_type.name}.json')
    elif scenario_configuration.model_type == ModelType.detailed:
        return DetailedModelScheduler(container_mapping=container_mapping,
                                      scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                      config_name=scenario_configuration.omnet_config,
                                      inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                      simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                      omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')
    elif scenario_configuration.model_type == ModelType.meta_model:
        return MetaModelScheduler(container_mapping=container_mapping,
                                  scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                  config_name=scenario_configuration.omnet_config,
                                  inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                  simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                  omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                  training_df=get_training_df(scenario_configuration),
                                  in_training_mode=False,
                                  output_file_name=f'results/phase{phase}/cocoon_{scenario_configuration.scenario_id}.csv'
                                  if phase is not None else f'results/cocoon_{scenario_configuration.scenario_id}.csv',
                                  cluster_distance_threshold=scenario_configuration.cluster_distance_threshold.value,
                                  i_pupa=scenario_configuration.i_pupa.value,
                                  butterfly_threshold_value=scenario_configuration.butterfly_threshold_value.value,
                                  learning_rate_weighting=scenario_configuration.learning_rate_weighting.value,
                                  substitution_priority=scenario_configuration.substitution_priority.value,
                                  substitution_enabled=True
                                  if scenario_configuration.substitution == Substitution.enabled else False
                                  )
    elif scenario_configuration.model_type == ModelType.meta_model_training:
        return MetaModelScheduler(container_mapping=container_mapping,
                                  scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                  config_name=scenario_configuration.omnet_config,
                                  inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                  simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                  omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                  in_training_mode=True,
                                  output_file_name=f'cocoon_training_data/{scenario_configuration.scenario_id}.csv'
                                  )
    else:
        logging.warning(f'Unknown model type: {scenario_configuration.model_type}')
        return None


async def initialize_constant_bitrate_broadcast_agents(clock: ExternalClock,
                                                       results_recorder: ResultsRecorder,
                                                       scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []
    for n_agents in range(1, scenario_configuration.num_devices.value):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        cbr_receiver_role = ReceiverRole()
        cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
        container.register(cbr_receiver_role_agent)
        receiver_addresses.append(cbr_receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node0',
                                          codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    container_mapping[f'node0'] = container2

    return container_mapping


async def initialize_poisson_broadcast_agents(clock: ExternalClock,
                                              results_recorder: ResultsRecorder,
                                              scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []
    for n_agents in range(1, scenario_configuration.num_devices.value):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        receiver_role = ReceiverRole()
        receiver_role_agent = agent_composed_of(receiver_role, ResultsRecorderRole(results_recorder))
        container.register(receiver_role_agent)
        receiver_addresses.append(receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node0',
                                          codec=my_codec, clock=clock)
    poisson_sender_role_agent = agent_composed_of(
        PoissonSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(poisson_sender_role_agent)

    container_mapping[f'node0'] = container2

    return container_mapping


async def initialize_unicast_communication_agents(clock: ExternalClock,
                                                  results_recorder: ResultsRecorder,
                                                  scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addr = []

    agents = []
    for n_agents in range(scenario_configuration.num_devices.value):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        receiver_role = ReceiverRole()
        agent = agent_composed_of(receiver_role, ResultsRecorderRole(results_recorder))
        container.register(agent)
        receiver_addr.append(agent.addr)
        container_mapping[f'node{index}'] = container
        agents.append((container, agent))

    # Second pass: Add UnicastRole to each agent with addresses of all OTHER agents
    for i, (container, agent) in enumerate(agents):
        # Get receiver addresses (all agents except this one)
        receiver_addresses = [addr for j, addr in enumerate(receiver_addr) if j != i]

        # Add UnicastRole to the existing agent
        unicast_role = UnicastSenderRole(receiver_addresses=receiver_addresses,
                                         scenario_config=scenario_configuration,
                                         start_at_s=i * 10 + 1)
        agent.add_role(unicast_role)

    return container_mapping


async def initialize_central_dsb_agents(clock: ExternalClock,
                                        results_recorder: ResultsRecorder,
                                        scenario_configuration: ScenarioConfiguration):
    container_mapping = {}

    container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
    control_role = ControlDSbRole(scenario_config=scenario_configuration)

    control_agent = agent_composed_of(control_role, ResultsRecorderRole(results_recorder))
    container1.register(control_agent)

    container_mapping['node0'] = container1

    num_local_agents = scenario_configuration.num_devices.value - 1

    for i in range(1, num_local_agents + 1):
        container = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
        local_role = LocalDSbRole(scenario_config=scenario_configuration,
                                  control_address=control_agent.addr)

        local_agent = agent_composed_of(local_role, ResultsRecorderRole(results_recorder))

        container.current_start_time_of_step = time.time()
        container.register(local_agent)
        container_mapping[f'node{i}'] = container

    return container_mapping


async def initialize_deer_use_case_agents(clock: ExternalClock,
                                          results_recorder: ResultsRecorder,
                                          scenario_configuration: ScenarioConfiguration):
    container_mapping = {}

    container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=0.05)
    aggregator_agent = agent_composed_of(aggregator_role, ResultsRecorderRole(results_recorder))
    container1.register(aggregator_agent)

    container_mapping['node0'] = container1

    num_flex_agents = scenario_configuration.num_devices.value - 1
    agent_addresses = []
    flex_agent_roles = []

    baseline_values = [100] * num_flex_agents
    flex_values = [random.randint(0, 100) for _ in range(num_flex_agents)]

    for i in range(1, num_flex_agents + 1):
        container2 = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
        flex_agent_role = FlexAgentRole(aggregator_address=aggregator_agent.addr,
                                        scenario_config=scenario_configuration,
                                        can_provide_power=False if i == 2 else True,
                                        baseline_value=baseline_values[i - 2],
                                        flexibility_value=flex_values[i - 2])
        flex_agent = agent_composed_of(flex_agent_role, ResultsRecorderRole(results_recorder))

        container2.current_start_time_of_step = time.time()
        container2.register(flex_agent)

        agent_addresses.append(flex_agent.addr)
        flex_agent_roles.append(flex_agent_role)

        container_mapping[f'node{i}'] = container2

    aggregator_role.flex_agent_addresses = agent_addresses

    return container_mapping


async def run_scenario(container_mapping: Dict[str, ExternalSchedulingContainer],
                       results_recorder: ResultsRecorder,
                       scheduler: CommunicationScheduler):
    results_recorder.set_scheduler(scheduler)
    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await scheduler.scenario_finished
    results_recorder.stop_scenario_recording()


async def run_scenario_config(scenario_configuration: ScenarioConfiguration, phase: int = None,
                              run: int = 0, timeout_seconds: int = 600, output_dir: str = None):
    scenario_configuration.run = run
    if phase is not None:
        output_dir = f'results/phase{phase}'
    else:
        if output_dir:
            output_dir = f'results/{output_dir}'
        else:
            output_dir = 'results'
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration, output_dir=output_dir)
    clock = ExternalClock(start_time=0)

    container_mapping = {}
    if scenario_configuration.traffic_configuration in [TrafficConfig.cbr_broadcast_1_mps,
                                                        TrafficConfig.cbr_broadcast_1_mpm,
                                                        TrafficConfig.cbr_broadcast_4_mph]:
        container_mapping = \
            await initialize_constant_bitrate_broadcast_agents(clock=clock,
                                                               results_recorder=results_recorder,
                                                               scenario_configuration=scenario_configuration)
    elif scenario_configuration.traffic_configuration in [TrafficConfig.poisson_broadcast_1_mps_1,
                                                          TrafficConfig.poisson_broadcast_1_mpm_1,
                                                          TrafficConfig.poisson_broadcast_1_mps_2,
                                                          TrafficConfig.poisson_broadcast_1_mpm_2]:
        container_mapping = \
            await initialize_poisson_broadcast_agents(clock=clock,
                                                      results_recorder=results_recorder,
                                                      scenario_configuration=scenario_configuration)
    elif scenario_configuration.traffic_configuration in [TrafficConfig.unicast_1s_delay,
                                                          TrafficConfig.unicast_5s_delay,
                                                          TrafficConfig.unicast_10s_delay]:
        container_mapping = \
            await initialize_unicast_communication_agents(clock=clock,
                                                          results_recorder=results_recorder,
                                                          scenario_configuration=scenario_configuration)
    elif scenario_configuration.traffic_configuration == TrafficConfig.deer_use_case:
        container_mapping = \
            await initialize_deer_use_case_agents(clock=clock,
                                                  results_recorder=results_recorder,
                                                  scenario_configuration=scenario_configuration)
    elif scenario_configuration.traffic_configuration in [TrafficConfig.central_dsb_1mpm_5s_50p,
                                                          TrafficConfig.central_dsb_5mpm_30s_75,
                                                          TrafficConfig.central_dsb_10mph_60s_25p]:
        container_mapping = \
            await initialize_central_dsb_agents(clock=clock,
                                                results_recorder=results_recorder,
                                                scenario_configuration=scenario_configuration)

    scheduler = get_scheduler(scenario_configuration=scenario_configuration,
                              container_mapping=container_mapping,
                              phase=phase)

    if scheduler is not None:
        print(f'Running scenario with config: {scenario_configuration.scenario_id}')
        try:
            await asyncio.wait_for(
                run_scenario(container_mapping=container_mapping,
                             results_recorder=results_recorder,
                             scheduler=scheduler),
                timeout=timeout_seconds
            )

            print(f'Scenario {scenario_configuration.scenario_id} completed successfully')
        except asyncio.TimeoutError:
            print(
                f'ERROR: Scenario {scenario_configuration.scenario_id} timed out after {timeout_seconds} seconds')
            results_recorder.record_timeout(
                timeout_seconds=timeout_seconds,
                error_message=f"Scenario execution exceeded {timeout_seconds} second timeout"
            )
            await kill_omnet_processes()
            await asyncio.sleep(5)
        except Exception as e:
            print(f'ERROR: Scenario {scenario_configuration.scenario_id} failed with error: {e}')
            results_recorder.record_error(
                error_message=f"Scenario execution failed: {str(e)}",
                exception=e
            )
            await kill_omnet_processes()
            await asyncio.sleep(5)


async def kill_omnet_processes():
    """Kill any remaining OMNeT++ or simulation processes"""
    try:
        # Find and kill OMNeT++ processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                process_info = proc.info
                cmdline = ' '.join(process_info['cmdline'] or [])

                # Kill processes related to your simulation
                if any(keyword in cmdline.lower() for keyword in [
                    'omnetpp', 'opp_run', 'inet', 'simu5g', 'mango', 'ned'
                ]):
                    logger.debug(f"Killing simulation process {process_info['pid']}: {process_info['name']}")
                    proc.kill()
                    await asyncio.sleep(0.1)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    except Exception as e:
        logger.error(f"Error killing OMNeT++ processes: {e}")


async def run_benchmark_suite_screening(phase: int = None):
    num_repetitions = 3
    if phase != 0:
        # Check if 'results' folder exists, create if it doesn't
        if not os.path.exists(f'results/phase{phase}'):
            os.makedirs(f'results/phase{phase}')

    if phase is None or phase == 0:
        meta_model_training_configs = get_scenario_configurations_for_phase_0()
    else:
        meta_model_training_configs = []

    if phase is None or phase == 1:
        face_centered_central_composite_design_configs = get_scenario_configurations_for_phase_1()
    else:
        face_centered_central_composite_design_configs = []

    if phase is None or phase == 2:
        requirement_analysis_configs = get_scenario_configurations_for_phase_2()
    else:
        requirement_analysis_configs = []

    num_scen = (len(meta_model_training_configs) +
                len(face_centered_central_composite_design_configs) + len(requirement_analysis_configs))

    print(f'Phase 0: {len(meta_model_training_configs)} scenarios for meta-model training.\n'
          f'Phase 1: {len(face_centered_central_composite_design_configs)} scenarios for meta-model optimization. \n'
          f'Phase 2: {len(requirement_analysis_configs)} scenarios for requirement analysis. '
          f'Worst case execution time: {num_scen * num_repetitions * 10} minutes /'
          f'{num_scen * num_repetitions * 10 / 60} hours.')
    for i, scenario_configuration in enumerate(meta_model_training_configs):
        print(f'Run config {i}/{len(meta_model_training_configs)}')
        await run_scenario_config(scenario_configuration=scenario_configuration, run=0,
                                  phase=0, timeout_seconds=60 * 20)  # 20 minute timeout

    for r in range(num_repetitions):
        for i, scenario_configuration in enumerate(face_centered_central_composite_design_configs):
            print(f'Run config {i}/{len(face_centered_central_composite_design_configs)} '
                  f'in repetition {r + 1}/{num_repetitions}')
            await run_scenario_config(scenario_configuration=scenario_configuration, run=r,
                                      phase=1)
        for i, scenario_configuration in enumerate(requirement_analysis_configs):
            print(f'Run config {i}/{len(requirement_analysis_configs)} '
                  f'in repetition {r + 1}/{num_repetitions}')
            await run_scenario_config(scenario_configuration=scenario_configuration, run=r,
                                      phase=2, timeout_seconds=5 * 60)  # 5 minute timeout


if __name__ == "__main__":
    # 0: meta-model training data generation
    # 1: meta-model optimization
    # 2: base model comparison with requirement analysis
    asyncio.run(run_benchmark_suite_screening(phase=None))
