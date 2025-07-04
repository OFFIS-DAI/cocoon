import asyncio
import logging
import os
import random
import time
from typing import Dict, Optional

import pandas as pd
import psutil
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.external_coupling import ExternalSchedulingContainer
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler, CommunicationScheduler
from integration_environment.messages import TrafficMessage, deer_message_classes
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    PoissonSenderRole, UnicastSenderRole, FlexAgentRole, AggregatorAgentRole
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


def get_training_df(scenario_configuration: ScenarioConfiguration, same_technology=True):
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(current_dir, 'cocoon_training_data')

    existing_configurations = [ScenarioConfiguration.from_scenario_id(f.split('.')[0])
                               for f in os.listdir(training_data_path)]
    different_traffic_configs = [c for c in existing_configurations
                                 if c.traffic_configuration != scenario_configuration.traffic_configuration]
    if same_technology:
        different_traffic_configs = [c for c in different_traffic_configs
                                     if c.network_type == scenario_configuration.network_type]
    dataframes = []
    for c in different_traffic_configs:
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


def get_scenario_configurations_for_meta_model_training():
    if not os.path.exists('cocoon_training_data'):
        os.makedirs('cocoon_training_data')
    existing_configuration_ids = [f.split('.')[0] for f in os.listdir('cocoon_training_data')]
    scenario_configurations = []
    for network in [NetworkModelType.simbench_ethernet, NetworkModelType.simbench_5g]:
        for payload_size in [PayloadSizeConfig.medium]:
            for n_devices in [NumDevices.ten]:
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
        (ScenarioDuration.one_hour, TrafficConfig.cbr_broadcast_1_mpm),
        (ScenarioDuration.one_day, TrafficConfig.cbr_broadcast_4_mph),
        (ScenarioDuration.one_min, TrafficConfig.poisson_broadcast_1_mps),
        (ScenarioDuration.one_hour, TrafficConfig.poisson_broadcast_1_mpm),
        (ScenarioDuration.one_min, TrafficConfig.unicast_1s_delay),
        (ScenarioDuration.one_hour, TrafficConfig.unicast_5s_delay),
        (ScenarioDuration.one_day, TrafficConfig.unicast_10s_delay),
        (ScenarioDuration.one_day, TrafficConfig.deer_use_case)
    ]


def get_duration_traffic_list_for_screening_design():
    return [
        (ScenarioDuration.one_min, TrafficConfig.cbr_broadcast_1_mps),
        (ScenarioDuration.one_min, TrafficConfig.poisson_broadcast_1_mps),
        (ScenarioDuration.one_day, TrafficConfig.deer_use_case)
    ]


def get_scenario_configurations_for_screening_design():
    scenario_configurations = []
    for payload_size in [PayloadSizeConfig.small, PayloadSizeConfig.large]:
        for model_type in [ModelType.detailed,
                           ModelType.ideal,
                           ModelType.meta_model,
                           ModelType.channel,
                           ModelType.static_graph,
                           ]:
            if not model_type == ModelType.ideal:
                networks = [NetworkModelType.simbench_ethernet,
                            NetworkModelType.simbench_5g]
            else:
                networks = [NetworkModelType.none]
            for network in networks:
                for n_devices in [
                    NumDevices.five,
                    NumDevices.hundred
                ]:
                    for scenario_duration, traffic_config in get_duration_traffic_list_for_screening_design():
                        if model_type == ModelType.meta_model:
                            scenario_configurations.append(
                                ScenarioConfiguration(payload_size=payload_size,
                                                      num_devices=n_devices,
                                                      model_type=model_type,
                                                      scenario_duration=scenario_duration,
                                                      traffic_configuration=traffic_config,
                                                      network_type=network,
                                                      cluster_distance_threshold=ClusterDistanceThreshold.one,
                                                      i_pupa=BatchSizeIPupa.ten,
                                                      learning_rate_weighting=LearningRateWeighting.center,
                                                      butterfly_threshold_value=ButterflyThresholdValue.center,
                                                      substitution_priority=SubstitutionPriority.none))
                        else:
                            scenario_configurations.append(
                                ScenarioConfiguration(payload_size=payload_size,
                                                      num_devices=n_devices,
                                                      model_type=model_type,
                                                      scenario_duration=scenario_duration,
                                                      traffic_configuration=traffic_config,
                                                      network_type=network))
    return scenario_configurations


def get_scenario_configurations_for_central_composite_design():
    scenario_configurations = []
    payload_size = PayloadSizeConfig.medium
    n_devices = NumDevices.ten

    for model_type in [ModelType.detailed,
                       ModelType.ideal,
                       ModelType.meta_model,
                       ModelType.channel,
                       ModelType.static_graph,
                       ]:
        if not model_type == ModelType.ideal:
            network = NetworkModelType.simbench_5g
        else:
            network = NetworkModelType.none

        for scenario_duration, traffic_config in get_duration_traffic_list_for_screening_design():
            if model_type == ModelType.meta_model:
                # first, combine all cube points
                for dt in [ClusterDistanceThreshold.half, ClusterDistanceThreshold.three]:
                    for ip in [BatchSizeIPupa.fifty, BatchSizeIPupa.hundred_fifty]:
                        for lr in [LearningRateWeighting.small_medium, LearningRateWeighting.large_medium]:
                            for bt in [ButterflyThresholdValue.small_medium, ButterflyThresholdValue.large_medium]:
                                for sp in [SubstitutionPriority.error_trend, SubstitutionPriority.error_level]:
                                    scenario_configurations.append(
                                        ScenarioConfiguration(payload_size=payload_size,
                                                              num_devices=n_devices,
                                                              model_type=model_type,
                                                              scenario_duration=scenario_duration,
                                                              traffic_configuration=traffic_config,
                                                              network_type=network,
                                                              cluster_distance_threshold=dt,
                                                              i_pupa=ip,
                                                              learning_rate_weighting=lr,
                                                              butterfly_threshold_value=bt,
                                                              substitution_priority=sp))

                dt_center = ClusterDistanceThreshold.two
                ip_center = BatchSizeIPupa.hundred
                lr_center = LearningRateWeighting.center
                bt_center = ButterflyThresholdValue.center
                sp_center = SubstitutionPriority.none

                for dt in [ClusterDistanceThreshold.zero_one,
                           ClusterDistanceThreshold.five]:
                    scenario_configurations.append(
                        ScenarioConfiguration(payload_size=payload_size,
                                              num_devices=n_devices,
                                              model_type=model_type,
                                              scenario_duration=scenario_duration,
                                              traffic_configuration=traffic_config,
                                              network_type=network,
                                              cluster_distance_threshold=dt,
                                              i_pupa=ip_center,
                                              learning_rate_weighting=lr_center,
                                              butterfly_threshold_value=bt_center,
                                              substitution_priority=sp_center))
                for ip in [BatchSizeIPupa.ten, BatchSizeIPupa.two_hundred]:
                    scenario_configurations.append(
                        ScenarioConfiguration(payload_size=payload_size,
                                              num_devices=n_devices,
                                              model_type=model_type,
                                              scenario_duration=scenario_duration,
                                              traffic_configuration=traffic_config,
                                              network_type=network,
                                              cluster_distance_threshold=dt_center,
                                              i_pupa=ip,
                                              learning_rate_weighting=lr_center,
                                              butterfly_threshold_value=bt_center,
                                              substitution_priority=sp_center))
                for lr in [LearningRateWeighting.small, LearningRateWeighting.large]:
                    scenario_configurations.append(
                        ScenarioConfiguration(payload_size=payload_size,
                                              num_devices=n_devices,
                                              model_type=model_type,
                                              scenario_duration=scenario_duration,
                                              traffic_configuration=traffic_config,
                                              network_type=network,
                                              cluster_distance_threshold=dt_center,
                                              i_pupa=ip_center,
                                              learning_rate_weighting=lr,
                                              butterfly_threshold_value=bt_center,
                                              substitution_priority=sp_center))
                for bt in [ButterflyThresholdValue.small, ButterflyThresholdValue.large]:
                    scenario_configurations.append(
                        ScenarioConfiguration(payload_size=payload_size,
                                              num_devices=n_devices,
                                              model_type=model_type,
                                              scenario_duration=scenario_duration,
                                              traffic_configuration=traffic_config,
                                              network_type=network,
                                              cluster_distance_threshold=dt_center,
                                              i_pupa=ip_center,
                                              learning_rate_weighting=lr_center,
                                              butterfly_threshold_value=bt,
                                              substitution_priority=sp_center))
                for sp in [SubstitutionPriority.cluster_distance, SubstitutionPriority.topology_stability]:
                    scenario_configurations.append(
                        ScenarioConfiguration(payload_size=payload_size,
                                              num_devices=n_devices,
                                              model_type=model_type,
                                              scenario_duration=scenario_duration,
                                              traffic_configuration=traffic_config,
                                              network_type=network,
                                              cluster_distance_threshold=dt_center,
                                              i_pupa=ip_center,
                                              learning_rate_weighting=lr_center,
                                              butterfly_threshold_value=bt_center,
                                              substitution_priority=sp))
            else:
                scenario_configurations.append(
                    ScenarioConfiguration(payload_size=payload_size,
                                          num_devices=n_devices,
                                          model_type=model_type,
                                          scenario_duration=scenario_duration,
                                          traffic_configuration=traffic_config,
                                          network_type=network))
    return scenario_configurations


def get_scheduler(scenario_configuration: ScenarioConfiguration,
                  container_mapping: Dict[str, ExternalSchedulingContainer]) -> Optional[CommunicationScheduler]:
    if scenario_configuration.model_type == ModelType.ideal:
        return IdealCommunicationScheduler(container_mapping=container_mapping,
                                           scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.channel:
        return ChannelModelScheduler(container_mapping=container_mapping,
                                     scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                     topology_file_name=f'network_definitions/channel_'
                                                        f'{scenario_configuration.network_type.name}.json')
    elif scenario_configuration.model_type == ModelType.static_graph:
        return StaticDelayGraphModelScheduler(container_mapping=container_mapping,
                                              scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                              topology_file_name=f'network_definitions/static_delay_graph_'
                                                                 f'{scenario_configuration.network_type.name}.json')
    elif scenario_configuration.model_type == ModelType.detailed:
        return DetailedModelScheduler(container_mapping=container_mapping,
                                      scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                      config_name=scenario_configuration.network_type.value,
                                      inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                      simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                      omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')
    elif scenario_configuration.model_type == ModelType.meta_model:
        return MetaModelScheduler(container_mapping=container_mapping,
                                  scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                  config_name=scenario_configuration.network_type.value,
                                  inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                  simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                  omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                  training_df=get_training_df(scenario_configuration),
                                  in_training_mode=False,
                                  output_file_name=f'results/cocoon_{scenario_configuration.scenario_id}.csv',
                                  cluster_distance_threshold=scenario_configuration.cluster_distance_threshold.value,
                                  i_pupa=scenario_configuration.i_pupa.value
                                  )
    elif scenario_configuration.model_type == ModelType.meta_model_training:
        return MetaModelScheduler(container_mapping=container_mapping,
                                  scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                  config_name=scenario_configuration.network_type.value,
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
    for n_agents in range(scenario_configuration.num_devices.value - 1):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        cbr_receiver_role = ReceiverRole()
        cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
        container.register(cbr_receiver_role_agent)
        receiver_addresses.append(cbr_receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node{scenario_configuration.num_devices.value - 1}',
                                          codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    container_mapping[f'node{scenario_configuration.num_devices.value - 1}'] = container2

    return container_mapping


async def initialize_poisson_broadcast_agents(clock: ExternalClock,
                                              results_recorder: ResultsRecorder,
                                              scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []
    for n_agents in range(scenario_configuration.num_devices.value - 1):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        receiver_role = ReceiverRole()
        receiver_role_agent = agent_composed_of(receiver_role, ResultsRecorderRole(results_recorder))
        container.register(receiver_role_agent)
        receiver_addresses.append(receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node{scenario_configuration.num_devices.value - 1}',
                                          codec=my_codec, clock=clock)
    poisson_sender_role_agent = agent_composed_of(
        PoissonSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(poisson_sender_role_agent)

    container_mapping[f'node{scenario_configuration.num_devices.value - 1}'] = container2

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


async def run_scenario_config(scenario_configuration: ScenarioConfiguration,
                              run: int = 0):
    scenario_configuration.run = run

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)
    clock = ExternalClock(start_time=0)

    container_mapping = {}
    if scenario_configuration.traffic_configuration in [TrafficConfig.cbr_broadcast_1_mps,
                                                        TrafficConfig.cbr_broadcast_1_mpm,
                                                        TrafficConfig.cbr_broadcast_4_mph]:
        container_mapping = \
            await initialize_constant_bitrate_broadcast_agents(clock=clock,
                                                               results_recorder=results_recorder,
                                                               scenario_configuration=scenario_configuration)
    elif scenario_configuration.traffic_configuration in [TrafficConfig.poisson_broadcast_1_mps,
                                                          TrafficConfig.poisson_broadcast_1_mpm,
                                                          TrafficConfig.poisson_broadcast_4_mph]:
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

    scheduler = get_scheduler(scenario_configuration=scenario_configuration,
                              container_mapping=container_mapping)

    if scheduler is not None:
        print(f'Running scenario with config: {scenario_configuration.scenario_id}')

        timeout_seconds = 900 if scenario_configuration.model_type == ModelType.meta_model_training \
            else 300  # 5 minutes timeout

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
                    'omnetpp', 'opp_run', 'inet', 'simu5g',
                    'simbenchnetwork', 'mango', 'ned'
                ]):
                    logger.debug(f"Killing simulation process {process_info['pid']}: {process_info['name']}")
                    proc.kill()
                    await asyncio.sleep(0.1)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    except Exception as e:
        logger.error(f"Error killing OMNeT++ processes: {e}")


async def run_benchmark_suite_screening():
    num_repetitions = 1
    # Check if 'results' folder exists, create if it doesn't
    if not os.path.exists('results'):
        os.makedirs('results')
    else:
        # If folder exists, remove all files in it
        for f in os.listdir('results'):
            file_path = os.path.join('results', f)
            if os.path.isfile(file_path):  # Only remove files, not subdirectories
                os.remove(file_path)

    meta_model_training_configs = get_scenario_configurations_for_meta_model_training()
    evaluation_configs = get_scenario_configurations_for_screening_design()
    face_centered_central_composite_design_configs = get_scenario_configurations_for_central_composite_design()

    num_scen = (len(meta_model_training_configs) + len(evaluation_configs) +
                len(face_centered_central_composite_design_configs))

    print(f'Phase 0: {len(meta_model_training_configs)} scenarios for meta-model training.\n '
          f'Phase 1: {len(evaluation_configs)} scenarios for model comparison. \n'
          f'Phase 2: {len(face_centered_central_composite_design_configs)} scenarios for meta-model optimization.'
          f'Worst case execution time: {(len(meta_model_training_configs) + len(evaluation_configs)) * 5} minutes /'
          f'{num_scen * 5 / 60} hours.')

    for r in range(num_repetitions):
        for scenario_configuration in (meta_model_training_configs + evaluation_configs):
            await run_scenario_config(scenario_configuration=scenario_configuration, run=r)


if __name__ == "__main__":
    asyncio.run(run_benchmark_suite_screening())
