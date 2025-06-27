import random
import time
import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock, AgentAddress
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import (IdealCommunicationScheduler, DetailedModelScheduler,
                                                                   MetaModelScheduler)
from integration_environment.messages import *
from integration_environment.model_comparison.execute_comparison import get_training_df
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ResultsRecorderRole, \
    AggregatorAgentRole, FlexAgentRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig, ModelType, \
    ScenarioDuration, TrafficConfig, NumDevices, NetworkModelType, ClusterDistanceThreshold, BatchSizeIPupa
from tests.integration_tests.utils import setup_logging

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())
for deer_message_class in deer_message_classes:
    my_codec.add_serializer(*deer_message_class.__serializer__())


@pytest.mark.asyncio
async def test_run_deer_scenario_with_different_models():
    """Time analysis"""
    scenario_duration = ScenarioDuration.one_day


    scenario_configuration = ScenarioConfiguration(payload_size=PayloadSizeConfig.none,
                                                   num_devices=NumDevices.ten,
                                                   model_type=ModelType.meta_model,
                                                   scenario_duration=scenario_duration,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   network_type=NetworkModelType.simbench_ethernet,
                                                   cluster_distance_threshold=ClusterDistanceThreshold.half,
                                                   i_pupa=BatchSizeIPupa.ten)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container_mapping = {}

    container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=0.5)
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
    training_df = get_training_df(scenario_configuration)

    communication_network_entity = (
        MetaModelScheduler(container_mapping=container_mapping,
                           inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                           simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                           config_name=scenario_configuration.network_type.value,
                           omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                           in_training_mode=False,
                           training_df=training_df,
                           cluster_distance_threshold=scenario_configuration.cluster_distance_threshold.value,
                           scenario_duration_ms=scenario_configuration.scenario_duration.value,
                           i_pupa=scenario_configuration.i_pupa.value,
                           output_file_name='results/cocoon_message_observations/production.csv'))

    results_recorder.set_scheduler(communication_network_entity)

    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    for flex_agent_role in flex_agent_roles:
        print('infeasible power request of: ', flex_agent_role.context.addr.protocol_addr,
              ': ', flex_agent_role.infeasible_requests, ' and time of re-planning msg: ',
              flex_agent_role.time_received_re_planning_message)

    scenario_configuration = ScenarioConfiguration(payload_size=PayloadSizeConfig.none,
                                                   num_devices=NumDevices.ten,
                                                   model_type=ModelType.ideal,
                                                   scenario_duration=scenario_duration,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   network_type=NetworkModelType.simbench_ethernet)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container_mapping = {}

    container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=0.5)
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

    communication_network_entity = IdealCommunicationScheduler(container_mapping=container_mapping,
                                                               scenario_duration_ms=scenario_configuration.scenario_duration.value)

    results_recorder.set_scheduler(communication_network_entity)

    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    for flex_agent_role in flex_agent_roles:
        print('infeasible power request of: ', flex_agent_role.context.addr.protocol_addr,
              ': ', flex_agent_role.infeasible_requests, ' and time of re-planning msg: ',
              flex_agent_role.time_received_re_planning_message)

    scenario_configuration = ScenarioConfiguration(payload_size=PayloadSizeConfig.none,
                                                   num_devices=NumDevices.ten,
                                                   model_type=ModelType.detailed,
                                                   scenario_duration=scenario_duration,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   network_type=NetworkModelType.simbench_ethernet)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container_mapping = {}

    container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=0.5)
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

    communication_network_entity = DetailedModelScheduler(container_mapping=container_mapping,
                                                          inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                                          config_name=scenario_configuration.network_type.value,
                                                          simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                                          omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                                          scenario_duration_ms=scenario_configuration.scenario_duration.value)

    results_recorder.set_scheduler(communication_network_entity)

    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    for flex_agent_role in flex_agent_roles:
        print('infeasible power request of: ', flex_agent_role.context.addr.protocol_addr,
              ': ', flex_agent_role.infeasible_requests, ' and time of re-planning msg: ',
              flex_agent_role.time_received_re_planning_message)
