import asyncio
import json
import random
import time

import numpy as np
import pandas as pd
import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock, AgentAddress
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler
from integration_environment.messages import *
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    AggregatorAgentRole, FlexAgentRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig, ModelType, \
    ScenarioDuration, TrafficConfig, NumDevices, NetworkModelType
from tests.integration_tests.utils import setup_logging

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())
for deer_message_class in deer_message_classes:
    my_codec.add_serializer(*deer_message_class.__serializer__())


@pytest.mark.asyncio
async def test_run_deer_scenario_with_detailed_model():
    """Analysis 1: Preparation for field trial"""
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.detailed,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   payload_size=PayloadSizeConfig.none,
                                                   scenario_duration=ScenarioDuration.one_hour,
                                                   num_devices=NumDevices.ten,
                                                   network_type=NetworkModelType.simbench_lte450)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

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

    communication_network_entity = DetailedModelScheduler(container_mapping=container_mapping,
                                                          inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                                          config_name=scenario_configuration.network_type.value,
                                                          simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                                          omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                                          scenario_duration_ms=scenario_configuration.scenario_duration.value)

    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    for flex_agent_role in flex_agent_roles:
        print('infeasible power request of: ', flex_agent_role.context.addr.protocol_addr,
              ': ', flex_agent_role.infeasible_requests, ' and time of re-planning msg: ',
              flex_agent_role.time_received_re_planning_message)


@pytest.mark.asyncio
async def test_run_deer_scenario_with_ideal_communication():
    """Analysis 2: Baseline performance with ideal communication"""
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.ideal,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   payload_size=PayloadSizeConfig.none,
                                                   scenario_duration=ScenarioDuration.one_hour,
                                                   num_devices=NumDevices.ten)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container_mapping = {}

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=0.1)
    aggregator_agent = agent_composed_of(aggregator_role, ResultsRecorderRole(results_recorder))
    container1.register(aggregator_agent)

    container_mapping['node1'] = container1

    num_flex_agents = scenario_configuration.num_devices.value - 1
    agent_addresses = []
    flex_agent_roles = []

    baseline_values = [100] * num_flex_agents
    flex_values = [random.randint(0, 100) for _ in range(num_flex_agents)]

    for i in range(2, num_flex_agents + 2):
        container2 = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
        flex_agent_role = FlexAgentRole(aggregator_address=aggregator_agent.addr,
                                        scenario_config=scenario_configuration,
                                        can_provide_power=False if i == 2 else True,
                                        baseline_value=baseline_values[i - 2],
                                        flexibility_value=flex_values[i - 2])
        flex_agent = agent_composed_of(flex_agent_role, ResultsRecorderRole(results_recorder))
        container2.register(flex_agent)

        agent_addresses.append(flex_agent.addr)
        flex_agent_roles.append(flex_agent_role)

        container_mapping[f'node{i}'] = container2

    aggregator_role.flex_agent_addresses = agent_addresses

    communication_network_entity = IdealCommunicationScheduler(container_mapping=container_mapping,
                                                               scenario_duration_ms=scenario_configuration.scenario_duration.value)

    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    for flex_agent_role in flex_agent_roles:
        print('infeasible power request of: ', flex_agent_role.context.addr.protocol_addr,
              ': ', flex_agent_role.infeasible_requests)


@pytest.mark.asyncio
async def test_run_deer_scenario_with_static_delay_model():
    """Analysis 2: Performance with static delay model communication"""
    for num_agents in [NumDevices.ten, NumDevices.hundred, NumDevices.thousand]:
        scenario_configuration = ScenarioConfiguration(model_type=ModelType.static_graph,
                                                       traffic_configuration=TrafficConfig.deer_use_case,
                                                       payload_size=PayloadSizeConfig.none,
                                                       scenario_duration=ScenarioDuration.one_hour,
                                                       num_devices=num_agents,
                                                       network_type=NetworkModelType.simbench_lte450)

        results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

        topology = {'nodes': [],
                    'links': []}

        for i in range(scenario_configuration.num_devices.value):
            topology['nodes'].append({
                'node_id': f'node{i}'
            })

        # Configuration parameters for delay distribution
        mean_delay_ms = 90.08 + num_agents.value
        std_delay_ms = 40.18

        # Add links between all pairs of nodes with delay values
        for i in range(scenario_configuration.num_devices.value):
            for j in range(i + 1, scenario_configuration.num_devices.value):
                # Generate delay from normal distribution
                delay_ms = np.random.normal(mean_delay_ms, std_delay_ms)

                # Ensure delay is positive (minimum 1ms)
                delay_ms = max(1.0, delay_ms)

                topology['links'].append({
                    "node_a": f"node{i}",
                    "node_b": f"node{j}",
                    "end-to-end-delay_ms": round(delay_ms, 2)  # Round to 2 decimal places
                })

        clock = ExternalClock(start_time=0)

        container_mapping = {}

        container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
        reactive_scheduling_time = 0
        if num_agents == NumDevices.ten:
            reactive_scheduling_time = 21
        elif num_agents == NumDevices.hundred:
            reactive_scheduling_time = 162
        elif num_agents == NumDevices.thousand:
            reactive_scheduling_time = 1628
        aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=3,
                                              reactive_scheduling_time=reactive_scheduling_time)
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
            container2.register(flex_agent)

            agent_addresses.append(flex_agent.addr)
            flex_agent_roles.append(flex_agent_role)

            container_mapping[f'node{i}'] = container2

        aggregator_role.flex_agent_addresses = agent_addresses

        communication_network_entity = StaticDelayGraphModelScheduler(container_mapping=container_mapping,
                                                                      topology_dict=topology,
                                                                      scenario_duration_ms=scenario_configuration.scenario_duration.value)

        async with activate([c for c in container_mapping.values()]) as _:
            results_recorder.start_scenario_recording()
            await communication_network_entity.scenario_finished
        results_recorder.stop_scenario_recording()

        replanning_messages = []
        for flex_agent_role in flex_agent_roles:
            replanning_messages.append(flex_agent_role.time_received_re_planning_message)
        print('Termination time for ', num_agents.value, ': ', max(replanning_messages))


