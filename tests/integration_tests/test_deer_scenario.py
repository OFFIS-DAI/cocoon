import asyncio
import json
import time

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
    ScenarioDuration, TrafficConfig, NumDevices
from tests.integration_tests.utils import setup_logging, visualize_channel_model_graph, visualize_static_graph

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())
for deer_message_class in deer_message_classes:
    my_codec.add_serializer(*deer_message_class.__serializer__())


@pytest.mark.asyncio
async def test_run_deer_scenario_with_ideal_communication():
    """Analysis 1: Baseline performance with ideal communication"""
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.ideal,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   payload_size=PayloadSizeConfig.none,
                                                   scenario_duration=ScenarioDuration.one_hour,
                                                   num_devices=NumDevices.ten)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container_mapping = {}

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=[AgentAddress('node2', 'agent0')])
    aggregator_agent = agent_composed_of(aggregator_role, ResultsRecorderRole(results_recorder))
    container1.register(aggregator_agent)

    container_mapping['node1'] = container1

    num_flex_agents = scenario_configuration.num_devices.value - 1
    agent_addresses = []
    flex_agent_roles = []

    for i in range(2, num_flex_agents + 2):
        container2 = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
        flex_agent_role = FlexAgentRole(aggregator_address=aggregator_agent.addr,
                                        scenario_config=scenario_configuration,
                                        can_provide_power=False if i == 2 else True)
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