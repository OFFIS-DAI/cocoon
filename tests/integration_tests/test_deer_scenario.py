import asyncio
import json
import time

import pandas as pd
import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock, AgentAddress
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler
from integration_environment.messages import TrafficMessage, PlanningDataMessage
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    AggregatorAgentRole, FlexAgentRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig, ModelType, \
    ScenarioDuration, TrafficConfig, NumDevices
from tests.integration_tests.utils import setup_logging, visualize_channel_model_graph, visualize_static_graph

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())
my_codec.add_serializer(*PlanningDataMessage.__serializer__())


@pytest.mark.asyncio
async def test_run_deer_scenario_with_ideal_communication():
    """Analysis 1: Baseline performance with ideal communication"""
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.ideal,
                                                   traffic_configuration=TrafficConfig.deer_use_case,
                                                   payload_size=PayloadSizeConfig.none,
                                                   scenario_duration=ScenarioDuration.one_hour,
                                                   num_devices=NumDevices.two)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = AggregatorAgentRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        FlexAgentRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                      scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = IdealCommunicationScheduler(container_mapping={'node1': container1,
                                                                                  'node2': container2},
                                                               scenario_duration_ms=scenario_configuration.scenario_duration.value)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0
