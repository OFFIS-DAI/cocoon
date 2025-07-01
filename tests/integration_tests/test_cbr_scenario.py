import asyncio
import json
import time

import pandas as pd
import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.model_comparison.execute_comparison import get_training_df, \
    initialize_constant_bitrate_broadcast_agents
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole
from integration_environment.scenario_configuration import *
from tests.integration_tests.utils import setup_logging, visualize_channel_model_graph, visualize_static_graph

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_scenario_with_ideal_communication():
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.ideal)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = ReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                                  scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = IdealCommunicationScheduler(container_mapping={'node1': container1,
                                                                                  'node2': container2})

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0


async def run_scenario_with_simple_channel_model():
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.channel)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    top_file = '../../integration_environment/model_comparison/network_definitions/channel_simbench_lte.json'
    with open(top_file, 'r') as file:
        data = json.load(file)
        top_dict = data['topology']
    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = ReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                                  scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = ChannelModelScheduler(container_mapping={'node1': container1,
                                                                            'node2': container2},
                                                         topology_dict=top_dict)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0


async def run_scenario_with_static_graph_model():
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.static_graph)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)
    topology = {
        'nodes': [
            {
                'node_id': 'node1'
            },
            {
                'node_id': 'node2'
            }
        ],
        'links': [
            {
                "node_a": "node1",
                "node_b": "node2",
                "end-to-end-delay_ms": 15
            }
        ]
    }
    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = ReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                                  scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = StaticDelayGraphModelScheduler(container_mapping={'node1': container1,
                                                                                     'node2': container2},
                                                                  topology_dict=topology)

    visualize_static_graph(communication_network_entity.static_graph_model)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0


@pytest.mark.asyncio
async def test_run_scenario_with_detailed_communication_simulation():
    for run in range(3):
        scenario_configuration = ScenarioConfiguration(payload_size=PayloadSizeConfig.small,
                                                       num_devices=NumDevices.two,
                                                       model_type=ModelType.detailed,
                                                       scenario_duration=ScenarioDuration.one_min,
                                                       traffic_configuration=TrafficConfig.cbr_broadcast_1_mps,
                                                       network_type=NetworkModelType.simbench_5g,
                                                       run=run)
        results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

        clock = ExternalClock(start_time=0)

        container_mapping = await initialize_constant_bitrate_broadcast_agents(
            scenario_configuration=scenario_configuration,
            results_recorder=results_recorder,
            clock=clock)

        communication_network_entity = DetailedModelScheduler(container_mapping=container_mapping,
                                                              inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                                              simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                                              config_name=scenario_configuration.network_type.value,
                                                              omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                                              scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                                              )

        async with activate([c for c in container_mapping.values()]) as _:
            results_recorder.start_scenario_recording()
            await communication_network_entity.scenario_finished
        results_recorder.stop_scenario_recording()


@pytest.mark.asyncio
async def test_run_scenario_with_meta_model():
    scenario_configuration = ScenarioConfiguration(payload_size=PayloadSizeConfig.small,
                                                   num_devices=NumDevices.two,
                                                   model_type=ModelType.meta_model,
                                                   scenario_duration=ScenarioDuration.one_min,
                                                   traffic_configuration=TrafficConfig.cbr_broadcast_1_mps,
                                                   network_type=NetworkModelType.simbench_5g,
                                                   cluster_distance_threshold=ClusterDistanceThreshold.half,
                                                   i_pupa=BatchSizeIPupa.ten)

    training_df = get_training_df(scenario_configuration)

    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container_mapping = await initialize_constant_bitrate_broadcast_agents(
        scenario_configuration=scenario_configuration,
        results_recorder=results_recorder,
        clock=clock)

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
