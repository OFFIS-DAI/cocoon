import asyncio
import time

import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ConstantBitrateReceiverRole, ResultsRecorderRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig
from tests.integration_tests.utils import setup_logging, visualize_channel_model_graph, visualize_static_graph

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_scenario_with_ideal_communication():
    scenario_configuration = ScenarioConfiguration(scenario_id='ideal', payload_size=PayloadSizeConfig.SMALL,
                                                   num_devices=2)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = ConstantBitrateReceiverRole()
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
    scenario_configuration = ScenarioConfiguration(scenario_id='channel', payload_size=PayloadSizeConfig.SMALL,
                                                   num_devices=2)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)
    topology = {
        'nodes': [
            {
                'node_id': 'node1',
                'position': [10, 100],
                'processing_delay_ms': 500
            },
            {
                'node_id': 'node2',
                'position': [100, 10],
                'processing_delay_ms': 400
            },
            {
                'node_id': 'router1',
                'position': [50, 50],
                'processing_delay_ms': 400
            }
        ],
        'links': [
            {
                "source": "node1",
                "target": "router1",
                "transmission_rate_bps": 1000000000,
                "propagation_speed_mps": 200000000
            },
            {
                "source": "router1",
                "target": "node2",
                "transmission_rate_bps": 1000000000,
                "propagation_speed_mps": 200000000
            }
        ]
    }
    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = ConstantBitrateReceiverRole()
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
                                                         topology_dict=topology)

    visualize_channel_model_graph(communication_network_entity.channel_model)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0


async def run_scenario_with_static_graph_model():
    scenario_configuration = ScenarioConfiguration(scenario_id='static_graph', payload_size=PayloadSizeConfig.SMALL,
                                                   num_devices=2)
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
    cbr_receiver_role = ConstantBitrateReceiverRole()
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


async def run_scenario_with_detailed_communication_simulation():
    scenario_configuration = ScenarioConfiguration(scenario_id='detailed', payload_size=PayloadSizeConfig.SMALL,
                                                   num_devices=2)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)

    communication_network_entity = DetailedModelScheduler(container_mapping={'node1': container1,
                                                                             'node2': container2},
                                                          inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                                          config_name='General',
                                                          omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')

    cbr_receiver_role = ConstantBitrateReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2.current_start_time_of_step = time.time()
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                                  scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0


async def run_scenario_with_meta_model():
    scenario_configuration = ScenarioConfiguration(scenario_id='meta-model', payload_size=PayloadSizeConfig.SMALL,
                                                   num_devices=2)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)

    communication_network_entity = MetaModelScheduler(container_mapping={'node1': container1,
                                                                         'node2': container2},
                                                      inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                                      config_name='General',
                                                      omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')

    cbr_receiver_role = ConstantBitrateReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2.current_start_time_of_step = time.time()
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                                  scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0

    communication_network_entity.meta_model.print_network_summary()


@pytest.mark.asyncio
def test_scenarios():
    asyncio.run(run_scenario_with_ideal_communication())
    asyncio.run(run_scenario_with_simple_channel_model())
    asyncio.run(run_scenario_with_static_graph_model())
    asyncio.run(run_scenario_with_detailed_communication_simulation())
    asyncio.run(run_scenario_with_meta_model())
