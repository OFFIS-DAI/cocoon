import asyncio
import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.roles import ConstantBitrateSenderRole, ConstantBitrateReceiverRole
from tests.integration_tests.utils import setup_logging, visualize_channel_model_graph, visualize_static_graph

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_scenario_with_ideal_communication():
    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    cbr_receiver_role = ConstantBitrateReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role)
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr]))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = IdealCommunicationScheduler(container_mapping={'node1': container1,
                                                                                  'node2': container2})

    async with activate(container1, container2) as _:
        # no more run call since everything now happens automatically within the roles
        await communication_network_entity.scenario_finished

    assert len(cbr_receiver_role.received_messages) > 0


async def run_scenario_with_simple_channel_model():
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
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role)
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr]))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = ChannelModelScheduler(container_mapping={'node1': container1,
                                                                            'node2': container2},
                                                         topology_dict=topology)

    visualize_channel_model_graph(communication_network_entity.channel_model)

    async with activate(container1, container2) as _:
        # no more run call since everything now happens automatically within the roles
        await communication_network_entity.scenario_finished

    assert len(cbr_receiver_role.received_messages) > 0


async def run_scenario_with_static_graph_model():
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
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role)
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr]))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = StaticDelayGraphModelScheduler(container_mapping={'node1': container1,
                                                                                     'node2': container2},
                                                                  topology_dict=topology)

    visualize_static_graph(communication_network_entity.static_graph_model)

    async with activate(container1, container2) as _:
        # no more run call since everything now happens automatically within the roles
        await communication_network_entity.scenario_finished

    assert len(cbr_receiver_role.received_messages) > 0


@pytest.mark.asyncio
def test_scenarios():
    asyncio.run(run_scenario_with_ideal_communication())
    asyncio.run(run_scenario_with_simple_channel_model())
    asyncio.run(run_scenario_with_static_graph_model())
