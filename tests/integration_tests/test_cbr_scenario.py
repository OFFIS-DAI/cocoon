import asyncio
import logging

import pytest
from mango import agent_composed_of, JSON, create_tcp_container, activate, ExternalClock
from mango.container.factory import create_external_coupling

from integration_environment.ideal_communication_scheduler import IdealCommunicationScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.roles import ConstantBitrateSenderRole, ConstantBitrateReceiverRole

logger = logging.getLogger(__name__)

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_scenario():
    clock = ExternalClock(start_time=0)
    logging.basicConfig(filename='scenario.log', level=logging.DEBUG)
    container1 = create_external_coupling(addr='container1', codec=my_codec, clock=clock)
    cbr_receiver_role = ConstantBitrateReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role)
    container1.register(cbr_receiver_role_agent)

    container2 = create_external_coupling(addr='container2', codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr]))
    container2.register(cbr_sender_role_agent)

    communication_network_entity = IdealCommunicationScheduler(container_mapping={'container1': container1,
                                                                                  'container2': container2})

    async with activate(container1, container2) as cl:
        # no more run call since everything now happens automatically within the roles
        await communication_network_entity.scenario_finished

    assert len(cbr_receiver_role.received_messages) > 0


@pytest.mark.asyncio
def test_scenario():
    asyncio.run(run_scenario())
