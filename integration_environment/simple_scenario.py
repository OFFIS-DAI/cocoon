import asyncio
import logging

from mango import agent_composed_of, JSON, create_tcp_container, activate
from mango.container.factory import create_external_coupling

from integration_environment.messages import TrafficMessage
from integration_environment.roles import ConstantBitrateSenderRole, ConstantBitrateReceiverRole

EXTERNAL_COUPLING = True

logger = logging.getLogger(__name__)

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_scenario():
    logging.basicConfig(filename='scenario.log', level=logging.DEBUG)
    if EXTERNAL_COUPLING:
        container1 = create_external_coupling(codec=my_codec)
    else:
        container1 = create_tcp_container(('localhost', 5555), my_codec)
    cbr_receiver_role_agent = agent_composed_of(ConstantBitrateReceiverRole())
    container1.register(cbr_receiver_role_agent)

    if EXTERNAL_COUPLING:
        container2 = create_external_coupling(codec=my_codec)
    else:
        container2 = create_tcp_container(('localhost', 5556), my_codec)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr]))
    container2.register(cbr_sender_role_agent)

    async with activate(container1, container2) as cl:
        # no more run call since everything now happens automatically within the roles
        await asyncio.sleep(5)


asyncio.run(run_scenario())
