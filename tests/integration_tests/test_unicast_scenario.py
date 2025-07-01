import asyncio
import time

import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import DetailedModelScheduler, IdealCommunicationScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    PoissonSenderRole, UnicastSenderRole
from integration_environment.scenario_configuration import ScenarioConfiguration, ModelType, \
    ScenarioDuration, NumDevices, TrafficConfig, NetworkModelType
from tests.integration_tests.utils import setup_logging

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_with_ideal_communication():
    for conf in [TrafficConfig.unicast_1s_delay, TrafficConfig.unicast_5s_delay, TrafficConfig.unicast_10s_delay]:
        scenario_configuration = ScenarioConfiguration(model_type=ModelType.ideal,
                                                       scenario_duration=ScenarioDuration.one_min,
                                                       traffic_configuration=conf)
        results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

        clock = ExternalClock(start_time=0)

        container_mapping = {}
        receiver_addr = []

        for i in range(10):
            container = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
            cbr_receiver_role = ReceiverRole()
            cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
            container.register(cbr_receiver_role_agent)
            container_mapping[f'node{i}'] = container
            receiver_addr.append(cbr_receiver_role_agent.addr)

        container2 = create_external_coupling(addr='node10', codec=my_codec, clock=clock)
        container_mapping[f'node10'] = container2

        communication_network_entity = IdealCommunicationScheduler(container_mapping=container_mapping,
                                                                   scenario_duration_ms=
                                                                   scenario_configuration.scenario_duration.value)

        container2.current_start_time_of_step = time.time()
        cbr_sender_role_agent = agent_composed_of(
            UnicastSenderRole(receiver_addresses=receiver_addr,
                              scenario_config=scenario_configuration),
            ResultsRecorderRole(results_recorder))
        container2.register(cbr_sender_role_agent)

        async with activate([c for c in container_mapping.values()]) as _:
            results_recorder.start_scenario_recording()
            await communication_network_entity.scenario_finished
        results_recorder.stop_scenario_recording()


async def run_with_detailed_communication():
    for network in [n for n in NetworkModelType]:
        if network == NetworkModelType.none:
            continue
        for conf in [TrafficConfig.unicast_1s_delay]:
            scenario_configuration = ScenarioConfiguration(model_type=ModelType.detailed,
                                                           network_type=network,
                                                           traffic_configuration=conf)
            results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

            clock = ExternalClock(start_time=0)

            container_mapping = {}
            all_agent_addresses = []

            # First pass: Create all agents and collect their addresses
            agents = []
            for i in range(10):
                container = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
                # Create agent with only receiver role initially
                receiver_role = ReceiverRole()
                agent = agent_composed_of(receiver_role, ResultsRecorderRole(results_recorder))
                container.register(agent)
                container_mapping[f'node{i}'] = container
                all_agent_addresses.append(agent.addr)
                agents.append((container, agent))

            # Second pass: Add UnicastRole to each agent with addresses of all OTHER agents
            for i, (container, agent) in enumerate(agents):
                # Get receiver addresses (all agents except this one)
                receiver_addresses = [addr for j, addr in enumerate(all_agent_addresses) if j != i]

                # Add UnicastRole to the existing agent
                unicast_role = UnicastSenderRole(receiver_addresses=receiver_addresses,
                                                 scenario_config=scenario_configuration,
                                                 start_at_s=i*10+1)
                agent.add_role(unicast_role)

            communication_network_entity = DetailedModelScheduler(container_mapping=container_mapping,
                                                                  inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                                                  config_name=scenario_configuration.network_type.value,
                                                                  simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                                                  omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/',
                                                                  scenario_duration_ms=110*1000)

            # Set start time for all containers
            for container in container_mapping.values():
                container.current_start_time_of_step = time.time()

            async with activate([c for c in container_mapping.values()]) as _:
                results_recorder.start_scenario_recording()
                await communication_network_entity.scenario_finished
            results_recorder.stop_scenario_recording()

@pytest.mark.asyncio
def test_scenarios():
    #asyncio.run(run_with_ideal_communication())
    asyncio.run(run_with_detailed_communication())
