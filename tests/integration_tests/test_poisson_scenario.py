import asyncio
import time

import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import DetailedModelScheduler, IdealCommunicationScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    PoissonSenderRole
from integration_environment.scenario_configuration import ScenarioConfiguration, ModelType, \
    ScenarioDuration, NumDevices, TrafficConfig
from tests.integration_tests.utils import setup_logging

logger = setup_logging()

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())


async def run_with_ideal_communication():
    scenario_configuration = ScenarioConfiguration(model_type=ModelType.ideal,
                                                   scenario_duration=ScenarioDuration.one_min,
                                                   traffic_configuration=TrafficConfig.poisson_broadcast_1_mps,
                                                   num_devices=NumDevices.two)
    results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)

    clock = ExternalClock(start_time=0)

    container1 = create_external_coupling(addr='node1', codec=my_codec, clock=clock)
    container2 = create_external_coupling(addr='node2', codec=my_codec, clock=clock)

    communication_network_entity = IdealCommunicationScheduler(container_mapping={'node1': container1,
                                                                                  'node2': container2},
                                                               scenario_duration_ms=
                                                               scenario_configuration.scenario_duration.value)

    cbr_receiver_role = ReceiverRole()
    cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
    container1.register(cbr_receiver_role_agent)

    container2.current_start_time_of_step = time.time()
    cbr_sender_role_agent = agent_composed_of(
        PoissonSenderRole(receiver_addresses=[cbr_receiver_role_agent.addr],
                          scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    async with activate(container1, container2) as _:
        results_recorder.start_scenario_recording()
        await communication_network_entity.scenario_finished
    results_recorder.stop_scenario_recording()

    assert len(cbr_receiver_role.received_messages) > 0


@pytest.mark.asyncio
def test_scenarios():
    asyncio.run(run_with_ideal_communication())
