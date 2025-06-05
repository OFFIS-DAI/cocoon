import asyncio
import logging
from typing import List, Dict, Optional

from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.external_coupling import ExternalSchedulingContainer
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler, CommunicationScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    PoissonSenderRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig, ModelType, \
    ScenarioDuration, NumDevices, TrafficConfig

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_scenario_configurations():
    scenario_configurations = []
    for payload_size in [p for p in PayloadSizeConfig]:
        for model_type in [m for m in ModelType]:
            for scenario_duration in [s for s in ScenarioDuration]:
                for n_devices in [d for d in NumDevices]:
                    for traffic_config in [t for t in TrafficConfig]:
                        scenario_configurations.append(ScenarioConfiguration(payload_size=payload_size,
                                                                             num_devices=n_devices,
                                                                             model_type=model_type,
                                                                             scenario_duration=scenario_duration,
                                                                             traffic_configuration=traffic_config))
    return scenario_configurations


def get_scheduler(scenario_configuration: ScenarioConfiguration,
                  container_mapping: Dict[str, ExternalSchedulingContainer]) -> Optional[CommunicationScheduler]:
    if scenario_configuration.model_type == ModelType.ideal:
        return IdealCommunicationScheduler(container_mapping=container_mapping,
                                           scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.channel:
        return ChannelModelScheduler(container_mapping=container_mapping,
                                     scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.static_graph:
        return StaticDelayGraphModelScheduler(container_mapping=container_mapping,
                                              scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.detailed:
        return DetailedModelScheduler(container_mapping=container_mapping,
                                      scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.meta_model:
        return MetaModelScheduler(container_mapping=container_mapping,
                                  scenario_duration_ms=scenario_configuration.scenario_duration.value)
    else:
        logging.warning(f'Unknown model type: {scenario_configuration.model_type}')
        return None


async def initialize_constant_bitrate_broadcast_agents(clock: ExternalClock,
                                                       results_recorder: ResultsRecorder,
                                                       scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []
    for n_agents in range(scenario_configuration.num_devices.value):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        cbr_receiver_role = ReceiverRole()
        cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
        container.register(cbr_receiver_role_agent)
        receiver_addresses.append(cbr_receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node{scenario_configuration.num_devices.value}',
                                          codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    container_mapping[f'node{scenario_configuration.num_devices.value}'] = container2

    return container_mapping


async def initialize_poisson_broadcast_agents(clock: ExternalClock,
                                              results_recorder: ResultsRecorder,
                                              scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []
    for n_agents in range(scenario_configuration.num_devices.value - 1):
        index = n_agents + 1
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        receiver_role = ReceiverRole()
        receiver_role_agent = agent_composed_of(receiver_role, ResultsRecorderRole(results_recorder))
        container.register(receiver_role_agent)
        receiver_addresses.append(receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node{scenario_configuration.num_devices.value}',
                                          codec=my_codec, clock=clock)
    poisson_sender_role_agent = agent_composed_of(
        PoissonSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(poisson_sender_role_agent)

    container_mapping[f'node{scenario_configuration.num_devices.value}'] = container2

    return container_mapping


async def run_scenario(container_mapping: Dict[str, ExternalSchedulingContainer],
                       results_recorder: ResultsRecorder,
                       scheduler: CommunicationScheduler):
    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await scheduler.scenario_finished
    results_recorder.stop_scenario_recording()


async def run_benchmark_suite():
    for scenario_configuration in get_scenario_configurations():
        results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)
        clock = ExternalClock(start_time=0)

        container_mapping = {}
        if scenario_configuration.traffic_configuration in [TrafficConfig.cbr_broadcast_1_mps,
                                                            TrafficConfig.cbr_broadcast_1_mpm,
                                                            TrafficConfig.cbr_broadcast_4_mph]:
            container_mapping = \
                await initialize_constant_bitrate_broadcast_agents(clock=clock,
                                                                   results_recorder=results_recorder,
                                                                   scenario_configuration=scenario_configuration)
        elif scenario_configuration.traffic_configuration in [TrafficConfig.poisson_broadcast_1_mps,
                                                              TrafficConfig.poisson_broadcast_1_mpm,
                                                              TrafficConfig.poisson_broadcast_4_mph]:
            container_mapping = \
                await initialize_poisson_broadcast_agents(clock=clock,
                                                          results_recorder=results_recorder,
                                                          scenario_configuration=scenario_configuration)

        scheduler = get_scheduler(scenario_configuration=scenario_configuration,
                                  container_mapping=container_mapping)

        if scheduler is not None:
            print(f'Running scenario with config: {scenario_configuration.scenario_id}')
            await run_scenario(container_mapping=container_mapping,
                               results_recorder=results_recorder,
                               scheduler=scheduler)


if __name__ == "__main__":
    asyncio.run(run_benchmark_suite())
