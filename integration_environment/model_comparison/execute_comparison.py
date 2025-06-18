import asyncio
import logging
import random
import time
from typing import Dict, Optional

from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.external_coupling import ExternalSchedulingContainer
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler, CommunicationScheduler
from integration_environment.messages import TrafficMessage, deer_message_classes
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ReceiverRole, ResultsRecorderRole, \
    PoissonSenderRole, UnicastRole, FlexAgentRole, AggregatorAgentRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig, ModelType, \
    ScenarioDuration, NumDevices, TrafficConfig, NetworkModelType

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())
for deer_message_class in deer_message_classes:
    my_codec.add_serializer(*deer_message_class.__serializer__())

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
    for network in [NetworkModelType.simbench_lte450, NetworkModelType.simbench_ethernet,
                    NetworkModelType.simbench_lte, NetworkModelType.simbench_5g]:
        for payload_size in [PayloadSizeConfig.small, PayloadSizeConfig.medium, PayloadSizeConfig.large]:
            for model_type in [ModelType.channel, ModelType.ideal, ModelType.static_graph,
                               ModelType.detailed, ModelType.meta_model]:
                for scenario_duration in [ScenarioDuration.one_min, ScenarioDuration.five_min,
                                          ScenarioDuration.one_hour]:
                    for n_devices in [NumDevices.ten]:  # , NumDevices.two, NumDevices.fifty, NumDevices.hundred]:
                        for traffic_config in [#TrafficConfig.cbr_broadcast_1_mps, TrafficConfig.cbr_broadcast_1_mpm,
                                               #TrafficConfig.cbr_broadcast_4_mph,
                                               #TrafficConfig.poisson_broadcast_1_mps,
                                               #TrafficConfig.poisson_broadcast_1_mpm,
                                               #TrafficConfig.poisson_broadcast_4_mph,
                                               TrafficConfig.unicast_1s_delay, TrafficConfig.unicast_5s_delay,
                                               TrafficConfig.unicast_10s_delay, TrafficConfig.deer_use_case]:
                            scenario_configurations.append(ScenarioConfiguration(payload_size=payload_size,
                                                                                 num_devices=n_devices,
                                                                                 model_type=model_type,
                                                                                 scenario_duration=scenario_duration,
                                                                                 traffic_configuration=traffic_config,
                                                                                 network_type=network))
    return scenario_configurations


def get_scheduler(scenario_configuration: ScenarioConfiguration,
                  container_mapping: Dict[str, ExternalSchedulingContainer]) -> Optional[CommunicationScheduler]:
    if scenario_configuration.model_type == ModelType.ideal:
        return IdealCommunicationScheduler(container_mapping=container_mapping,
                                           scenario_duration_ms=scenario_configuration.scenario_duration.value)
    elif scenario_configuration.model_type == ModelType.channel:
        return ChannelModelScheduler(container_mapping=container_mapping,
                                     scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                     topology_file_name=f'network_definitions/channel_'
                                                        f'{scenario_configuration.network_type.name}.json')
    elif scenario_configuration.model_type == ModelType.static_graph:
        return StaticDelayGraphModelScheduler(container_mapping=container_mapping,
                                              scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                              topology_file_name=f'network_definitions/static_delay_graph_'
                                                                 f'{scenario_configuration.network_type.name}.json')
    elif scenario_configuration.model_type == ModelType.detailed:
        return DetailedModelScheduler(container_mapping=container_mapping,
                                      scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                      config_name=scenario_configuration.network_type.value,
                                      inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                      simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                      omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')
    elif scenario_configuration.model_type == ModelType.meta_model:
        return MetaModelScheduler(container_mapping=container_mapping,
                                  scenario_duration_ms=scenario_configuration.scenario_duration.value,
                                  config_name=scenario_configuration.network_type.value,
                                  inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                  simu5G_installation_path='/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src',
                                  omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/'
                                  )
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


async def initialize_unicast_communication_agents(clock: ExternalClock,
                                                  results_recorder: ResultsRecorder,
                                                  scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []

    agents = []
    for n_agents in range(scenario_configuration.num_devices.value):
        index = n_agents
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        receiver_role = ReceiverRole()
        agent = agent_composed_of(receiver_role, ResultsRecorderRole(results_recorder))
        container.register(agent)
        receiver_addresses.append(agent.addr)
        container_mapping[f'node{index}'] = container
        agents.append((container, agent))

    # Second pass: Add UnicastRole to each agent with addresses of all OTHER agents
    for i, (container, agent) in enumerate(agents):
        # Get receiver addresses (all agents except this one)
        receiver_addresses = [addr for j, addr in enumerate(receiver_addresses) if j != i]

        # Add UnicastRole to the existing agent
        unicast_role = UnicastRole(receiver_addresses=receiver_addresses,
                                   scenario_config=scenario_configuration,
                                   start_at_s=i * 10 + 1)
        agent.add_role(unicast_role)

    return container_mapping


async def initialize_deer_use_case_agents(clock: ExternalClock,
                                          results_recorder: ResultsRecorder,
                                          scenario_configuration: ScenarioConfiguration):
    container_mapping = {}

    container1 = create_external_coupling(addr='node0', codec=my_codec, clock=clock)
    aggregator_role = AggregatorAgentRole(flex_agent_addresses=None, x_minute_time_window=0.05)
    aggregator_agent = agent_composed_of(aggregator_role, ResultsRecorderRole(results_recorder))
    container1.register(aggregator_agent)

    container_mapping['node0'] = container1

    num_flex_agents = scenario_configuration.num_devices.value - 1
    agent_addresses = []
    flex_agent_roles = []

    baseline_values = [100] * num_flex_agents
    flex_values = [random.randint(0, 100) for _ in range(num_flex_agents)]

    for i in range(1, num_flex_agents + 1):
        container2 = create_external_coupling(addr=f'node{i}', codec=my_codec, clock=clock)
        flex_agent_role = FlexAgentRole(aggregator_address=aggregator_agent.addr,
                                        scenario_config=scenario_configuration,
                                        can_provide_power=False if i == 2 else True,
                                        baseline_value=baseline_values[i - 2],
                                        flexibility_value=flex_values[i - 2])
        flex_agent = agent_composed_of(flex_agent_role, ResultsRecorderRole(results_recorder))

        container2.current_start_time_of_step = time.time()
        container2.register(flex_agent)

        agent_addresses.append(flex_agent.addr)
        flex_agent_roles.append(flex_agent_role)

        container_mapping[f'node{i}'] = container2

    aggregator_role.flex_agent_addresses = agent_addresses

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
        elif scenario_configuration.traffic_configuration in [TrafficConfig.unicast_1s_delay,
                                                              TrafficConfig.unicast_5s_delay,
                                                              TrafficConfig.unicast_10s_delay]:
            container_mapping = \
                await initialize_unicast_communication_agents(clock=clock,
                                                              results_recorder=results_recorder,
                                                              scenario_configuration=scenario_configuration)
        elif scenario_configuration.traffic_configuration == TrafficConfig.deer_use_case:
            container_mapping = \
                await initialize_deer_use_case_agents(clock=clock,
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
