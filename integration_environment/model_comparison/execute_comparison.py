import asyncio
import time
from typing import List, Dict

import pandas as pd
import pytest
from mango import agent_composed_of, JSON, activate, ExternalClock
from mango.container.external_coupling import ExternalSchedulingContainer
from mango.container.factory import create_external_coupling

from integration_environment.communication_model_scheduler import IdealCommunicationScheduler, ChannelModelScheduler, \
    StaticDelayGraphModelScheduler, DetailedModelScheduler, MetaModelScheduler, CommunicationScheduler
from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder
from integration_environment.roles import ConstantBitrateSenderRole, ConstantBitrateReceiverRole, ResultsRecorderRole
from integration_environment.scenario_configuration import ScenarioConfiguration, PayloadSizeConfig
from tests.integration_tests.utils import setup_logging, visualize_channel_model_graph, visualize_static_graph

my_codec = JSON()
my_codec.add_serializer(*TrafficMessage.__serializer__())

scenario_configurations = [
    ScenarioConfiguration(scenario_id='ideal-2',
                          payload_size=PayloadSizeConfig.SMALL,
                          num_devices=2),
    ScenarioConfiguration(scenario_id='ideal-3',
                          payload_size=PayloadSizeConfig.SMALL,
                          num_devices=3)
]


async def initialize_constant_bitrate_broadcast_agents(clock: ExternalClock,
                                                       results_recorder: ResultsRecorder,
                                                       scenario_configuration: ScenarioConfiguration):
    container_mapping = {}
    receiver_addresses = []
    for n_agents in range(scenario_configuration.num_devices - 1):
        index = n_agents + 1
        container = create_external_coupling(addr=f'node{index}', codec=my_codec, clock=clock)
        cbr_receiver_role = ConstantBitrateReceiverRole()
        cbr_receiver_role_agent = agent_composed_of(cbr_receiver_role, ResultsRecorderRole(results_recorder))
        container.register(cbr_receiver_role_agent)
        receiver_addresses.append(cbr_receiver_role_agent.addr)
        container_mapping[f'node{index}'] = container

    container2 = create_external_coupling(addr=f'node{scenario_configuration.num_devices}',
                                          codec=my_codec, clock=clock)
    cbr_sender_role_agent = agent_composed_of(
        ConstantBitrateSenderRole(receiver_addresses=receiver_addresses, scenario_config=scenario_configuration),
        ResultsRecorderRole(results_recorder))
    container2.register(cbr_sender_role_agent)

    container_mapping[f'node{scenario_configuration.num_devices}'] = container2

    return container_mapping


async def run_scenario(container_mapping: Dict[str, ExternalSchedulingContainer],
                       results_recorder: ResultsRecorder,
                       scheduler: CommunicationScheduler):
    async with activate([c for c in container_mapping.values()]) as _:
        results_recorder.start_scenario_recording()
        await scheduler.scenario_finished
    results_recorder.stop_scenario_recording()


async def run_benchmark_suite():
    # TODO: generate set of scenario configurations (for 2, 5, 10, 50 agents)
    for scenario_configuration in scenario_configurations:
        results_recorder = ResultsRecorder(scenario_configuration=scenario_configuration)
        clock = ExternalClock(start_time=0)

        container_mapping = await initialize_constant_bitrate_broadcast_agents(clock=clock,
                                                                               results_recorder=results_recorder,
                                                                               scenario_configuration=scenario_configuration)

        scheduler = IdealCommunicationScheduler(container_mapping=container_mapping)

        await run_scenario(container_mapping=container_mapping,
                           results_recorder=results_recorder,
                           scheduler=scheduler)


if __name__ == "__main__":
    asyncio.run(run_benchmark_suite())
