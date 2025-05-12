import time
from dataclasses import dataclass
import logging

from integration_environment.messages import TrafficMessage

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfiguration:
    scenario_duration_ms: int


class ResultsRecorder:
    def __init__(self, scenario_configuration: ScenarioConfiguration):
        self.scenario_configuration = scenario_configuration

        # message recording
        self.messages = []

        # performance metrics
        self.execution_start_time = 0
        self.execution_end_time = 0
        self.execution_runtime = 0

    def start_scenario_recording(self):
        self.execution_start_time = time.time()

    def stop_scenario_recording(self):
        self.execution_end_time = time.time()
        self.execution_runtime = self.execution_end_time - self.execution_start_time
        logger.debug(f'Stop scenario. Execution took {self.execution_runtime} seconds.')

    def record_message(self, message: TrafficMessage):
        self.messages.append(message)
