import time
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfiguration:
    scenario_id: str


class ResultsRecorder:
    def __init__(self, scenario_configuration: ScenarioConfiguration, output_dir: str = "results"):
        self.scenario_configuration = scenario_configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # message recording
        self.send_message_events = []
        self.receive_message_events = []

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

        self._create_summary_csv()

    def _create_summary_csv(self):
        """Create a summary CSV."""
        summary_file = self.output_dir / f"{self.scenario_configuration.scenario_id}.csv"

        # Match send and receive events by msg_id
        message_delays = []
        for send_event in self.send_message_events:
            for receive_event in self.receive_message_events:
                if send_event.msg_id == receive_event.msg_id:
                    delay_ms = receive_event.time_receive_ms - send_event.time_send_ms
                    message_delays.append({
                        'msg_id': send_event.msg_id,
                        'sender': send_event.sender.protocol_addr,
                        'receiver': send_event.receiver.protocol_addr,
                        'size_B': send_event.size_B,
                        'time_send_ms': send_event.time_send_ms,
                        'time_receive_ms': receive_event.time_receive_ms,
                        'delay_ms': delay_ms
                    })
                    break
        df = pd.DataFrame(message_delays)
        df.to_csv(summary_file)

    def record_message_send_event(self, event):
        self.send_message_events.append(event)

    def record_message_receive_event(self, event):
        self.receive_message_events.append(event)