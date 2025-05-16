import datetime
import json
import time
import asyncio
from dataclasses import dataclass
import logging
from enum import Enum
from pathlib import Path

import pandas as pd
import psutil

import os

logger = logging.getLogger(__name__)


class PayloadSizeConfig(Enum):
    SMALL = 8
    MEDIUM = 100
    LARGE = 200


@dataclass
class ScenarioConfiguration:
    scenario_id: str
    payload_size: PayloadSizeConfig
    num_devices: int


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

        # memory monitoring - simplified
        self.process = psutil.Process(os.getpid())
        self.memory_measurements = []  # [(timestamp, memory_mb), ...]
        self._memory_monitoring_task = None
        self._stop_monitoring = False

    def start_scenario_recording(self):
        self.execution_start_time = time.time()
        self._stop_monitoring = False
        self.memory_measurements = []

        # Start memory monitoring every second
        self._memory_monitoring_task = asyncio.create_task(self._monitor_memory())
        logger.debug("Started memory monitoring")

    async def _monitor_memory(self):
        """Monitor memory usage every second."""
        while not self._stop_monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.memory_measurements.append((time.time(), memory_mb))
                await asyncio.sleep(1.0)  # Wait 1 second
            except Exception as e:
                logger.error(f"Error monitoring memory: {e}")
                break

    def stop_scenario_recording(self):
        self.execution_end_time = time.time()
        self.execution_runtime = self.execution_end_time - self.execution_start_time

        # Stop memory monitoring
        self._stop_monitoring = True
        if self._memory_monitoring_task:
            self._memory_monitoring_task.cancel()

        logger.debug(f'Stop scenario. Execution took {self.execution_runtime} seconds.')
        logger.debug(f'Collected {len(self.memory_measurements)} memory measurements.')

        self._create_summary_csv()

    def _create_summary_csv(self):
        """Create a summary CSV and statistics JSON file."""
        summary_file = self.output_dir / f"messages_{self.scenario_configuration.scenario_id}.csv"
        statistics_file = self.output_dir / f"statistics_{self.scenario_configuration.scenario_id}.json"

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
                        'size_B': send_event.payload_size_B,
                        'time_send_ms': send_event.time_send_ms,
                        'time_receive_ms': receive_event.time_receive_ms,
                        'delay_ms': delay_ms
                    })
                    break
        df = pd.DataFrame(message_delays)
        df.to_csv(summary_file, index=False)

        # Calculate simple memory statistics
        memory_stats = self._calculate_memory_statistics()

        statistics = {
            'scenario_id': self.scenario_configuration.scenario_id,
            'execution_start_time': str(datetime.datetime.fromtimestamp(self.execution_start_time)),
            'execution_end_time': str(datetime.datetime.fromtimestamp(self.execution_end_time)),
            'execution_run_time_s': self.execution_runtime,
            'memory_statistics': memory_stats
        }

        with open(statistics_file, 'w') as outfile:
            json.dump(statistics, outfile, indent=2)

        logger.info(f"Results saved to {summary_file} and {statistics_file}")

    def _calculate_memory_statistics(self):
        """Calculate simple memory statistics from measurements."""
        if not self.memory_measurements:
            return {}

        memory_values = [mem_mb for _, mem_mb in self.memory_measurements]

        return {
            'initial_memory_mb': memory_values[0] if memory_values else 0,
            'final_memory_mb': memory_values[-1] if memory_values else 0,
            'peak_memory_mb': max(memory_values) if memory_values else 0,
            'min_memory_mb': min(memory_values) if memory_values else 0,
            'avg_memory_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
            'memory_change_mb': (memory_values[-1] - memory_values[0]) if len(memory_values) > 1 else 0,
            'measurements_count': len(memory_values)
        }

    def record_message_send_event(self, event):
        self.send_message_events.append(event)

    def record_message_receive_event(self, event):
        self.receive_message_events.append(event)
