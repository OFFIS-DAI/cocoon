import datetime
import json
import time
import asyncio
import logging
from pathlib import Path

import pandas as pd
import psutil

import os

from integration_environment.communication_model_scheduler import CommunicationScheduler
from integration_environment.scenario_configuration import ScenarioConfiguration

logger = logging.getLogger(__name__)


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

        self.substitution_event = None  # Will store substitution information

        self.scheduler = None

    def set_scheduler(self, scheduler: CommunicationScheduler):
        self.scheduler = scheduler

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

        self.record_meta_model_substitution()
        self.record_time_advancement()
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

        if self.substitution_event:
            statistics['meta_model_substitution'] = self.substitution_event
        else:
            statistics['meta_model_substitution'] = {'substitution_occurred': False}

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

    def record_timeout(self, timeout_seconds: int, error_message: str = None):
        """
        Record a timeout event for the current scenario.

        Args:
            timeout_seconds: The timeout duration that was exceeded
            error_message: Optional additional error message
        """
        self.execution_end_time = time.time()
        self.execution_runtime = self.execution_end_time - self.execution_start_time

        # Stop memory monitoring
        self._stop_monitoring = True
        if self._memory_monitoring_task:
            self._memory_monitoring_task.cancel()

        logger.warning(f'Scenario {self.scenario_configuration.scenario_id} timed out after {timeout_seconds}s '
                       f'(actual runtime: {self.execution_runtime:.2f}s)')

        # Create timeout-specific files
        self._create_timeout_summary(timeout_seconds, error_message)

    def _create_timeout_summary(self, timeout_seconds: int, error_message: str = None):
        """Create summary files for a timed-out scenario."""
        summary_file = self.output_dir / f"messages_{self.scenario_configuration.scenario_id}.csv"
        statistics_file = self.output_dir / f"statistics_{self.scenario_configuration.scenario_id}.json"

        # Create CSV with whatever messages we have so far
        message_delays = []
        for send_event in self.send_message_events:
            # Try to find matching receive event
            receive_event = None
            for recv_event in self.receive_message_events:
                if send_event.msg_id == recv_event.msg_id:
                    receive_event = recv_event
                    break

            if receive_event:
                # Complete message with delay
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
            else:
                # Incomplete message (sent but not received)
                message_delays.append({
                    'msg_id': send_event.msg_id,
                    'sender': send_event.sender.protocol_addr,
                    'receiver': send_event.receiver.protocol_addr,
                    'size_B': send_event.payload_size_B,
                    'time_send_ms': send_event.time_send_ms,
                    'time_receive_ms': None,  # Not received
                    'delay_ms': None  # Cannot calculate delay
                })

        df = pd.DataFrame(message_delays)
        df.to_csv(summary_file, index=False)

        # Calculate memory statistics
        memory_stats = self._calculate_memory_statistics()

        # Create statistics with timeout information
        statistics = {
            'scenario_id': self.scenario_configuration.scenario_id,
            'execution_start_time': str(datetime.datetime.fromtimestamp(self.execution_start_time)),
            'execution_end_time': str(datetime.datetime.fromtimestamp(self.execution_end_time)),
            'execution_run_time_s': self.execution_runtime,
            'timeout_occurred': True,
            'timeout_limit_s': timeout_seconds,
            'timeout_error_message': error_message,
            'messages_sent': len(self.send_message_events),
            'messages_received': len(self.receive_message_events),
            'messages_completed': len([msg for msg in message_delays if msg['delay_ms'] is not None]),
            'memory_statistics': memory_stats
        }

        with open(statistics_file, 'w') as outfile:
            json.dump(statistics, outfile, indent=2)

        logger.info(f"Timeout results saved to {summary_file} and {statistics_file}")
        logger.info(f"Partial results: {len(self.send_message_events)} sent, "
                    f"{len(self.receive_message_events)} received")

    def record_time_advancement(self):
        """
        Record time advancement data from the scheduler.
        Saves both to statistics JSON and as a separate CSV file.
        """
        if hasattr(self.scheduler, 'time_advancement') and self.scheduler.time_advancement:
            time_advancement_data = self.scheduler.time_advancement

            # Save as separate CSV file for detailed analysis
            time_advancement_file = self.output_dir / f"time_advancement_{self.scenario_configuration.scenario_id}.csv"

            # Convert to DataFrame for easy CSV export
            time_advancement_list = []
            for real_time, sim_time in time_advancement_data.items():
                time_advancement_list.append({
                    'real_timestamp': real_time,
                    'real_datetime': str(datetime.datetime.fromtimestamp(real_time)),
                    'simulation_time_s': sim_time,
                    'relative_real_time_s': real_time - self.execution_start_time if self.execution_start_time else 0
                })

            df_time_advancement = pd.DataFrame(time_advancement_list)
            df_time_advancement = df_time_advancement.sort_values('real_timestamp')
            df_time_advancement.to_csv(time_advancement_file, index=False)

            # Also add summary statistics to the main statistics file
            time_advancement_stats = {
                'total_data_points': len(time_advancement_data),
                'simulation_start_time_s': min(time_advancement_data.values()) if time_advancement_data else 0,
                'simulation_end_time_s': max(time_advancement_data.values()) if time_advancement_data else 0,
                'total_simulation_duration_s': max(time_advancement_data.values()) - min(
                    time_advancement_data.values()) if time_advancement_data else 0,
                'real_start_timestamp': min(time_advancement_data.keys()) if time_advancement_data else 0,
                'real_end_timestamp': max(time_advancement_data.keys()) if time_advancement_data else 0,
                'total_real_duration_s': max(time_advancement_data.keys()) - min(
                    time_advancement_data.keys()) if time_advancement_data else 0
            }

            # Store for inclusion in statistics JSON
            self.time_advancement_stats = time_advancement_stats

            logger.info(f"Time advancement data saved to {time_advancement_file}")
            logger.info(f"Time advancement summary: {len(time_advancement_data)} data points, "
                        f"simulation time from {time_advancement_stats['simulation_start_time_s']:.2f}s "
                        f"to {time_advancement_stats['simulation_end_time_s']:.2f}s")
        else:
            logger.debug("No time advancement data available from scheduler")
            self.time_advancement_stats = None

    def record_meta_model_substitution(self):
        """
        Record when the meta-model substitutes the detailed simulation.
        """
        if hasattr(self.scheduler, 'meta_model') and self.scheduler.meta_model:
            substitution_info = self.scheduler.meta_model.substitution_info
            if substitution_info.occurred:
                substitution_time_s = substitution_info.current_time_s
                message_index = substitution_info.message_index
                additional_info = substitution_info.additional_metrics
                confidence_score = substitution_info.confidence_score

                substitution_data = {
                    'substitution_occurred': True,
                    'substitution_time_s': substitution_time_s,
                    'substitution_message_index': message_index,
                    'confidence_score': confidence_score
                }

                # Add any additional information provided
                if additional_info:
                    substitution_data.update(additional_info)

                self.substitution_event = substitution_data

                logger.info(f"Meta-model substitution recorded at simulation time {substitution_time_s}s, "
                            f"message index {message_index}")
