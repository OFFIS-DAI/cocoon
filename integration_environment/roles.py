import asyncio
import logging
from dataclasses import dataclass

from random import choice, expovariate
from string import ascii_uppercase
from mango import Role, AgentAddress

from integration_environment.messages import TrafficMessage, PlanningDataMessage
from integration_environment.results_recorder import ResultsRecorder, ScenarioConfiguration
from integration_environment.scenario_configuration import TrafficConfig

logger = logging.getLogger(__name__)


@dataclass
class SendMessage:
    sender: str
    receiver: AgentAddress
    payload_size_B: int
    time_send_ms: int
    msg_id: str


@dataclass
class ReceiveMessage:
    time_receive_ms: int
    msg_id: str


class ResultsRecorderRole(Role):
    def __init__(self, results_recorder: ResultsRecorder):
        super().__init__()
        self.results_recorder = results_recorder

    def setup(self):
        self.context.subscribe_event(self, SendMessage, self.handle_event_send_message)
        self.context.subscribe_event(self, ReceiveMessage, self.handle_event_receive_message)

    def handle_event_send_message(self, event, source):
        self.results_recorder.record_message_send_event(event)

    def handle_event_receive_message(self, event, source):
        self.results_recorder.record_message_receive_event(event)


def generate_payload_with_byte_size(byte_size: int):
    return ''.join(choice(ascii_uppercase) for _ in range(byte_size))


class ConstantBitrateSenderRole(Role):
    def __init__(self, receiver_addresses: list, scenario_config: ScenarioConfiguration):
        super().__init__()
        if scenario_config.traffic_configuration == TrafficConfig.cbr_broadcast_1_mps:
            self.frequency_s = 1  # every second
        elif scenario_config.traffic_configuration == TrafficConfig.cbr_broadcast_1_mpm:
            self.frequency_s = 60  # every 60 seconds
        elif scenario_config.traffic_configuration == TrafficConfig.cbr_broadcast_4_mph:
            self.frequency_s = 60 * 15  # every 15 minutes
        else:
            self.frequency_s = 1  # default
        self.receiver_addresses = receiver_addresses
        self.scenario_configuration = scenario_config

        self._message_counter = 0
        self._periodic_task = None

    def setup(self):
        pass

    def on_start(self):
        pass

    def on_ready(self):
        self._periodic_task = self.context.schedule_periodic_task(self.send_message, self.frequency_s)

    async def send_message(self):
        if self.context.current_timestamp == 0:
            return  # skip the first iteration
        logger.debug(f'Send message at time {self.context.current_timestamp}')
        time_send = round(self.context.current_timestamp * 1000)
        for receiver in self.receiver_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_{self._message_counter}'

            # send message
            payload = generate_payload_with_byte_size(self.scenario_configuration.payload_size.value)
            await self.context.send_message(
                TrafficMessage(msg_id=msg_id,
                               payload=payload),
                receiver_addr=receiver,
            )
            # initialize event for results recording
            event = SendMessage(sender=self.context.addr,
                                receiver=receiver,
                                msg_id=msg_id,
                                payload_size_B=self.scenario_configuration.payload_size.value,
                                time_send_ms=time_send)
            self.context.emit_event(event=event, event_source=self)

            self._message_counter += 1

    async def on_stop(self):
        """Clean shutdown - cancel the periodic task."""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass


class ReceiverRole(Role):
    def __init__(self):
        super().__init__()
        self.received_messages = []

    def setup(self):
        self.context.subscribe_message(self, self.handle_traffic_message,
                                       lambda content, meta: isinstance(content, TrafficMessage))

    def handle_traffic_message(self, content: TrafficMessage, meta):
        logger.debug(f'Traffic Message received at time {self.context.current_timestamp}.')
        # initialize event for results recording
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)
        self.received_messages.append(content)

    def on_start(self):
        pass

    def on_ready(self):
        pass

    async def on_stop(self):
        pass


class FlexAgentRole(Role):
    def __init__(self, aggregator_address: AgentAddress, scenario_config: ScenarioConfiguration):
        super().__init__()

        self.aggregator_address = aggregator_address
        self.scenario_configuration = scenario_config

        self._message_counter = 0
        self._periodic_task = None

    def setup(self):
        pass

    def on_start(self):
        pass

    def on_ready(self):
        # send planning data every 15 minutes
        self._periodic_task = self.context.schedule_periodic_task(self.send_planning_data, 15 * 60)

    async def send_planning_data(self):
        if self.context.current_timestamp == 0:
            return  # skip the first iteration
        logger.debug(f'Send planning data at time {self.context.current_timestamp/60} minutes.')
        time_send = round(self.context.current_timestamp * 1000)
        msg_id = f'{self.context.addr.protocol_addr}_planning_data_{self._message_counter}'

        await self.context.send_message(
            PlanningDataMessage(msg_id=msg_id,
                                baseline=0,
                                min_p=0,
                                max_p=1),
            receiver_addr=self.aggregator_address,
        )
        # initialize event for results recording
        event = SendMessage(sender=self.context.addr,
                            receiver=self.aggregator_address,
                            msg_id=msg_id,
                            payload_size_B=self.scenario_configuration.payload_size.value,
                            time_send_ms=time_send)
        self.context.emit_event(event=event, event_source=self)

        self._message_counter += 1

    async def on_stop(self):
        """Clean shutdown - cancel the periodic task."""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass


class AggregatorAgentRole(Role):
    def __init__(self):
        super().__init__()
        self.received_messages = []

        self._fix_power_tasks = []
        self.x_minute_time_window = 5

    def setup(self):
        self.context.subscribe_message(self, self.handle_planning_data,
                                       lambda content, meta: isinstance(content, PlanningDataMessage))

    def on_ready(self):
        # send planning data every 15 minutes
        time_stamps = [(15*i - self.x_minute_time_window)*60 for i in range(4)]
        for t in time_stamps:
            self._fix_power_tasks.append(self.context.schedule_timestamp_task(timestamp=t,
                                                                              coroutine=self.send_fixed_power()))

    async def send_fixed_power(self):
        logger.debug(f'Aggregator sends fixed power at time {self.context.current_timestamp/60} minutes.')

    def handle_planning_data(self, content: PlanningDataMessage, meta):
        logger.debug(f'Planning Data received at time {self.context.current_timestamp/60} minutes.')
        # initialize event for results recording
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)
        self.received_messages.append(content)

    def on_start(self):
        pass

    async def on_stop(self):
        for t in self._fix_power_tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass


class PoissonSenderRole(Role):
    def __init__(self, receiver_addresses: list, scenario_config: ScenarioConfiguration):
        super().__init__()
        self.receiver_addresses = receiver_addresses
        self.scenario_configuration = scenario_config
        self._message_counter = 0
        self._running = False
        self._scheduled_tasks = []

        # Configure lambda rate based on traffic configuration
        self.lambda_rate = self._get_lambda_rate_from_config()

    def _get_lambda_rate_from_config(self) -> float:
        """Map traffic configuration to Poisson rate parameter."""
        # You can extend this mapping based on your TrafficConfig enum
        if self.scenario_configuration.traffic_configuration == TrafficConfig.poisson_broadcast_1_mps:
            return 1.0  # 1 message per second on average
        elif self.scenario_configuration.traffic_configuration == TrafficConfig.poisson_broadcast_1_mpm:
            return 1.0 / 60.0  # 1 message per minute on average
        elif self.scenario_configuration.traffic_configuration == TrafficConfig.poisson_broadcast_4_mph:
            return 4.0 / 3600.0  # 4 messages per hour on average
        else:
            return 1.0  # Default: 1 message per second

    def setup(self):
        pass

    def on_start(self):
        pass

    def on_ready(self):
        self._running = True
        self._schedule_all_poisson_events()

    def _schedule_all_poisson_events(self):
        """Schedule all Poisson-distributed message sending events."""
        current_time = self.context.current_timestamp

        # Schedule events for the entire simulation duration
        # You may want to set a reasonable upper bound based on your simulation length
        max_simulation_time = current_time + 3600  # Example: 1 hour from start

        while current_time < max_simulation_time and self._running:
            # Generate exponentially distributed inter-arrival time
            inter_arrival_time = expovariate(self.lambda_rate)
            current_time += inter_arrival_time

            # Schedule the message sending task
            task = self.context.schedule_timestamp_task(
                timestamp=current_time,
                coroutine=self._send_message_to_all_receivers()
            )
            self._scheduled_tasks.append(task)

    async def _send_message_to_all_receivers(self):
        """Send message to all receivers with event recording."""
        if not self._running:
            return

        time_send = round(self.context.current_timestamp * 1000)

        for receiver in self.receiver_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_{self._message_counter}'

            await self.context.send_message(
                TrafficMessage(msg_id=msg_id, payload=self.scenario_configuration.payload_size.value),
                receiver_addr=receiver,
            )

            event = SendMessage(
                sender=self.context.addr,
                receiver=receiver,
                msg_id=msg_id,
                payload_size_B=self.scenario_configuration.payload_size.value,
                time_send_ms=time_send
            )
            self.context.emit_event(event=event, event_source=self)
            self._message_counter += 1

        logger.debug(f'Sent Poisson message at time {self.context.current_timestamp}, '
                     f'lambda rate: {self.lambda_rate}, message count: {self._message_counter}')

    async def on_stop(self):
        """Clean shutdown."""
        self._running = False
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
