import asyncio
import logging
from dataclasses import dataclass

from random import choice, expovariate
from string import ascii_uppercase
from mango import Role

from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder, ScenarioConfiguration
from integration_environment.scenario_configuration import TrafficConfig

logger = logging.getLogger(__name__)


@dataclass
class SendMessage:
    sender: str
    receiver: str
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
            self.frequency_s = 60*15  # every 15 minutes
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
