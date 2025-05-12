import logging
from dataclasses import dataclass

from mango import Role

from integration_environment.messages import TrafficMessage
from integration_environment.results_recorder import ResultsRecorder

logger = logging.getLogger(__name__)


@dataclass
class SendMessage:
    sender: str
    receiver: str
    size_B: int
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


class ConstantBitrateSenderRole(Role):
    def __init__(self, receiver_addresses: list, frequency_ms=1000):
        super().__init__()
        self.frequency_s = frequency_ms / 1000
        self.receiver_addresses = receiver_addresses

        self._message_counter = 0

    def setup(self):
        pass

    def on_start(self):
        pass

    def on_ready(self):
        self.context.schedule_periodic_task(self.send_message, self.frequency_s)

    async def send_message(self):
        logger.debug(f'Send message at time {self.context.current_timestamp}')
        time_send = round(self.context.current_timestamp * 1000)
        for receiver in self.receiver_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_{self._message_counter}'
            # initialize event for results recording
            event = SendMessage(sender=self.context.addr,
                                receiver=receiver,
                                msg_id=msg_id,
                                size_B=10,
                                time_send_ms=time_send)
            self.context.emit_event(event=event, event_source=self)
            # send message
            await self.context.send_message(
                TrafficMessage(msg_id),
                receiver_addr=receiver,
            )
            self._message_counter += 1

    async def on_stop(self):
        pass


class ConstantBitrateReceiverRole(Role):
    def __init__(self):
        super().__init__()
        self.received_messages = []

    def setup(self):
        self.context.subscribe_message(self, self.handle_cbr_message,
                                       lambda content, meta: isinstance(content, TrafficMessage))

    def handle_cbr_message(self, content: TrafficMessage, meta):
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
