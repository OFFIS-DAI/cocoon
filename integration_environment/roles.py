import logging

from mango import Role

from integration_environment.messages import TrafficMessage

logger = logging.getLogger(__name__)


class ConstantBitrateSenderRole(Role):
    def __init__(self, receiver_addresses: list, frequency_ms=1000):
        super().__init__()
        self.frequency_ms = frequency_ms
        self.receiver_addresses = receiver_addresses

    def setup(self):
        pass

    def on_start(self):
        self.context.schedule_instant_task(self.send_message())

    def on_ready(self):
        pass

    async def send_message(self):
        logger.debug(f'Send message at time {self.context.current_timestamp}')
        for receiver in self.receiver_addresses:
            await self.context.send_message(
                TrafficMessage(),
                receiver_addr=receiver,
            )

    async def on_stop(self):
        pass


class ConstantBitrateReceiverRole(Role):
    def __init__(self):
        super().__init__()
        self.received_messages = []

    def setup(self):
        self.context.subscribe_message(self, self.handle_cbr_message,
                                       lambda content, meta: isinstance(content, TrafficMessage))

    def handle_cbr_message(self, content, meta):
        logger.debug(f'Traffic Message received at time {self.context.current_timestamp}.')
        self.received_messages.append(content)

    def on_start(self):
        pass

    def on_ready(self):
        pass

    async def on_stop(self):
        pass
