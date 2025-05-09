import asyncio

from mango.container.external_coupling import ExternalSchedulingContainer


class IdealCommunicationScheduler:
    def __init__(self, container_mapping: dict[str, ExternalSchedulingContainer]):
        self._loop = asyncio.get_running_loop()
        self._loop.create_task(self.run_scenario())
        self.container_mapping = container_mapping
        self._next_activities = []
        self._current_time = 0
        self.duration = 1000
        self.scenario_finished = asyncio.Future()

    async def run_scenario(self):
        message_buffer = []
        while True:
            for container_name, container in self.container_mapping.items():
                incoming_messages_for_container = [m.message for m in message_buffer
                                                   if m.receiver == container_name]
                message_buffer = [m for m in message_buffer
                                  if m.receiver != container_name]
                while container.inbox is None:
                    await asyncio.sleep(1)

                output = await container.step(incoming_messages=incoming_messages_for_container,
                                              simulation_time=self._current_time)
                message_buffer.extend(output.messages)

                if output.next_activity:
                    self._next_activities.append(output.next_activity)

            self._next_activities = [n_a for n_a in self._next_activities if n_a > self._current_time]

            if len(message_buffer) > 0:
                self._current_time = message_buffer[0].time
            elif len(self._next_activities) > 0:
                self._current_time = min(self._next_activities)
            else:
                # no more activities or messages in mango -> finalize scenario
                break
            # simulation has reached the defined duration -> finalize scenario
            if self._current_time >= self.duration:
                self.scenario_finished.set_result(True)
                break
