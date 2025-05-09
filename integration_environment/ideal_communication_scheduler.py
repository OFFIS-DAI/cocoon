import asyncio
import math
from abc import ABC, abstractmethod

from mango.container.external_coupling import ExternalSchedulingContainer, ExternalAgentMessage


class CommunicationScheduler(ABC):
    """
    Abstract class responsible for scheduling message dispatch with different communication modeling approaches.
    """
    def __init__(self,
                 container_mapping: dict[str, ExternalSchedulingContainer],
                 scenario_duration_ms=200 * 1000):
        self._loop = asyncio.get_running_loop()
        self._loop.create_task(self.run_scenario())
        self._container_mapping = container_mapping
        self._next_activities = []
        self._current_time = 0
        self._duration = scenario_duration_ms

        self._message_buffer = {}  # time: message

        # create Future in order to wait for scenario finalization
        self.scenario_finished = asyncio.Future()

    def get_incoming_messages_for_container(self, container_name):
        container_msgs = []
        for time, messages in self._message_buffer.items():
            if time <= self._current_time:
                for message in messages:
                    if message.receiver == container_name:
                        container_msgs.append(message.message)
                self._message_buffer[time] = [m for m in self._message_buffer[time] if m.message not in container_msgs]

        times_without_messages = [time for time, obj in self._message_buffer.items() if len(obj) == 0]
        for time in times_without_messages:
            del self._message_buffer[time]
        return container_msgs

    async def run_scenario(self):
        while True:
            container_messages_dict = {}
            next_activities_in_current_step = []
            for container_name, container in self._container_mapping.items():
                incoming_messages_for_container = self.get_incoming_messages_for_container(container_name)
                while container.inbox is None:
                    await asyncio.sleep(1)

                output = await container.step(incoming_messages=incoming_messages_for_container,
                                              simulation_time=self._current_time)
                container_messages_dict[container_name] = output.messages
                next_activities_in_current_step.append(output.next_activity)

            await self.process_message_output(container_messages_dict=container_messages_dict,
                                              next_activities=next_activities_in_current_step)

            if len(self._message_buffer) > 0:
                self._current_time = min(self._message_buffer.keys())
            elif len(self._next_activities) > 0:
                self._current_time = min(self._next_activities)
            else:
                # no more activities or messages in mango -> finalize scenario
                self.scenario_finished.set_result(True)
                break

            if self._current_time >= self._duration:
                # simulation has reached the defined duration -> finalize scenario
                self.scenario_finished.set_result(True)
                break

    @abstractmethod
    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        pass


class IdealCommunicationScheduler(CommunicationScheduler):
    def __init__(self, container_mapping: dict[str, ExternalSchedulingContainer]):
        super().__init__(container_mapping)

    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        for container_name, messages in container_messages_dict.items():
            for message in messages:
                message_time_in_ms = math.ceil(message.time * 1000)
                if message.time not in self._message_buffer:
                    self._message_buffer[message_time_in_ms] = []
                self._message_buffer[message_time_in_ms].append(message)
        self._next_activities.extend([na for na in next_activities if na is not None])
        self._next_activities = [na for na in self._next_activities if na >= self._current_time]
