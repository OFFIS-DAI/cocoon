import asyncio
import logging
import math
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional

import pandas as pd
from mango.container.external_coupling import ExternalSchedulingContainer, ExternalAgentMessage

from integration_environment.network_models.channel_network_model import ChannelNetworkModel
from integration_environment.network_models.cocoon_meta_model import CocoonMetaModel
from integration_environment.network_models.detailed_network_model import DetailedNetworkModel

logger = logging.getLogger(__name__)


class CommunicationScheduler(ABC):
    """
    Abstract class responsible for scheduling message dispatch with different communication modeling approaches.
    """

    def __init__(self,
                 container_mapping: dict[str, ExternalSchedulingContainer],
                 scenario_duration_ms=200 * 1000):
        """
        Initialize a new communication scheduler.

        :param container_mapping: Dictionary mapping container names to their ExternalSchedulingContainer objects.
        :param scenario_duration_ms: Total duration of the scenario in milliseconds. Defaults to 200 seconds.
        """
        self._loop = asyncio.get_running_loop()
        self._loop.create_task(self.run_scenario())
        self._container_mapping = container_mapping
        self._next_activities = []
        self.current_time = 0
        self._duration_s = scenario_duration_ms / 1000

        self._message_buffer = {}  # time (in sec): message

        # create Future in order to wait for scenario finalization
        self.scenario_finished = asyncio.Future()

    def get_incoming_messages_for_container(self, container_name) -> list:
        """
        Retrieve all pending messages intended for a specific container.

        This method checks the message buffer for any messages that should be delivered
        to the specified container at the current simulation time or earlier.

        :param container_name: The name of the container to get messages for.
        :return: A list of message objects intended for the specified container.
        """
        container_msgs = []
        for time, messages in self._message_buffer.items():
            if time <= self.current_time:
                for message in messages:
                    if message.receiver == container_name:
                        container_msgs.append(message.message)
                self._message_buffer[time] = [m for m in self._message_buffer[time] if m.message not in container_msgs]

        times_without_messages = [time for time, obj in self._message_buffer.items() if len(obj) == 0]
        for time in times_without_messages:
            del self._message_buffer[time]
        return container_msgs

    async def run_scenario(self):
        """
        Run the simulation scenario until completion.

        This method implements the main simulation loop, advancing time and processing
        messages between containers. The loop continues until either there are no more
        scheduled activities or messages, or the scenario duration is reached.

        Sets the scenario_finished future when complete.
        """
        await self._on_scenario_start()

        for container_name, container in self._container_mapping.items():
            while container.inbox is None:
                await asyncio.sleep(1)

        while True:
            container_messages_dict = {}
            next_activities_in_current_step = []
            for container_name, container in self._container_mapping.items():
                incoming_messages_for_container = self.get_incoming_messages_for_container(container_name)

                output = await container.step(incoming_messages=incoming_messages_for_container,
                                              simulation_time=self.current_time)
                container_messages_dict[container_name] = output.messages
                next_activities_in_current_step.append(output.next_activity)

            await self.process_message_output(container_messages_dict=container_messages_dict,
                                              next_activities=next_activities_in_current_step)

            if self._waiting_for_messages():
                await self.handle_waiting()

            if len(self._message_buffer) > 0:
                self.current_time = min(self._message_buffer.keys())
            elif len(self._next_activities) > 0:
                self.current_time = min(self._next_activities)
            elif not self._waiting_for_messages():
                # no more activities or messages in mango or external simulation -> finalize scenario
                await self._on_scenario_finished()
                self.scenario_finished.set_result(True)
                break

            if self.current_time >= self._duration_s:
                # simulation has reached the defined duration -> finalize scenario
                await self._on_scenario_finished()
                self.scenario_finished.set_result(True)
                break

    async def _on_scenario_start(self):
        pass

    def _waiting_for_messages(self):
        return False

    async def handle_waiting(self):
        pass

    async def _on_scenario_finished(self):
        """
        Called when scenario is finished. Override in subclasses for cleanup tasks.
        """
        pass

    @abstractmethod
    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        """
        Process message outputs from containers and schedule their delivery.

        This abstract method must be implemented by concrete subclasses to define
        how messages are processed and scheduled according to the specific
        communication model being simulated.

        :param container_messages_dict: Dictionary mapping container names to their outgoing messages.
        :param next_activities: List of timestamps for the next scheduled activities of containers.
        """
        pass


class IdealCommunicationScheduler(CommunicationScheduler):
    """
    Implementation of a communication scheduler with ideal (instant) message delivery.

    This class models perfect communication with no delays or packet losses.
    Messages are delivered exactly at their specified dispatch time without any
    additional communication overhead or constraints.

    This scheduler is useful for establishing a baseline for comparison with more
    realistic communication models, or for simulations where communication effects
    are not a concern.
    """

    def __init__(self, container_mapping: dict[str, ExternalSchedulingContainer]):
        """
        Initialize an ideal communication scheduler.
        :param container_mapping: Dictionary mapping container names to their ExternalSchedulingContainer objects.
        """
        super().__init__(container_mapping)

    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        """
        Process message outputs with ideal (instant) delivery scheduling.

        Messages are scheduled for delivery at exactly their specified dispatch time
        without any additional delays or modifications.

        :param container_messages_dict: Dictionary mapping container names to their outgoing messages.
        :param next_activities: List of timestamps for the next scheduled activities of containers.
        """
        for container_name, messages in container_messages_dict.items():
            for message in messages:
                if message.time not in self._message_buffer:
                    self._message_buffer[message.time] = []
                self._message_buffer[message.time].append(message)
        self._next_activities.extend([na for na in next_activities if na is not None])
        self._next_activities = [na for na in self._next_activities if na >= self.current_time]


class ChannelModelScheduler(CommunicationScheduler):
    """
    Implementation of a communication scheduler with message end-to-end delays based on
    network parameter configurations.

    A channel model forms the basis for this scheduler and can be found in network_models/channel_network_model.py.
    """

    def __init__(self,
                 container_mapping: dict[str, ExternalSchedulingContainer],
                 topology_dict: dict = None,
                 topology_file_name: str = None):
        super().__init__(container_mapping)
        if topology_dict:
            self.channel_model = ChannelNetworkModel.from_dict(topology_data=topology_dict)
        elif topology_file_name:
            self.channel_model = ChannelNetworkModel.from_json_file(file_path=topology_file_name)
        else:
            raise ValueError('Topology information must be provided in order to a initialize SimpleChannelModel. ')

    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        for container_name, messages in container_messages_dict.items():
            for message in messages:
                delay_ms = self.channel_model.calculate_end_to_end_delay(sender_id=container_name,
                                                                         receiver_id=message.receiver,
                                                                         message_size_bits=len(message.message))
                message_departure_time_in_ms = math.ceil(message.time * 1000) + delay_ms
                message_departure_time_in_s = message_departure_time_in_ms / 1000
                if message_departure_time_in_s not in self._message_buffer:
                    self._message_buffer[message_departure_time_in_s] = []
                self._message_buffer[message_departure_time_in_s].append(message)
        self._next_activities.extend([na for na in next_activities if na is not None])
        self._next_activities = [na for na in self._next_activities if na >= self.current_time]


class DetailedModelScheduler(CommunicationScheduler):
    """
    Implementation of a communication scheduler which coordinates message simulation over the
    communication network simulator OMNeT++. For this, a DetailedNetworkModel with an OMNeTConnection is used.
    """

    def __init__(self,
                 container_mapping: dict[str, ExternalSchedulingContainer],
                 inet_installation_path: str,
                 config_name: str,
                 omnet_project_path: str):
        super().__init__(container_mapping)
        self.detailed_network_model = DetailedNetworkModel(inet_installation_path=inet_installation_path,
                                                           config_name=config_name,
                                                           omnet_project_path=omnet_project_path)

    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        self._next_activities.extend([na for na in next_activities if na is not None])
        self._next_activities = [na for na in self._next_activities if na > self.current_time]

        max_advance = self._get_max_advance_in_ms(self._next_activities)
        if sum([len(values) for values in container_messages_dict.values()]) > 0:
            await self.detailed_network_model.simulate_message_dispatch(sender_message_dict=container_messages_dict,
                                                                        max_advance_ms=max_advance)

        if self.detailed_network_model.waiting_for_messages_from_omnet():
            message_buffer = await self.detailed_network_model.get_received_messages_from_omnet_connection()
            for time_s, messages in message_buffer.items():
                if time_s not in self._message_buffer:
                    self._message_buffer[time_s] = []
                self._message_buffer[time_s].extend(messages)

    def _get_max_advance_in_ms(self, next_activities=None):
        """
        Gets max advance value in ms.
        @param: next_activities: next activities from containers in seconds.
        """
        if not next_activities:
            next_activities = self._next_activities
        next_activities = [na for na in next_activities if na]  # get next activity values
        max_advance = min(next_activities) * 1000 if len(next_activities) > 0 else self._duration_s * 1000
        return max_advance

    def _waiting_for_messages(self):
        return self.detailed_network_model.waiting_for_messages_from_omnet()

    async def handle_waiting(self):
        if not self.detailed_network_model.omnet_connection or not self.detailed_network_model.omnet_connection.running:
            return
        message_buffer = await self.detailed_network_model.handle_waiting_with_omnet(
            max_advance_ms=self._get_max_advance_in_ms())
        for time_s, messages in message_buffer.items():
            if time_s not in self._message_buffer:
                self._message_buffer[time_s] = []
            self._message_buffer[time_s].extend(messages)


class MetaModelScheduler(DetailedModelScheduler):
    """
    Implementation of a communication scheduler which incorporates the meta-model cocoon.
    """

    def __init__(self, container_mapping: dict[str, ExternalSchedulingContainer], inet_installation_path: str,
                 config_name: str, omnet_project_path: str, output_file_name: str = 'cocoon_output.csv',
                 in_training_mode: bool = True, training_df: Optional[pd.DataFrame] = None,
                 cluster_distance_threshold: float = 5):
        super().__init__(container_mapping, inet_installation_path, config_name, omnet_project_path)
        self.training_df = training_df
        self.meta_model = CocoonMetaModel(output_file_name=output_file_name,
                                          mode=CocoonMetaModel.Mode.TRAINING
                                          if in_training_mode else CocoonMetaModel.Mode.PRODUCTION,
                                          cluster_distance_threshold=cluster_distance_threshold)
        self.meta_model_only = False
        self.msg_id_to_msg = None
        self.meta_model_msg_counter = 0

    async def _on_scenario_start(self):
        if self.meta_model.mode == CocoonMetaModel.Mode.PRODUCTION:
            if not isinstance(self.training_df, pd.DataFrame):
                logger.warning('No training data provided for meta-model in production phase.')
            else:
                self.meta_model.execute_egg_phase(self.training_df)

    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        if not self.meta_model_only:
            await super().process_message_output(container_messages_dict=container_messages_dict,
                                                 next_activities=next_activities)

        for container_name, messages in container_messages_dict.items():
            for message in messages:
                if not self.meta_model_only:
                    msg_id = self.detailed_network_model.get_message_id_for_message(message)
                else:
                    msg_id = f'msg_{self.meta_model_msg_counter}'
                    self.msg_id_to_msg[self.meta_model_msg_counter] = message
                    self.meta_model_msg_counter += 1
                if msg_id:
                    self.meta_model.process_sent_message(sender=container_name,
                                                         receiver=message.receiver,
                                                         payload_size_B=len(message.message),
                                                         current_time_ms=int(self.current_time * 1000),
                                                         msg_id=msg_id)
                else:
                    logger.warning('ID of message cannot be resolved.')

        stand_alone_meta_model_before = copy(self.meta_model_only)
        await self.meta_model.process_observations()
        self.meta_model_only = self.meta_model.substitution_threshold_reached

        if not stand_alone_meta_model_before and self.meta_model_only:
            logger.info('Now switching to meta-model only mode.')
            self.detailed_network_model.cleanup()
            self.meta_model_msg_counter = max(self.detailed_network_model.msg_id_to_msg.keys()) + 1
            self.msg_id_to_msg = self.detailed_network_model.msg_id_to_msg
            logger.debug(f'Message counter starts at {self.meta_model_msg_counter}')
            await asyncio.sleep(0.1)  # wait to terminate OMNeT++

        if self.meta_model_only:
            messages_in_transit = self.meta_model.get_messages_in_transit()
            for message_in_transit in messages_in_transit:
                if ('msg_id' not in message_in_transit
                        or 'time_send_ms' not in message_in_transit
                        or 'delay_ms' not in message_in_transit):
                    logger.warning('Missing keys in message.')
                    continue
                msg_id_num = message_in_transit['msg_id'].split('_')
                if len(msg_id_num) == 2:
                    msg_id_num = int(msg_id_num[1])
                else:
                    logger.warning('ID of message cannot be resolved.')
                    continue
                time_receive = (message_in_transit['time_send_ms'] + message_in_transit['delay_ms'])/1000
                if time_receive not in self._message_buffer:
                    self._message_buffer[time_receive] = []
                self._message_buffer[time_receive].append(self.msg_id_to_msg[msg_id_num])
            # Update next activities
            self._next_activities.extend([na for na in next_activities if na is not None])
            self._next_activities = [na for na in self._next_activities if na >= self.current_time]

        for time_s, messages in self._message_buffer.items():
            for message in messages:
                if not self.meta_model_only:
                    msg_id = self.detailed_network_model.get_message_id_for_message(message)
                else:
                    msg_id = self.get_message_id_for_message(message)
                if msg_id:
                    self.meta_model.process_received_message(current_time_ms=int(time_s * 1000),
                                                             msg_id=msg_id)
                else:
                    logger.warning('ID of message cannot be resolved.')

    def get_message_id_for_message(self, message: ExternalAgentMessage):
        fits = [(key, value) for key, value in self.msg_id_to_msg.items() if value == message]
        if len(fits) == 1:
            return f'msg_{fits[0][0]}'
        return None

    async def handle_waiting(self):
        if self.meta_model_only:
            return
        await super().handle_waiting()
        for time_s, messages in self._message_buffer.items():
            for message in messages:
                msg_id = self.detailed_network_model.get_message_id_for_message(message)
                if msg_id:
                    self.meta_model.process_received_message(current_time_ms=int(time_s * 1000),
                                                             msg_id=msg_id)
                else:
                    logger.warning('ID of message cannot be resolved.')

    async def _on_scenario_finished(self):
        """
        Save meta-model observations when scenario finishes.
        """
        # Final call to process observations
        await self.meta_model.process_observations()
        await self.meta_model.save_observations()


class StaticDelayGraphModelScheduler(CommunicationScheduler):
    """
    Implementation of a communication scheduler with message end-to-end delays based on a static network configuration.
    For this, the end-to-end delays between two nodes have to be pre-configured by the user and are then used
    during the scenario in order to delay message dispatch.

    A static graph model forms the basis for this scheduler and can be found in network_models/static_graph_model.py.
    """

    def __init__(self,
                 container_mapping: dict[str, ExternalSchedulingContainer],
                 topology_dict: dict = None,
                 topology_file_name: str = None):
        """
        Initialize a static delay graph model scheduler.

        :param container_mapping: Dictionary mapping container names to their ExternalSchedulingContainer objects.
        :param topology_dict: Dictionary containing network topology information.
        :param topology_file_name: Path to JSON file containing topology information.
        """
        super().__init__(container_mapping)

        # We'll import StaticGraphModel locally to avoid circular imports
        from integration_environment.network_models.static_graph_model import StaticGraphModel

        if topology_dict:
            self.static_graph_model = StaticGraphModel.from_dict(topology_data=topology_dict)
        elif topology_file_name:
            self.static_graph_model = StaticGraphModel.from_json_file(file_path=topology_file_name)
        else:
            raise ValueError(
                'Topology information must be provided in order to initialize StaticDelayGraphModelScheduler.')

    async def process_message_output(self,
                                     container_messages_dict: dict[str, list[ExternalAgentMessage]],
                                     next_activities):
        """
        Process message outputs with static pre-configured end-to-end delays.

        :param container_messages_dict: Dictionary mapping container names to their outgoing messages.
        :param next_activities: List of timestamps for the next scheduled activities of containers.
        """
        for container_name, messages in container_messages_dict.items():
            for message in messages:
                try:
                    # Get the pre-configured delay for this sender-receiver pair
                    delay_ms = self.static_graph_model.get_delay(sender_id=container_name,
                                                                 receiver_id=message.receiver)

                    # Calculate the message delivery time
                    message_delivery_time_ms = math.ceil(message.time * 1000) + delay_ms
                    message_delivery_time_s = message_delivery_time_ms / 1000

                    # Add message to buffer for delivery at the calculated time
                    if message_delivery_time_s not in self._message_buffer:
                        self._message_buffer[message_delivery_time_s] = []
                    self._message_buffer[message_delivery_time_s].append(message)

                except ValueError as e:
                    # Log warning about missing path and deliver instantly as fallback
                    print(f"Warning: {e}. Delivering message instantly as fallback.")
                    message_delivery_time_ms = math.ceil(message.time * 1000)
                    message_delivery_time_s = message_delivery_time_ms / 1000
                    if message_delivery_time_s not in self._message_buffer:
                        self._message_buffer[message_delivery_time_s] = []
                    self._message_buffer[message_delivery_time_s].append(message)

        # Update next activities
        self._next_activities.extend([na for na in next_activities if na is not None])
        self._next_activities = [na for na in self._next_activities if na >= self.current_time]
