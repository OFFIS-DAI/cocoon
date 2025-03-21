from abc import ABC
from enum import Enum
import numpy as np
import pandas as pd

from src.events import MessageArrival, MessageDeparture
from src.utils.util import LoggingOption


class State(ABC):
    """
    Abstract class for state definitions.
    """

    def __init__(self):
        pass

    class StateDescription(Enum):
        """
        Description of the state (could be idle, busy (or error)).
        """
        IDLE = 0
        BUSY = 1


class NodeState(State):
    """
    Holds the state variables of a node in the network graph.
    """

    def __init__(self, node_id: str, logging_option: LoggingOption):
        super().__init__()
        self.node_id = node_id
        self.logging_option = logging_option

        self.arrived_messages = {}  # {arrival_time1: [message1, message2], ... }
        self.departure_messages = {}  # {departure_time1: [message1, message2], ... }

        self.average_incoming_delay_time = np.nan  # average delay time for incoming messages
        self.average_outgoing_delay_time = np.nan  # average delay time for outgoing messages
        self.median_incoming_delay_time = np.nan  # average delay time for incoming messages
        self.median_outgoing_delay_time = np.nan  # average delay time for outgoing messages

        self.current_inter_arrival_time = np.nan  # current time between now and receiving the last message
        self.average_inter_arrival_time = np.nan  # average time between receiving messages
        self.current_inter_departure_time = np.nan  # current time between now and sending the last message
        self.average_inter_departure_time = np.nan  # average time between sending messages

        self.num_messages_sent_simultaneously = np.nan
        self.average_num_messages_sent_simultaneously = np.nan

    def get_as_dataframe(self):
        # Define the node state metrics
        node_state_metrics = {
            'node_id': self.node_id,
            'average_incoming_delay_time': self.average_incoming_delay_time,
            'average_outgoing_delay_time': self.average_outgoing_delay_time,
            'median_incoming_delay_time': self.median_incoming_delay_time,
            'median_outgoing_delay_time': self.median_outgoing_delay_time,
            'current_inter_arrival_time': self.current_inter_arrival_time,
            'average_inter_arrival_time': self.average_inter_arrival_time,
            'current_inter_departure_time': self.current_inter_departure_time,
            'average_inter_departure_time': self.average_inter_departure_time,
            'num_messages_sent_simultaneously': self.num_messages_sent_simultaneously,
            'average_num_messages_sent_simultaneously': self.average_num_messages_sent_simultaneously
        }
        return pd.DataFrame([node_state_metrics])

    def update_state(self,
                     current_time_ms: int):
        """
        Updates state at given time.
        :param current_time_ms: time in ms.
        """
        if len(self.arrived_messages) > 0:
            self.num_messages_sent_simultaneously = len(self.arrived_messages[current_time_ms]) \
                if current_time_ms in self.arrived_messages.keys() else 0
            self.average_num_messages_sent_simultaneously = np.mean([len(msg_list)
                                                                     for msg_list in self.arrived_messages.values()])

            arr_messages = np.concatenate(list(self.arrived_messages.values()))
            self.average_outgoing_delay_time = np.mean([msg.departure_time_ms - msg.event_time_ms
                                                        for msg in arr_messages])
            self.median_outgoing_delay_time = np.median([msg.departure_time_ms - msg.event_time_ms
                                                        for msg in arr_messages])
            arrival_times = list(self.arrived_messages.keys())

            self.current_inter_arrival_time = current_time_ms - max(arrival_times)

            if len(arrival_times) == 1:
                self.average_inter_arrival_time = self.current_inter_arrival_time
            else:
                self.average_inter_arrival_time = np.mean([arrival_times[i + 1] - arrival_times[i]
                                                           for i in range(len(arrival_times) - 1)])

        if len(self.departure_messages) > 0:
            dep_messages = np.concatenate(list(self.departure_messages.values()))
            self.average_incoming_delay_time = np.mean([msg.event_time_ms - msg.arrival_time_ms
                                                        for msg in dep_messages])
            self.median_incoming_delay_time = np.median([msg.event_time_ms - msg.arrival_time_ms
                                                        for msg in dep_messages])

            departure_times = list(self.departure_messages.keys())

            self.current_inter_departure_time = current_time_ms - max(departure_times)

            if len(departure_times) == 1:
                self.average_inter_departure_time = self.current_inter_departure_time
            else:
                self.average_inter_departure_time = np.mean([departure_times[i + 1] - departure_times[i]
                                                             for i in range(len(departure_times) - 1)])

    def process_message_arrival(self,
                                message: MessageArrival):
        """
        Processes message arrival event and updates state accordingly.
        :param message: event object.
        """

        event_time = message.event_time_ms

        if self.logging_option == LoggingOption.DEBUG:
            print(f'{event_time}: Process message arrival at node')

        if event_time not in self.arrived_messages.keys():
            self.arrived_messages[event_time] = []
        self.arrived_messages[event_time].append(message)

    def process_message_departure(self,
                                  message: MessageDeparture):
        """
        Processes message departure event and updates state accordingly.
        :param message: event object.
        """
        event_time = message.event_time_ms

        if self.logging_option == LoggingOption.DEBUG:
            print(f'{event_time}: Process message departure')

        # add message to departure map
        if event_time not in self.departure_messages.keys():
            self.departure_messages[event_time] = []
        self.departure_messages[event_time].append(message)


class LinkState(State):
    """
    Holds the state variables of the link in the network graph.
    """

    def __init__(self, link_id: str, logging_option: LoggingOption):
        super().__init__()
        self.link_id = link_id
        self.logging_option = logging_option

        self.state = self.StateDescription.IDLE
        self.arrived_messages = {}
        self.departure_messages = {}
        self.state_transitions = {0: self.StateDescription.IDLE}  # time_ms : StateDescription
        self.queue = []

        ###
        self.resource_utilization = 0
        self.throughput_bps = 0
        self.average_delay_time = 0
        self.median_delay_time = 0
        self.average_inter_departure_time = 0
        self.current_inter_arrival_time = 0
        self.average_inter_arrival_time = 0
        self.current_idle_time = 0
        self.average_idle_time = 0

    def get_as_dataframe(self):
        # Define the link state metrics
        link_state_metrics = {
            'state': self.state.value,
            'resource_utilization': self.resource_utilization,
            'throughput_bps': self.throughput_bps,
            'average_delay_time': self.average_delay_time,
            'median_delay_time': self.median_delay_time,
            'average_inter_departure_time': self.average_inter_departure_time,
            'current_inter_arrival_time': self.current_inter_arrival_time,
            'average_inter_arrival_time': self.average_inter_arrival_time,
            'current_idle_time': self.current_idle_time,
            'average_idle_time': self.average_idle_time
        }
        return pd.DataFrame([link_state_metrics])

    def update_state(self,
                     current_time_ms: int):
        """
        Updates state at given time.
        :param current_time_ms: time in ms.
        """
        self.current_inter_arrival_time = self.get_current_inter_arrival_time(current_time_ms)
        self.average_inter_arrival_time = self.get_average_inter_arrival_time()
        self.average_inter_departure_time = self.get_average_inter_departure_time()
        self.current_idle_time = self.get_current_idle_time(current_time_ms)
        self.average_idle_time = self.get_average_idle_time()
        self.average_delay_time = self.get_average_delay_time()
        self.median_delay_time = self.get_median_delay_time()
        self.throughput_bps = self.get_throughput()
        self.resource_utilization = self.get_resource_utilization(current_time_ms)

        self.print_state()

    def print_state(self):
        if self.logging_option == LoggingOption.DEBUG:
            print('---- State description -----')
            print('current inter-arrival time = ', self.current_inter_arrival_time)
            print('average inter-arrival time = ', self.average_inter_arrival_time)
            print('average inter-departure time = ', self.average_inter_departure_time)
            print('average delay time = ', self.average_delay_time)
            print('current idle time = ', self.current_idle_time)
            print('average idle time = ', self.average_idle_time)
            print('current queue length = ', len(self.queue))
            print('average throughput [in Byte per second] = ', self.throughput_bps)
            print('resource utilization [in %] = ', self.resource_utilization)

    def get_current_queue_length(self):
        """
        Gets current length of the queue.
        :return: length of queue.
        """
        return len(self.queue)

    def get_average_delay_time(self):
        """
        Gets average delay time.
        :return: average delay time in ms.
        """
        if len(self.departure_messages) == 0:
            return np.nan
        messages = np.concatenate(list(self.departure_messages.values()))
        return np.mean([message.event_time_ms - message.arrival_time_ms for message in messages])

    def get_median_delay_time(self):
        """
        Gets average delay time.
        :return: average delay time in ms.
        """
        if len(self.departure_messages) == 0:
            return np.nan
        messages = np.concatenate(list(self.departure_messages.values()))
        return np.median([message.event_time_ms - message.arrival_time_ms for message in messages])

    def get_resource_utilization(self,
                                 current_time_ms: int):
        """
        Resource utilization is defined as percentage of the time in state busy.
        :param current_time_ms: current time in ms.
        :return: resource utilization in %.
        """
        # is defined as percentage of the time in state busy
        busy_since = None
        busy_times = []
        for transition_time, state_transition in self.state_transitions.items():
            if state_transition == self.StateDescription.IDLE and busy_since is not None:
                busy_times.append(transition_time - busy_since)
            elif state_transition == self.StateDescription.BUSY:
                busy_since = transition_time
        if len(busy_times) == 0:
            return 0
        return (sum(busy_times) / current_time_ms) * 100  # in %

    def get_throughput(self):
        """
        Calculates throughput, with is total Bytes per time.
        :return: throughput in Bytes per second.
        """
        if len(self.departure_messages) == 0:
            return 0
        last_event_time = max(list(self.departure_messages.keys()))
        if last_event_time == 0:
            return 0
        messages = np.concatenate(list(self.departure_messages.values()))
        total_bytes_sent = sum([message.size_B for message in messages])
        # throughput is total Bytes per time (multiplied with 1000 because we convert from ms to s)
        return total_bytes_sent / last_event_time * 1000

    def get_current_inter_arrival_time(self,
                                       current_time_ms: int):
        """
        Get current inter-arrival time, which is the time since last arrival event.
        :param current_time_ms: current time in ms.
        :return: current inter-arrival time in ms.
        """
        if len(self.arrived_messages) == 0:
            return np.nan
        return current_time_ms - max(list(self.arrived_messages.keys()))

    def get_average_inter_departure_time(self):
        """
        Gets average inter-departure time.
        :return: average inter-departure time in ms.
        """
        if len(self.departure_messages) == 0:
            return np.nan
        departure_times = list(self.departure_messages.keys())
        if len(departure_times) == 1:
            return np.nan
        inter_departure_times = [departure_times[i + 1] - departure_times[i] for i in range(len(departure_times) - 1)]
        if len(inter_departure_times) == 0:
            return np.nan
        return np.mean(inter_departure_times)

    def get_average_inter_arrival_time(self):
        """
        Gets average inter-arrival time.
        :return: average inter-arrival time in ms, if messages arrived. Else inf.
        """
        if len(self.arrived_messages) == 0:
            return np.nan
        arrival_times = list(self.arrived_messages.keys())
        if len(arrival_times) == 1:
            return arrival_times[0]
        inter_arrival_times = [arrival_times[i + 1] - arrival_times[i] for i in range(len(arrival_times) - 1)]
        if len(inter_arrival_times) == 0:
            return np.nan
        return np.mean(inter_arrival_times)

    def get_current_idle_time(self,
                              current_time_ms: int):
        """
        Gets current time in state idle.
        :param current_time_ms: current time in ms.
        :return: current time in state idle, if in idle. Else 0.
        """
        if self.state == self.StateDescription.BUSY:
            return 0
        else:
            return current_time_ms - max(list(self.state_transitions.keys()))

    def get_average_idle_time(self):
        """
        Gets average time in state idle.
        :return: Average idle time in ms.
        """
        idle_times = []
        currently_idle = True
        idle_start = 0
        for state_transition_time, transition in self.state_transitions.items():
            if transition == self.StateDescription.BUSY and currently_idle:
                idle_times.append(state_transition_time - idle_start)
                currently_idle = False
            elif transition == self.StateDescription.IDLE and not currently_idle:
                currently_idle = True
                idle_start = state_transition_time
        if len(idle_times) == 0:
            return np.nan
        return np.mean(idle_times)

    def process_message_arrival(self,
                                message: MessageArrival):
        """
        Processes message arrival event and updates state accordingly.
        :param message: event object.
        """
        if self.logging_option == LoggingOption.DEBUG:
            print(f'{message.event_time_ms}: Process message arrival')
        event_time = message.event_time_ms
        if self.state == self.StateDescription.IDLE:
            self.state_transitions[event_time] = self.StateDescription.BUSY
        self.state = self.StateDescription.BUSY

        if event_time not in self.arrived_messages.keys():
            self.arrived_messages[event_time] = []
        self.arrived_messages[event_time].append(message)
        self.queue.append(message)

    def process_message_departure(self,
                                  message: MessageDeparture):
        """
        Processes message departure event and updates state accordingly.
        :param message: event object.
        """
        if self.logging_option == LoggingOption.DEBUG:
            print(f'{message.event_time_ms}: Process message departure')
        event_time = message.event_time_ms

        # enqueue message
        self.queue = [event for event in self.queue if event.msg_id != message.msg_id]
        # update current state
        if self.state == self.StateDescription.IDLE:
            raise ValueError('Communication Link is in infeasible state IDLE. ')
        if len(self.queue) == 0 and self.state == self.StateDescription.BUSY:
            # change state to idle
            self.state_transitions[event_time] = self.StateDescription.IDLE
            self.state = self.StateDescription.IDLE
        # add message to departure map
        if event_time not in self.departure_messages.keys():
            self.departure_messages[event_time] = []
        self.departure_messages[event_time].append(message)


class NetworkState(State):
    """
    Class that defines the state of the network.
    """

    def __init__(self, logging_option):  # TODO: maybe move some parameter to State class
        super().__init__()  # TODO: define state, initialize from historic simulation_data
        self.logging_option = logging_option

        self.link_states = {}

        # average link states in network
        self.average_throughput_bps = 0
        self.average_delay_time = 0
        self.median_delay_time = 0
        self.average_inter_departure_time = 0
        self.average_inter_arrival_time = 0
        self.average_idle_time = 0
        self.mean_resource_utilization = 0
        self.messages_sent_at_current_time = 0

        # network state
        self.num_messages_in_transit = 0
        self.num_busy_links = 0
        self.num_network_nodes = 0

    def get_as_dataframe(self) -> pd.DataFrame:
        # Define the network state metrics
        network_state_metrics = {
            'average_throughput_bps': self.average_throughput_bps,
            'average_delay_time': self.average_delay_time,
            'median_delay_time': self.median_delay_time,
            'average_inter_departure_time': self.average_inter_departure_time,
            'average_inter_arrival_time': self.average_inter_arrival_time,
            'average_idle_time': self.average_idle_time,
            'mean_resource_utilization': self.mean_resource_utilization,
            'num_messages_in_transit': self.num_messages_in_transit,
            'num_busy_links': self.num_busy_links,
            'num_network_nodes': self.num_network_nodes,
            'messages_sent_at_current_time': self.messages_sent_at_current_time
        }
        return pd.DataFrame([network_state_metrics])

    def update_state(self,
                     link_state: LinkState,
                     network,
                     current_time_ms: int):
        """
        Updates network state given the link state.
        :param link_state: link state object.
        :param network: communication network object.
        :param current_time_ms: current time in ms.
        """
        self.link_states[link_state.link_id] = link_state

        self.messages_sent_at_current_time = sum([len(state.arrived_messages[current_time_ms])
                                                  if current_time_ms in state.arrived_messages.keys()
                                                  else 0 for state in self.link_states.values()])

        self.average_delay_time = np.mean([state.average_delay_time for state in self.link_states.values()])
        self.median_delay_time = np.median([state.average_delay_time for state in self.link_states.values()])
        self.average_throughput_bps = np.mean([state.throughput_bps for state in self.link_states.values()])
        self.average_inter_departure_time = np.mean(
            [state.average_inter_departure_time for state in self.link_states.values()])
        self.average_inter_arrival_time = np.mean(
            [state.average_inter_arrival_time for state in self.link_states.values()])
        self.average_idle_time = np.mean(
            [state.average_idle_time for state in self.link_states.values()])
        self.mean_resource_utilization = np.mean([state.resource_utilization for state in self.link_states.values()])

        self.num_messages_in_transit = len(np.concatenate([state.queue for state in self.link_states.values()]))
        self.num_busy_links = len([1 for state in self.link_states.values()
                                   if state.state == LinkState.StateDescription.BUSY])
        self.num_network_nodes = len(network.nodes)

        if self.logging_option == LoggingOption.DEBUG:
            print('---- NETWORK STATE ----')
            print('number of link states: ', len(self.link_states))
            print('average delay times: ', self.average_delay_time)
            print('average throughput: ', self.average_throughput_bps)
            print('average inter departure time: ', self.average_inter_departure_time)
            print('average inter arrival time: ', self.average_inter_arrival_time)
            print('average idle time: ', self.average_idle_time)
            print('mean resource utilization: ', self.mean_resource_utilization)
            print('num messages in transit: ', self.num_messages_in_transit)
            print('num busy links: ', self.num_busy_links)
            print('num network nodes: ', self.num_network_nodes)
