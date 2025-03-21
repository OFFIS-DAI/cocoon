from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from src.events import MessageArrival, MessageDeparture
from src.state_definitions import LinkState, NetworkState, NodeState
from src.utils.util import LoggingOption


class CommunicationNode:
    """
    Holds a node in the communication network graph.
    """

    def __init__(self,
                 node_id: str,
                 logging_option: LoggingOption):
        self.node_id = node_id

        self.node_state = NodeState(node_id=self.node_id, logging_option=logging_option)
        self.state_data = pd.DataFrame()

    def update_state(self, current_time_ms: int):
        self.node_state.update_state(current_time_ms=current_time_ms)


class CommunicationLink:
    """
    Holds a link in the communication network graph.
    """

    def __init__(self,
                 node1: CommunicationNode,
                 node2: CommunicationNode,
                 network,
                 logging_option: LoggingOption):
        self.logging_option = logging_option

        self.node1 = node1
        self.node2 = node2
        self.link_id = node1.node_id + '_' + node2.node_id

        self.link_state = LinkState(link_id=self.link_id, logging_option=logging_option)
        self.network = network

        self.state_data = pd.DataFrame()

    def update_states(self,
                      message_arrival: MessageArrival):
        """
        Updates link and network state on message arrival.
        :param message_arrival: message arrival event.
        """
        event_time = message_arrival.event_time_ms

        if self.logging_option == LoggingOption.DEBUG:
            print(f'{event_time}: arrival on ', self.node1.node_id, '-->', self.node2.node_id)
        self.link_state.update_state(event_time)
        self.node1.update_state(event_time)
        self.node2.update_state(event_time)
        self.network.network_state.update_state(link_state=self.link_state, network=self.network,
                                                current_time_ms=message_arrival.event_time_ms)

    def save_state_data(self,
                        message_arrival: MessageArrival):
        """
        Saves state data in training phase on message arrival event.
        :param message_arrival: event of message arrival.
        """
        # in training phase: save link and network state and delay time
        network_state_df = self.network.network_state.get_as_dataframe()
        columns = network_state_df.columns
        network_state_df.rename(columns={col: f'network_{col}' for col in columns}, inplace=True)
        link_state_df = self.link_state.get_as_dataframe()
        sender_node_state_df = self.node1.node_state.get_as_dataframe()
        sender_node_state_df.rename(columns={col: f'sender_{col}' for col in sender_node_state_df.columns},
                                    inplace=True)
        receiver_node_state_df = self.node2.node_state.get_as_dataframe()
        receiver_node_state_df.rename(columns={col: f'receiver_{col}' for col in receiver_node_state_df.columns},
                                      inplace=True)
        agg_df = pd.concat([network_state_df, link_state_df, sender_node_state_df, receiver_node_state_df],
                           axis=1)
        agg_df['delay_ms'] = message_arrival.departure_time_ms - message_arrival.event_time_ms
        agg_df['time_ms'] = int(message_arrival.event_time_ms)
        agg_df['packet_size_B'] = message_arrival.size_B

        self.state_data = pd.concat([self.state_data, agg_df])

    def predict_message_departure(self,
                                  message_arrival: MessageArrival):
        # in production phase
        # predicted_departure = self.link_behaviour_predictor.predict_message_departure(message_arrival,
        #                                                                              self.link_state)
        pass

    def process_message_arrival_event(self,
                                      message_arrival: MessageArrival):
        """
        Processes message arrival in link state.
        :param message_arrival: event of message arrival.
        """
        self.link_state.process_message_arrival(message_arrival)
        self.node1.node_state.process_message_arrival(message_arrival)

    def process_message_departure_event(self,
                                        message_departure: MessageDeparture):
        """
        Processes message departure in link and network state.
        :param message_departure: event of message departure.
        """
        self.link_state.process_message_departure(message_departure)
        self.node2.node_state.process_message_departure(message_departure)
        self.link_state.update_state(current_time_ms=message_departure.event_time_ms)
        self.network.network_state.update_state(link_state=self.link_state, network=self.network,
                                                current_time_ms=message_departure.event_time_ms)


class CommunicationNetworkGraph:
    """
    Holds network graph with links and nodes.
    """

    def __init__(self, logging_option):
        self.nodes = []
        self.links = []
        self.network_state = NetworkState(logging_option=logging_option)
        self.logging_option = logging_option

    def get_state_data_from_links(self) -> pd.DataFrame:
        """
        Gets state data from links for results recording.
        :return: dataframe with state data.
        """
        aggregated_df = pd.DataFrame()
        for link in self.links:
            aggregated_df = pd.concat([aggregated_df, link.state_data])
        return aggregated_df

    def node_exists(self,
                    node_name: str):
        """
        Returns if node with given name exists.
        :param node_name: name of the node.
        :return: True, if node exists in graph.
        """
        return len([node for node in self.nodes if node.node_id == node_name]) > 0

    def initialize_node(self,
                        node_id: str) -> CommunicationNode:
        """
        Initializes node with given name.
        :param node_id: name of the node.
        :return: node object.
        """
        node = CommunicationNode(node_id, logging_option=self.logging_option)
        self.nodes.append(node)
        return node

    def get_node(self,
                 node_name: str) -> Optional[CommunicationNode]:
        """
        Gets node from graph.
        :param node_name: name of the node.
        :return: node object.
        """
        node = [node for node in self.nodes if node.node_id == node_name]
        if len(node) == 0:
            return None
        return node[0]

    def link_exists(self,
                    node1: CommunicationNode,
                    node2: CommunicationNode):
        """
        Check if link between nodes exists.
        :param node1: sender node.
        :param node2: receiver node.
        :return: True, if link exists.
        """
        return len([link for link in self.links if link.node1 == node1 and link.node2 == node2]) > 0

    def initialize_link(self,
                        node1: CommunicationNode,
                        node2: CommunicationNode):
        """
        Initializes link between nodes.
        :param node1: sender node.
        :param node2: receiver node.
        :return: link object.
        """
        link = CommunicationLink(node1=node1,
                                 node2=node2,
                                 network=self,
                                 logging_option=self.logging_option)
        self.links.append(link)
        return link

    def get_link(self,
                 node1: CommunicationNode,
                 node2: CommunicationNode) -> Optional[CommunicationLink]:
        """
        Returns link between nodes.
        :param node1: sender node.
        :param node2: receiver node.
        :return: link object.
        """
        link = [link for link in self.links if link.node1 == node1 and link.node2 == node2]
        if len(link) == 0:
            return None
        return link[0]

    def check_or_initialize_model_for_message_transit(self,
                                                      sender: str,
                                                      receiver: str) -> (CommunicationNode,
                                                                         CommunicationNode,
                                                                         CommunicationLink):
        """
        Check if modeled elements exist for message transit.
        :param sender: sender name.
        :param receiver: receiver name.
        :return: sender node object, receiver node object, link object.
        """
        if not self.node_exists(sender):
            node1 = self.initialize_node(sender)
        else:
            node1 = self.get_node(node_name=sender)
        if not self.node_exists(receiver):
            node2 = self.initialize_node(receiver)
        else:
            node2 = self.get_node(node_name=receiver)

        if not self.link_exists(node1, node2):
            link = self.initialize_link(node1, node2)
        else:
            link = self.get_link(node1, node2)
        return node1, node2, link

    def get_link_by_sender_and_receiver_names(self,
                                              sender: str,
                                              receiver: str) -> Optional[CommunicationLink]:
        """
        Gets link by sender and receiver name.
        :param sender: sender name.
        :param receiver: receiver name.
        :return: link object.
        """
        node1 = self.get_node(node_name=sender)
        node2 = self.get_node(node_name=receiver)

        if node1 is None or node2 is None:
            return None

        return self.get_link(node1, node2)

    def plot_graph(self):
        """
        Plots graph of communication network.
        """
        G = nx.Graph()

        for node in self.nodes:
            G.add_node(node.name)

        for link in self.links:
            G.add_edge(link.node1.name, link.node2.name)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, edge_color='gray', linewidths=1.5,
                font_size=8)
        plt.title("Communication Network Graph")
        plt.show()
