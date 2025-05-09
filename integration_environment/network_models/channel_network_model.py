import json
import math
from typing import List, Dict, Any

import networkx as nx


class ChannelNetworkModel:
    """
    Network topology model based on NetworkX graph library.

    This class provides a wrapper around NetworkX for managing network topology
    and calculating message delays in cyber-physical energy systems simulations.
    """

    def __init__(self):
        """
        Initialize a new NetworkX-based topology model.
        """
        self.graph = nx.Graph()

    def add_node(self, node_id: str, **attributes):
        """
        Add a node to the network topology.
        :param node_id: Unique identifier for the node.
        :param attributes: Node attributes such as position, processing_delay_ms, etc.
        """
        self.graph.add_node(node_id, **attributes)

    def add_link(self, source_id: str, target_id: str, **attributes):
        """
        Add a link (edge) between two nodes in the network.

        :param source_id: ID of the source node.
        :param target_id: ID of the target node.
        :param attributes: Link attributes such as transmission_rate_bps, propagation_speed_mps, etc.
        """
        # Calculate distance between nodes if positions are available
        if ('position' in self.graph.nodes[source_id] and
                'position' in self.graph.nodes[target_id]):
            pos_a = self.graph.nodes[source_id]['position']
            pos_b = self.graph.nodes[target_id]['position']
            # Calculate Euclidean distance
            dimensions = min(len(pos_a), len(pos_b))
            distance = math.sqrt(sum((pos_a[i] - pos_b[i]) ** 2 for i in range(dimensions)))
            attributes['distance'] = distance

        # Add edge to the graph with all attributes
        self.graph.add_edge(source_id, target_id, **attributes)

    def calculate_propagation_delay(self, source_id: str, target_id: str) -> float:
        """
        Calculate the propagation delay for a link.
        Propagation delay is the time taken for a signal to travel from one end of the
        link to the other, determined by distance and propagation speed.

        :param source_id: ID of the source node.
        :param target_id: ID of the target node.
        :return: Propagation delay in milliseconds.
        """
        if not self.graph.has_edge(source_id, target_id):
            raise ValueError(f"No link exists between {source_id} and {target_id}")

        edge_data = self.graph.get_edge_data(source_id, target_id)

        if 'distance' not in edge_data or 'propagation_speed_mps' not in edge_data:
            raise ValueError(f"Link between {source_id} and {target_id} is missing distance or propagation_speed_mps")

        # Calculate propagation delay: distance / speed, converted to milliseconds
        return (edge_data['distance'] / edge_data['propagation_speed_mps']) * 1000

    def calculate_transmission_delay(self, source_id: str, target_id: str, message_size_bits: int) -> float:
        """
        Calculate the transmission delay for a message over a link.

        Transmission delay is the time taken to push all message bits onto the link,
        determined by message size and link bandwidth.

        :param source_id: ID of the source node.
        :param target_id: ID of the target node.
        :param message_size_bits: Size of the message in bits.
        :return: Transmission delay in milliseconds.
        """
        if not self.graph.has_edge(source_id, target_id):
            raise ValueError(f"No link exists between {source_id} and {target_id}")

        edge_data = self.graph.get_edge_data(source_id, target_id)

        if 'transmission_rate_bps' not in edge_data:
            raise ValueError(f"Link between {source_id} and {target_id} is missing transmission_rate_bps")

        # Calculate transmission delay: message_size / rate, converted to milliseconds
        return (message_size_bits / edge_data['transmission_rate_bps']) * 1000

    def find_shortest_path(self, source_id: str, target_id: str) -> List[str]:
        """
        Find the shortest path between two nodes in the network.
        :param source_id: ID of the source node.
        :param target_id: ID of the target node.
        :return: Ordered list of node IDs that form the shortest path.
        """
        return nx.shortest_path(self.graph, source=source_id, target=target_id, weight='delay')

    def calculate_end_to_end_delay(self, sender_id: str, receiver_id: str, message_size_bits: int) -> float:
        """
        Calculate the total end-to-end delay for a message from source to destination.

        The end-to-end delay includes:
        - Processing delays at all nodes in the path
        - Transmission delays for each link (based on message size and link bandwidth)
        - Propagation delays for each link (based on distance and propagation speed)

        :param sender_id: ID of the source node.
        :param receiver_id: ID of the target node.
        :param message_size_bits: Size of the message in bits.
        :return: Total end-to-end delay in milliseconds.
        """
        try:
            # Find the shortest path between source and target
            path = self.find_shortest_path(sender_id, receiver_id)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found from {sender_id} to {receiver_id}")

        # Calculate processing delays at nodes
        processing_delay = sum(self.graph.nodes[node_id].get('processing_delay_ms', 0)
                               for node_id in path)

        # Calculate transmission and propagation delays for each link in the path
        transmission_delay = 0
        propagation_delay = 0

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            transmission_delay += self.calculate_transmission_delay(
                current_node, next_node, message_size_bits)
            propagation_delay += self.calculate_propagation_delay(
                current_node, next_node)

        return round(processing_delay + transmission_delay + propagation_delay)

    @classmethod
    def from_dict(cls, topology_data: Dict[str, Any]) -> 'ChannelNetworkModel':
        """
        Create a NetworkXTopologyModel from a dictionary representation.
        :param topology_data: Dictionary containing nodes and links data.
        :return: A new topology model instance.
        """
        model = cls()

        # Add nodes
        for node_data in topology_data.get("nodes", []):
            node_id = node_data.pop("node_id")
            model.add_node(node_id, **node_data)

        # Add links
        for link_data in topology_data.get("links", []):
            source_id = link_data.pop("source")
            target_id = link_data.pop("target")
            model.add_link(source_id, target_id, **link_data)

        return model

    @classmethod
    def from_json_file(cls, file_path: str) -> 'ChannelNetworkModel':
        """
        Load a topology model from a JSON file.
        :param file_path: Path to the JSON file containing topology data.
        :return: A new topology model instance.
        """
        with open(file_path, 'r') as f:
            topology_data = json.load(f)

        return cls.from_dict(topology_data)
