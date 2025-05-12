import json
import networkx as nx
from typing import Dict, List, Optional, Tuple


class StaticGraphModel:
    """
    A static graph model for communication networks with pre-configured end-to-end delays.

    This model uses NetworkX graphs internally to store network topology information including
    nodes and links with their associated end-to-end delays as edge weights. It provides methods
    to query delays between any two nodes in the network and supports multi-hop path finding.
    """

    def __init__(self, topology_data: Optional[Dict] = None):
        """
        Initialize the static graph model.

        :param topology_data: Dictionary containing network topology information with nodes and links.
        """
        # Use undirected graph by default for bidirectional communication
        self.graph = nx.Graph()

        if topology_data:
            self._load_topology(topology_data)

    @classmethod
    def from_dict(cls, topology_data: Dict) -> 'StaticGraphModel':
        """
        Create a StaticGraphModel instance from a topology dictionary.

        :param topology_data: Dictionary containing network topology information.
        :return: StaticGraphModel instance.
        """
        return cls(topology_data)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'StaticGraphModel':
        """
        Create a StaticGraphModel instance from a JSON file.

        :param file_path: Path to the JSON file containing topology information.
        :return: StaticGraphModel instance.
        """
        with open(file_path, 'r') as file:
            topology_data = json.load(file)
        return cls(topology_data)

    def _load_topology(self, topology_data: Dict) -> None:
        """
        Load topology data into the NetworkX graph.

        :param topology_data: Dictionary containing network topology information.
        """
        # Load nodes
        if 'nodes' in topology_data:
            for node in topology_data['nodes']:
                node_id = node['node_id']
                # Store node attributes (excluding node_id as it's already the key)
                node_attrs = {k: v for k, v in node.items() if k != 'node_id'}
                self.graph.add_node(node_id, **node_attrs)

        # Load links and their delays as edge weights
        if 'links' in topology_data:
            for link in topology_data['links']:
                node_a = link['node_a']
                node_b = link['node_b']
                delay_ms = link['end-to-end-delay_ms']

                # Add edge with weight (delay) and other attributes
                edge_attrs = {k: v for k, v in link.items()
                              if k not in ['node_a', 'node_b', 'end-to-end-delay_ms']}
                self.graph.add_edge(node_a, node_b, weight=delay_ms, **edge_attrs)

    def get_delay(self, sender_id: str, receiver_id: str, method: str = 'shortest') -> float:
        """
        Get the end-to-end delay between two nodes.

        :param sender_id: ID of the sender node.
        :param receiver_id: ID of the receiver node.
        :param method: Method to calculate delay. Options: 'shortest', 'direct'
                      - 'shortest': Uses shortest path algorithm (default)
                      - 'direct': Only considers direct links
        :return: End-to-end delay in milliseconds.
        :raises ValueError: If no path exists between the nodes.
        """
        if sender_id == receiver_id:
            return 0.0

        if not self.has_node(sender_id):
            raise ValueError(f"Node {sender_id} not found in the network")
        if not self.has_node(receiver_id):
            raise ValueError(f"Node {receiver_id} not found in the network")

        if method == 'direct':
            # Check for direct link only
            if self.graph.has_edge(sender_id, receiver_id):
                return self.graph[sender_id][receiver_id]['weight']
            else:
                raise ValueError(f"No direct path found between {sender_id} and {receiver_id}")

        elif method == 'shortest':
            try:
                # Use NetworkX shortest path algorithm with weights
                path_length = nx.shortest_path_length(self.graph, sender_id, receiver_id, weight='weight')
                return path_length
            except nx.NetworkXNoPath:
                raise ValueError(f"No path found between {sender_id} and {receiver_id}")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'shortest' or 'direct'.")

    def get_shortest_path(self, sender_id: str, receiver_id: str) -> Tuple[List[str], float]:
        """
        Get the shortest path and its total delay between two nodes.

        :param sender_id: ID of the sender node.
        :param receiver_id: ID of the receiver node.
        :return: Tuple of (path_as_list_of_nodes, total_delay)
        :raises ValueError: If no path exists between the nodes.
        """
        if sender_id == receiver_id:
            return ([sender_id], 0.0)

        try:
            path = nx.shortest_path(self.graph, sender_id, receiver_id, weight='weight')
            delay = nx.shortest_path_length(self.graph, sender_id, receiver_id, weight='weight')
            return (path, delay)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found between {sender_id} and {receiver_id}")

    def get_all_shortest_paths_delays(self, sender_id: str) -> Dict[str, float]:
        """
        Get the shortest path delays from a sender to all other nodes.

        :param sender_id: ID of the sender node.
        :return: Dictionary mapping receiver IDs to their shortest path delays.
        """
        if not self.has_node(sender_id):
            raise ValueError(f"Node {sender_id} not found in the network")

        try:
            lengths = nx.single_source_shortest_path_length(self.graph, sender_id, weight='weight')
            # Remove self-loop (delay to self)
            lengths.pop(sender_id, None)
            return lengths
        except nx.NetworkXError as e:
            raise ValueError(f"Error calculating shortest paths: {e}")

    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the network.

        :param node_id: ID of the node to check.
        :return: True if the node exists, False otherwise.
        """
        return self.graph.has_node(node_id)

    def has_edge(self, node_a: str, node_b: str) -> bool:
        """
        Check if an edge exists between two nodes.

        :param node_a: ID of the first node.
        :param node_b: ID of the second node.
        :return: True if the edge exists, False otherwise.
        """
        return self.graph.has_edge(node_a, node_b)

    def get_node_info(self, node_id: str) -> Dict:
        """
        Get information about a specific node.

        :param node_id: ID of the node.
        :return: Dictionary containing node information.
        :raises KeyError: If the node doesn't exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} not found in the network")
        return dict(self.graph.nodes[node_id])

    def get_edge_info(self, node_a: str, node_b: str) -> Dict:
        """
        Get information about a specific edge.

        :param node_a: ID of the first node.
        :param node_b: ID of the second node.
        :return: Dictionary containing edge information.
        :raises KeyError: If the edge doesn't exist.
        """
        if not self.has_edge(node_a, node_b):
            raise KeyError(f"Edge between {node_a} and {node_b} not found")
        return dict(self.graph[node_a][node_b])

    def get_all_nodes(self) -> List[str]:
        """
        Get a list of all node IDs in the network.

        :return: List of node IDs.
        """
        return list(self.graph.nodes())

    def get_all_edges(self) -> List[Tuple[str, str, Dict]]:
        """
        Get all edges in the network with their attributes.

        :return: List of tuples (node_a, node_b, edge_attributes).
        """
        return [(u, v, d) for u, v, d in self.graph.edges(data=True)]

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get all neighbors of a specific node.

        :param node_id: ID of the node.
        :return: List of neighbor node IDs.
        :raises KeyError: If the node doesn't exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} not found in the network")
        return list(self.graph.neighbors(node_id))

    def get_node_degree(self, node_id: str) -> int:
        """
        Get the degree (number of connections) of a node.

        :param node_id: ID of the node.
        :return: Degree of the node.
        :raises KeyError: If the node doesn't exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} not found in the network")
        return self.graph.degree[node_id]

    def add_node(self, node_id: str, **node_attrs) -> None:
        """
        Add a new node to the network.

        :param node_id: ID of the new node.
        :param node_attrs: Additional attributes for the node.
        """
        self.graph.add_node(node_id, **node_attrs)

    def add_link(self, node_a: str, node_b: str, delay_ms: float, **edge_attrs) -> None:
        """
        Add a link between two nodes with a specified delay.

        :param node_a: ID of the first node.
        :param node_b: ID of the second node.
        :param delay_ms: End-to-end delay in milliseconds.
        :param edge_attrs: Additional attributes for the edge.
        """
        if not self.has_node(node_a):
            raise ValueError(f"Node {node_a} not found in the network")
        if not self.has_node(node_b):
            raise ValueError(f"Node {node_b} not found in the network")

        self.graph.add_edge(node_a, node_b, weight=delay_ms, **edge_attrs)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its edges from the network.

        :param node_id: ID of the node to remove.
        :raises KeyError: If the node doesn't exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node {node_id} not found in the network")
        self.graph.remove_node(node_id)

    def remove_link(self, node_a: str, node_b: str) -> None:
        """
        Remove a link between two nodes.

        :param node_a: ID of the first node.
        :param node_b: ID of the second node.
        :raises KeyError: If the edge doesn't exist.
        """
        if not self.has_edge(node_a, node_b):
            raise KeyError(f"Edge between {node_a} and {node_b} not found")
        self.graph.remove_edge(node_a, node_b)

    def update_link_delay(self, node_a: str, node_b: str, new_delay_ms: float) -> None:
        """
        Update the delay for an existing link.

        :param node_a: ID of the first node.
        :param node_b: ID of the second node.
        :param new_delay_ms: New delay in milliseconds.
        :raises KeyError: If the edge doesn't exist.
        """
        if not self.has_edge(node_a, node_b):
            raise KeyError(f"Edge between {node_a} and {node_b} not found")
        self.graph[node_a][node_b]['weight'] = new_delay_ms

    def is_connected(self) -> bool:
        """
        Check if the graph is connected (all nodes can reach all other nodes).

        :return: True if the graph is connected, False otherwise.
        """
        return nx.is_connected(self.graph)

    def get_connected_components(self) -> List[List[str]]:
        """
        Get all connected components in the graph.

        :return: List of connected components, each as a list of node IDs.
        """
        return [list(component) for component in nx.connected_components(self.graph)]

    def to_dict(self) -> Dict:
        """
        Convert the static graph model to a dictionary representation.

        :return: Dictionary containing the network topology.
        """
        # Convert nodes with their attributes
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            node_dict = {'node_id': node_id}
            node_dict.update(attrs)
            nodes.append(node_dict)

        # Convert edges with their attributes
        links = []
        for node_a, node_b, attrs in self.graph.edges(data=True):
            link_dict = {
                'node_a': node_a,
                'node_b': node_b,
                'end-to-end-delay_ms': attrs['weight']
            }
            # Add other attributes (excluding weight which is already added as end-to-end-delay_ms)
            link_dict.update({k: v for k, v in attrs.items() if k != 'weight'})
            links.append(link_dict)

        return {
            'nodes': nodes,
            'links': links
        }

    def save_to_json(self, file_path: str) -> None:
        """
        Save the static graph model to a JSON file.

        :param file_path: Path where to save the JSON file.
        """
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)

    def get_graph_info(self) -> Dict:
        """
        Get general information about the graph.

        :return: Dictionary containing graph statistics.
        """
        return {
            'number_of_nodes': self.graph.number_of_nodes(),
            'number_of_edges': self.graph.number_of_edges(),
            'is_connected': self.is_connected(),
            'number_of_connected_components': nx.number_connected_components(self.graph),
            'average_degree': sum(dict(
                self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }