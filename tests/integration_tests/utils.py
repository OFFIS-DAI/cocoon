import logging
import os

import networkx as nx
from matplotlib import pyplot as plt


# Configure logging once at module level
# This ensures all loggers use this configuration
def setup_logging():
    # Remove any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging to file
    logging.basicConfig(
        filename='integration_test_log.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='w'  # 'w' to overwrite the file each time
    )

    # Add console handler to see logs in terminal
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Suppress verbose debug logging from matplotlib and other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Create logger for this module
    return logging.getLogger(__name__)


def visualize_channel_model_graph(topology, save_path='channel_network_topology.png', display=False):
    # Extract positions for visualization (using just x,y coordinates)
    pos = {node: data['position'][:2] for node, data in topology.graph.nodes(data=True)}

    # Create figure
    plt.figure(figsize=(10, 8))

    # Draw nodes and edges
    nx.draw_networkx_nodes(topology.graph, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(topology.graph, pos, width=2, edge_color='gray')
    nx.draw_networkx_labels(topology.graph, pos, font_size=12)

    # Add edge labels (transmission rates)
    edge_labels = {(u, v): f"{d.get('transmission_rate_bps', 0) / 1e6:.1f} Mbps"
                   for u, v, d in topology.graph.edges(data=True)}
    nx.draw_networkx_edge_labels(topology.graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Network Topology")
    plt.axis('off')
    plt.tight_layout()

    # Always save the file
    plt.savefig(save_path)
    print(f"Network visualization saved to {os.path.abspath(save_path)}")

    # Only try to display if specifically requested
    if display:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot interactively: {e}")
            print(f"The plot has been saved to {save_path} instead.")

    # Close the figure to free memory
    plt.close()


def visualize_static_graph(topology, save_path='static_network_topology.png', **kwargs) -> None:
    """
    Visualize the network graph using matplotlib.

    :param save_path: Optional path to save the visualization.
    :param kwargs: Additional arguments for nx.draw().
    """
    try:
        import matplotlib.pyplot as plt

        # Set default layout and drawing parameters
        default_kwargs = {
            'with_labels': True,
            'node_color': 'lightblue',
            'node_size': 500,
            'font_size': 10,
            'font_weight': 'bold',
            'edge_color': 'gray',
            'width': 2
        }
        default_kwargs.update(kwargs)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(topology.graph)
        nx.draw(topology.graph, pos, **default_kwargs)

        # Draw edge labels with weights (delays)
        edge_labels = {(u, v): f"{d['weight']:.1f}ms"
                       for u, v, d in topology.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(topology.graph, pos, edge_labels, font_size=8)

        plt.title("Network Topology with Delays")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

    except ImportError:
        print("Matplotlib not available. Cannot visualize the network.")
