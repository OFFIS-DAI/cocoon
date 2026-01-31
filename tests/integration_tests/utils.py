"""
Test Utilities for COCOON Integration Tests.

This module provides common utilities for integration tests including
logging setup, path configuration, and visualization helpers.

Environment Variables (required for OMNeT++ integration tests):
    INET_INSTALLATION_PATH: Path to INET framework src directory
    SIMU5G_INSTALLATION_PATH: Path to Simu5G src directory
    OMNET_PROJECT_PATH: Path to the cocoon_omnet_project directory
"""

import logging
import os

import networkx as nx
from matplotlib import pyplot as plt

# OMNeT++ path configuration from environment variables
# Modify the defaults below for your system
INET_INSTALLATION_PATH = os.environ.get(
    'INET_INSTALLATION_PATH',
    '/home/malin/cocoon_omnet_workspace/inet4.5/src'
)
SIMU5G_INSTALLATION_PATH = os.environ.get(
    'SIMU5G_INSTALLATION_PATH',
    '/home/malin/PycharmProjects/trace/Simu5G-1.2.2/src'
)
OMNET_PROJECT_PATH = os.environ.get(
    'OMNET_PROJECT_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 'cocoon_omnet_project')
)


def omnet_configured() -> bool:
    """Check if OMNeT++ paths are configured and exist."""
    return os.path.exists(INET_INSTALLATION_PATH)


def initialize_results_dir():
    path = "results"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")


# Configure logging once at module level
# This ensures all loggers use this configuration
def setup_logging():
    initialize_results_dir()

    # Remove any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging to file
    logging.basicConfig(
        filename='results/integration_test_log.log',
        level=logging.INFO,
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


def visualize_channel_model_graph(topology, save_path='results/channel_network_topology.png', display=False):
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


def visualize_static_graph(topology, save_path='results/static_network_topology.png', **kwargs) -> None:
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
