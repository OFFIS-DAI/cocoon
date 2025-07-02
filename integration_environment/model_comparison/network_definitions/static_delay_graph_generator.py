#!/usr/bin/env python3
"""
Static Delay Graph Generator

This script reads a CSV file containing message data with sender, receiver, and delay information,
and generates a static delay graph as a JSON file representing the network topology with
end-to-end delays between nodes.
"""

import pandas as pd
import json
from typing import Dict, List, Set


def load_message_data(csv_file: str) -> pd.DataFrame:
    """
    Load message data from CSV file.

    Args:
        csv_file: Path to the CSV file

    Returns:
        DataFrame with message data
    """
    try:
        df = pd.read_csv(csv_file)

        # Validate required columns
        required_columns = ['sender', 'receiver', 'delay_ms']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")


def extract_unique_nodes(df: pd.DataFrame) -> Set[str]:
    """
    Extract unique nodes from sender and receiver columns.

    Args:
        df: DataFrame with message data

    Returns:
        Set of unique node identifiers
    """
    senders = set(df['sender'].unique())
    receivers = set(df['receiver'].unique())
    return senders.union(receivers)


def calculate_pairwise_delays(df: pd.DataFrame) -> Dict[tuple, float]:
    """
    Calculate average delays between each pair of nodes.

    Args:
        df: DataFrame with message data

    Returns:
        Dictionary mapping (sender, receiver) tuples to average delays
    """
    # Group by sender-receiver pairs and calculate mean delay
    delay_stats = df.groupby(['sender', 'receiver'])['delay_ms'].agg(['mean', 'count']).reset_index()

    pairwise_delays = {}
    for _, row in delay_stats.iterrows():
        sender = row['sender']
        receiver = row['receiver']
        mean_delay = row['mean']
        message_count = row['count']

        print(f"  {sender} -> {receiver}: {mean_delay:.2f} ms (from {message_count} messages)")
        pairwise_delays[(sender, receiver)] = mean_delay

    return pairwise_delays


def calculate_global_mean_delay(df: pd.DataFrame) -> float:
    """
    Calculate the global mean delay across all messages.

    Args:
        df: DataFrame with message data

    Returns:
        Global mean delay
    """
    return df['delay_ms'].mean()


def generate_topology_links(nodes: Set[str], pairwise_delays: Dict[tuple, float],
                            global_mean_delay: float, bidirectional: bool = True) -> List[Dict]:
    """
    Generate topology links with end-to-end delays.

    Args:
        nodes: Set of unique nodes
        pairwise_delays: Dictionary of measured delays between node pairs
        global_mean_delay: Global mean delay to use when no specific delay is measured
        bidirectional: Whether to create bidirectional links

    Returns:
        List of link dictionaries
    """
    links = []
    processed_pairs = set()

    for node_a in nodes:
        for node_b in nodes:
            if node_a == node_b:
                continue

            # Skip if we've already processed this pair (for bidirectional links)
            if bidirectional and (node_b, node_a) in processed_pairs:
                continue

            # Check if we have measured delay data for this direction
            delay_a_to_b = pairwise_delays.get((node_a, node_b))
            delay_b_to_a = pairwise_delays.get((node_b, node_a))

            # Determine the delay to use
            if delay_a_to_b is not None and delay_b_to_a is not None:
                # Use average of both directions
                delay = (delay_a_to_b + delay_b_to_a) / 2
                source = "measured (both directions)"
            elif delay_a_to_b is not None:
                delay = delay_a_to_b
                source = "measured (A->B)"
            elif delay_b_to_a is not None:
                delay = delay_b_to_a
                source = "measured (B->A)"
            else:
                # Use global mean delay
                delay = global_mean_delay
                source = "global mean"

            link = {
                "node_a": node_a,
                "node_b": node_b,
                "end-to-end-delay_ms": round(delay, 2)
            }

            links.append(link)
            print(f"  Link {node_a} <-> {node_b}: {delay:.2f} ms ({source})")

            if bidirectional:
                processed_pairs.add((node_a, node_b))

    return links


def create_topology_dict(nodes: Set[str], links: List[Dict]) -> Dict:
    """
    Create the final topology dictionary.

    Args:
        nodes: Set of unique nodes
        links: List of link dictionaries

    Returns:
        Complete topology dictionary
    """
    topology = {
        'nodes': [{'node_id': node} for node in sorted(nodes)],
        'links': links
    }

    return topology


def save_topology_json(topology: Dict, output_file: str) -> None:
    """
    Save topology to JSON file.

    Args:
        topology: Topology dictionary
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(topology, f, indent=2)


def print_summary(df: pd.DataFrame, topology: Dict, pairwise_delays: Dict[tuple, float],
                  global_mean_delay: float) -> None:
    """
    Print summary statistics.

    Args:
        df: Original DataFrame
        topology: Generated topology
        pairwise_delays: Measured pairwise delays
        global_mean_delay: Global mean delay
    """
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total messages processed: {len(df)}")
    print(f"Unique nodes found: {len(topology['nodes'])}")
    print(f"Node IDs: {[node['node_id'] for node in topology['nodes']]}")
    print(f"Links generated: {len(topology['links'])}")
    print(f"Measured node pairs: {len(pairwise_delays)}")
    print(f"Global mean delay: {global_mean_delay:.2f} ms")

    # Delay statistics
    delays = df['delay_ms']
    print(f"\nDelay Statistics:")
    print(f"  Min delay: {delays.min():.2f} ms")
    print(f"  Max delay: {delays.max():.2f} ms")
    print(f"  Mean delay: {delays.mean():.2f} ms")
    print(f"  Median delay: {delays.median():.2f} ms")
    print(f"  Std deviation: {delays.std():.2f} ms")


def main():
    """Main function to orchestrate the delay graph generation."""
    for network in ['5g', 'ethernet', 'lte', 'lte450']:
        try:
            input_path = f'messages_detailed_{network}.csv'
            output_path = f'static_delay_graph_simbench_{network}.json'
            # Load and process data
            df = load_message_data(input_path)

            print(f"Loaded {len(df)} messages")
            print(f"Columns: {list(df.columns)}")

            # Extract unique nodes
            nodes = extract_unique_nodes(df)
            print(f"\nFound {len(nodes)} unique nodes: {sorted(nodes)}")
            nodes = [f'node{i}' for i in range(100)]

            # Calculate delays
            print(f"\nCalculating pairwise delays...")
            pairwise_delays = calculate_pairwise_delays(df)

            global_mean_delay = calculate_global_mean_delay(df)
            print(f"\nGlobal mean delay: {global_mean_delay:.2f} ms")

            # Generate topology
            print(f"\nGenerating topology links...")
            links = generate_topology_links(nodes, pairwise_delays, global_mean_delay, True)

            # Create final topology
            topology = create_topology_dict(nodes, links)

            # Save to JSON
            save_topology_json(topology, output_path)
            print(f"\nTopology saved to: {output_path}")

            # Print summary
            print_summary(df, topology, pairwise_delays, global_mean_delay)

        except Exception as e:
            print(f"Error: {e}")
            return 1

    return 0


if __name__ == '__main__':
    main()
