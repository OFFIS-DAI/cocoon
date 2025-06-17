import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend


def load_and_analyze_communication_data(csv_file):
    """
    Load communication data and analyze delay times and re-planning completion.

    Args:
        csv_file (str): Path to the CSV file containing communication data
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert time columns to seconds for easier analysis
    df['time_send_s'] = df['time_send_ms'] / 1000
    df['time_receive_s'] = df['time_receive_ms'] / 1000
    df['delay_s'] = df['delay_ms'] / 1000

    return df


def plot_delay_analysis(df):
    """
    Create visualizations of delay times using seaborn.
    """
    # Set seaborn style
    sns.set_palette("Set3")
    plt.rcParams.update({'font.size': 25})  # Increase base font size

    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    # 4. Violin plot of delays
    sns.violinplot(data=df, y='delay_ms', ax=axes)
    axes.set_ylabel('Delay (ms)')

    plt.tight_layout()

    # Save the plot
    plt.savefig('delay_analysis.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    print("Delay analysis plot saved as 'delay_analysis.png'")


def print_delay_statistics(df):
    """
    Print summary statistics about delays.
    """
    print("=== Delay Analysis Summary ===\n")

    print(f"Total messages: {len(df)}")
    print(f"Time period: {df['time_send_s'].min():.1f}s to {df['time_send_s'].max():.1f}s")
    print(f"Duration: {(df['time_send_s'].max() - df['time_send_s'].min()) / 60:.1f} minutes\n")

    print("Delay Statistics:")
    print(f"  Mean delay: {df['delay_ms'].mean():.2f} ms")
    print(f"  Median delay: {df['delay_ms'].median():.2f} ms")
    print(f"  Min delay: {df['delay_ms'].min():.2f} ms")
    print(f"  Max delay: {df['delay_ms'].max():.2f} ms")
    print(f"  Std deviation: {df['delay_ms'].std():.2f} ms")
    print(f"  95th percentile: {df['delay_ms'].quantile(0.95):.2f} ms")
    print(f"  99th percentile: {df['delay_ms'].quantile(0.99):.2f} ms")


def plot_termination_analysis(max_termination, x_to_duration):
    """
    Plot termination times by x value with max termination indicator.
    """

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(x_to_duration)

    # Convert termination times to minutes for better readability
    df['termination_min'] = df['termination'] / 60000  # Convert ms to minutes
    max_termination_min = max_termination / 60000

    # Set style and font size
    plt.rcParams.update({'font.size': 25})

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot the data points
    sns.scatterplot(data=df, x='x', y='termination_min', s=200, color='steelblue', ax=ax)

    # Connect points with lines
    ax.plot(df['x'], df['termination_min'], 'o-', color='slategrey', linewidth=2, markersize=10)

    # Add horizontal line for max termination
    ax.axhline(y=max_termination_min, color='crimson', linestyle='--', linewidth=2,
               label=f'Deadline ({max_termination_min:.1f} min)', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('x-value (minutes)', fontsize=25)
    ax.set_ylabel('Termination (minutes)', fontsize=25)

    # Set x-axis to log scale for better visualization
    # .set_xscale('log')

    # Add legend
    ax.legend(loc='upper right')

    # Set y-axis limits to show some margin
    y_min = df['termination_min'].min() * 0.98
    y_max = max(df['termination_min'].max(), max_termination_min) * 1.02
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save the plot
    plt.savefig('termination_analysis.pdf', dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

    print("Termination analysis plot saved as 'termination_analysis.png'")


def load_and_process_csv(csv_file, agent_count):
    """
    Load and process a single CSV file with agent count information.

    Args:
        csv_file (str): Path to the CSV file
        agent_count (int): Number of agents in this simulation

    Returns:
        pd.DataFrame: Processed dataframe with message types and agent count
    """
    df = pd.read_csv(csv_file)

    # Convert time columns
    df['time_send_s'] = df['time_send_ms'] / 1000
    df['time_receive_s'] = df['time_receive_ms'] / 1000
    df['time_send_min'] = df['time_send_s'] / 60

    # Extract message type from msg_id
    df['message_type'] = df['msg_id'].str.extract(r'[^_]+_([^_]+(?:_[^_]+)*?)(?:_\d+)?$')[0]
    df['message_type'] = df['message_type'].str.replace('_', ' ')

    # Add agent count information
    df['agent_count'] = agent_count
    df['scenario'] = f'{agent_count} agents'

    df = df[(df['time_send_min'] >= 10) & (df['time_send_min'] <= 20)]

    return df


def plot_message_types_comparison(csv_files, agent_counts, output_filename='message_types_comparison.png'):
    """
    Plot message types over time for multiple CSV files with different agent counts.

    Args:
        csv_files (list): List of paths to CSV files
        agent_counts (list): List of agent counts corresponding to each CSV file
        output_filename (str): Name of the output file
    """
    # Load and combine all data
    all_data = []

    for csv_file, agent_count in zip(csv_files, agent_counts):
        try:
            df = load_and_process_csv(csv_file, agent_count)
            all_data.append(df)
            print(f"Loaded {len(df)} messages for {agent_count} agents scenario")
        except FileNotFoundError:
            print(f"Warning: Could not find file {csv_file}")
            continue
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            continue

    if not all_data:
        print("No data loaded successfully")
        return

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Set style and font size
    plt.rcParams.update({'font.size': 16})

    # Create subplots - one for each agent count
    n_scenarios = len(all_data)
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(14, 6 * n_scenarios))

    # Handle case of single subplot
    if n_scenarios == 1:
        axes = [axes]

    # Get unique message types and assign consistent colors
    all_message_types = combined_df['message_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_message_types)))
    color_map = dict(zip(all_message_types, colors))

    # Plot each scenario
    for i, (df, agent_count) in enumerate(zip(all_data, agent_counts)):
        ax = axes[i]

        # Create timeline plot
        for msg_type in df['message_type'].unique():
            type_data = df[df['message_type'] == msg_type]
            ax.scatter(type_data['time_send_min'], [msg_type] * len(type_data),
                       c=[color_map[msg_type]], label=msg_type, alpha=0.7, s=50)

        # Add vertical line at minute 15 with deadline label
        ax.axvline(x=15, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(15.1, ax.get_ylim()[1] * 0.95, 'deadline', rotation=0,
                fontsize=14, color='red', fontweight='bold')

        ax.set_title(f'Message Types Over Time - {agent_count} Agents',
                     fontsize=18, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=16)
        ax.set_ylabel('Message Type', fontsize=16)
        ax.set_xlim(10, 20)  # Set x-axis limits to 10-20 minutes
        ax.tick_params(left=False, labelleft=True, labelsize=14)
        ax.grid(True, alpha=0.3)

        # Add legend only for the first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved as '{output_filename}'")


def main():
    """
    Main function to run the analysis.
    """
    # File path - adjust as needed
    csv_file = "results/messages_detailed-ten-none-one_hour-deer_use_case-simbench_lte450_x0_5.csv"

    try:
        # Load and analyze data
        print("Loading communication data...")
        df = load_and_analyze_communication_data(csv_file)

        # Print delay statistics
        print_delay_statistics(df)

        # Create delay visualizations
        print("Creating delay visualizations...")
        plot_delay_analysis(df)

        print("Analysis complete!")

    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

    max_termination = 900000

    x_to_duration = [
        {
            'x': 0.5,
            'termination': 870678
        },
        {
            'x': 0.1,
            'termination': 894658
        },
        {
            'x': 0.0,
            'termination': 897668
        }
    ]

    # Create visualization
    print("Creating termination time visualization...")
    plot_termination_analysis(max_termination, x_to_duration)

    csv_files = [
        "results/messages_static_graph-ten-none-one_hour-deer_use_case-simbench_lte450.csv",
        "results/messages_static_graph-hundred-none-one_hour-deer_use_case-simbench_lte450.csv",
        "results/messages_static_graph-thousand-none-one_hour-deer_use_case-simbench_lte450.csv"
    ]
    agent_counts = [10, 100, 1000]

    plot_message_types_comparison(csv_files, agent_counts)


if __name__ == "__main__":
    main()
