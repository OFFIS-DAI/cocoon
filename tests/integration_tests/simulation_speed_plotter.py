#!/usr/bin/env python3
"""
Simulation Speed Analysis Plotter

This script automatically reads all time advancement CSV files from the 'results' folder
and creates plots showing simulation speed (simulation time / real time ratio) over time 
for different models.

Usage: python simulation_speed_plotter.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SimulationSpeedAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer.
        Automatically looks for CSV files with 'time_advancement' prefix in 'results' folder.
        """
        self.data_directory = Path("results")
        self.file_pattern = "time_advancement*.csv"
        self.data = {}

    def extract_model_name(self, filename: str) -> str:
        """
        Extract model name from filename.

        Args:
            filename: Name of the CSV file

        Returns:
            Extracted model name
        """
        if 'meta' in filename:
            return 'cocoon meta-model'
        if 'detailed' in filename:
            return 'detailed model'
        if 'ideal' in filename:
            return 'ideal communication'


    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all matching CSV files and their corresponding statistics files.

        Returns:
            Dictionary mapping model names to DataFrames
        """
        csv_files = list(self.data_directory.glob(self.file_pattern))

        if not csv_files:
            raise FileNotFoundError(f"No files found matching pattern '{self.file_pattern}' in {self.data_directory}")

        print(f"Found {len(csv_files)} CSV files:")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)

                # Validate required columns
                required_columns = ['real_timestamp', 'simulation_time_s', 'relative_real_time_s']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: {file_path.name} missing required columns. Skipping.")
                    continue

                # Extract model name
                model_name = self.extract_model_name(file_path.name)
                print(f"Debug: Processing file: {file_path.name}")
                print(f"Debug: Extracted model name: '{model_name}'")

                # Try to load corresponding statistics file
                stats_filename = file_path.name.replace('time_advancement_', 'statistics_').replace('csv', 'json')
                stats_path = self.data_directory / stats_filename

                substitution_info = None
                if stats_path.exists():
                    try:
                        import json
                        with open(stats_path, 'r') as f:
                            stats_data = json.load(f)

                        # Extract substitution information
                        if 'meta_model_substitution' in stats_data:
                            substitution_data = stats_data['meta_model_substitution']
                            if substitution_data.get('substitution_occurred', False):
                                substitution_info = {
                                    'substitution_time_s': substitution_data.get('substitution_time_s'),
                                    'confidence_score': substitution_data.get('confidence_score'),
                                    'substitution_message_index': substitution_data.get('substitution_message_index')
                                }
                                print(f"    Found substitution at time {substitution_info['substitution_time_s']:.2f}s")
                    except Exception as e:
                        print(f"    Warning: Could not load statistics from {stats_filename}: {e}")

                # Store data with substitution info
                self.data[model_name] = {
                    'dataframe': df,
                    'substitution_info': substitution_info
                }
                print(f"  - {file_path.name} -> Model: '{model_name}' ({len(df)} data points)")

            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

        if not self.data:
            raise ValueError("No valid data files were loaded")

        return self.data

    def calculate_simulation_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simulation speed (sim_time/real_time ratio) for a DataFrame.

        Args:
            df: DataFrame with time advancement data

        Returns:
            DataFrame with added simulation speed columns
        """
        df = df.copy().sort_values('relative_real_time_s')

        # Calculate time differences
        df['real_time_diff'] = df['relative_real_time_s'].diff()
        df['sim_time_diff'] = df['simulation_time_s'].diff()

        # Calculate instantaneous simulation speed (avoid division by zero)
        mask = df['real_time_diff'] > 0
        df.loc[mask, 'simulation_speed'] = df.loc[mask, 'sim_time_diff'] / df.loc[mask, 'real_time_diff']

        # Calculate cumulative simulation speed
        total_sim_time = df['simulation_time_s'] - df['simulation_time_s'].iloc[0]
        total_real_time = df['relative_real_time_s'] - df['relative_real_time_s'].iloc[0]

        mask_cumulative = total_real_time > 0
        df.loc[mask_cumulative, 'cumulative_simulation_speed'] = total_sim_time[mask_cumulative] / total_real_time[
            mask_cumulative]

        return df

    def create_plots(self, save_plots: bool = True, show_plots: bool = True):
        """
        Create a single plot showing Real Time vs Simulation Time with substitution markers.

        Args:
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")

        # Create single plot with better styling
        plt.figure(figsize=(12, 5))

        # Set a clean, modern style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Define a professional color palette
        colors = ['#6C8EBF', '#67AB9F', '#B5739D', '#C73E1D', '#1B998B']

        # Plot data for each model
        for i, (model_name, data_info) in enumerate(self.data.items()):
            df = data_info['dataframe']
            substitution_info = data_info['substitution_info']

            # Plot the main line with better styling
            edge_color = plt.cm.colors.rgb2hex([c * 0.6 for c in plt.cm.colors.to_rgb(colors[i % len(colors)])])

            plt.plot(df['relative_real_time_s'], df['simulation_time_s'],
                     label=model_name, linewidth=3, marker='o', markersize=10,
                     alpha=0.9, color=colors[i % len(colors)],
                     markeredgecolor=edge_color,  # Darker shade of same color
                     markeredgewidth=1.5)

            # Add substitution marker if available
            if substitution_info and substitution_info['substitution_time_s'] is not None:
                substitution_real_time = substitution_info['substitution_time_s']

                # Convert to relative time by subtracting the first timestamp
                first_timestamp = df['real_timestamp'].iloc[0]
                substitution_relative_time = substitution_real_time - first_timestamp

                # Find the closest simulation time point
                closest_idx = (df['relative_real_time_s'] - substitution_relative_time).abs().idxmin()
                substitution_sim_time = df.loc[closest_idx, 'simulation_time_s']

                # Add substitution marker with better styling
                plt.scatter(substitution_relative_time, substitution_sim_time,
                            s=600, marker='*', color='black', edgecolors='grey',
                            linewidth=2, zorder=10, label='Substitution', alpha=0.9)

        fontsize = 20

        # Clean up tick styling
        plt.tick_params(axis='both', labelsize=fontsize, length=6, width=1.2,
                        colors='#2F2F2F', direction='out')

        # Style the plot elements
        plt.xlabel('Real Time (seconds)', fontsize=fontsize, color='#2F2F2F', weight='semibold')
        plt.ylabel('Simulation Time (seconds)', fontsize=fontsize, color='#2F2F2F', weight='semibold')
        plt.title('Real Time vs Simulation Time', fontsize=fontsize, color='#2F2F2F',
                  fontweight='bold', pad=20)

        # Improve legend styling
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize - 2,
                            frameon=True, fancybox=True, shadow=True, borderpad=0.8)
        legend.get_frame().set_facecolor('#FFFFFF')
        legend.get_frame().set_edgecolor('#CCCCCC')
        legend.get_frame().set_linewidth(1)

        # Enhance grid styling
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#CCCCCC')

        # Set axis limits and styling
        plt.xlim(0, 50)
        ax = plt.gca()
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_color('#2F2F2F')
        ax.spines['bottom'].set_color('#2F2F2F')

        plt.tight_layout()

        if save_plots:
            output_file = self.data_directory / 'simulation_time_analysis.pdf'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
            print(f"Plot saved as: {output_file}")

        if show_plots:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot due to matplotlib backend issue: {e}")
                print("Plot has been saved successfully to file.")

        # Print summary
        self.print_summary_statistics()

    def print_summary_statistics(self):
        """
        Print summary statistics for all models.
        """
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        for model_name, data_info in self.data.items():
            df = data_info['dataframe']
            substitution_info = data_info['substitution_info']

            print(f"\nModel: {model_name}")
            print("-" * 40)

            # Basic statistics
            total_sim_time = df['simulation_time_s'].iloc[-1] - df['simulation_time_s'].iloc[0]
            total_real_time = df['relative_real_time_s'].iloc[-1] - df['relative_real_time_s'].iloc[0]
            overall_speed = total_sim_time / total_real_time if total_real_time > 0 else 0

            print(f"Total simulation time: {total_sim_time:.2f} seconds")
            print(f"Total real time: {total_real_time:.2f} seconds")
            print(f"Overall simulation speed: {overall_speed:.2f}x")

            # Substitution information
            if substitution_info:
                substitution_real_time = substitution_info['substitution_time_s']
                first_timestamp = df['real_timestamp'].iloc[0]
                substitution_relative_time = substitution_real_time - first_timestamp

                print(f"\nMeta-model substitution:")
                print(f"  - Occurred at: {substitution_relative_time:.2f}s relative time")
                print(f"  - Real timestamp: {substitution_real_time:.2f}")
                print(f"  - Confidence score: {substitution_info.get('confidence_score', 'N/A'):.3f}")
                print(f"  - Message index: {substitution_info.get('substitution_message_index', 'N/A')}")
            else:
                print("\nNo meta-model substitution detected")


def main():
    """Main function to run the simulation speed analysis."""

    try:
        # Initialize analyzer (automatically uses 'results' folder)
        analyzer = SimulationSpeedAnalyzer()

        # Check if results directory exists
        if not analyzer.data_directory.exists():
            print(f"Error: Results directory '{analyzer.data_directory}' not found!")
            print("Please make sure you have a 'results' folder in the current directory.")
            return 1

        # Load data
        print(f"Loading time advancement data from '{analyzer.data_directory}'...")
        analyzer.load_data()

        # Create plots
        print("Creating simulation speed plots...")
        analyzer.create_plots(save_plots=True, show_plots=True)

        print("Analysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())