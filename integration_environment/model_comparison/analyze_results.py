#!/usr/bin/env python3
"""
Communication Model Evaluation Script

This script analyzes simulation results from different communication modeling approaches
and calculates RMSE and MAE compared to the detailed simulation baseline, plus runtime
and memory usage comparisons for all models including detailed simulations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

from integration_environment.scenario_configuration import *


@dataclass
class SubstitutionInfo:
    """Container for meta-model substitution information."""
    substitution_occurred: bool
    substitution_message_index: Optional[int] = None


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    scenario_config: ScenarioConfiguration
    model_type: ModelType

    # accuracy metrics
    rmse: Optional[float]  # None for detailed simulations (baseline)
    mae: Optional[float]  # None for detailed simulations (baseline)
    num_messages: int
    mean_delay_detailed: Optional[float]  # None for non-detailed simulations
    mean_delay_model: float
    std_delay_detailed: Optional[float]
    std_delay_model: Optional[float]

    # performance metric
    execution_time_s: float

    # substitution info (for meta-model)
    substitution_occurred: bool
    substitution_message_index: Optional[int] = None


def parse_filename_to_config(filename: str) -> Optional[ScenarioConfiguration]:
    """
    Parse a result filename to extract scenario configuration.

    Expected format: messages_{scenario_id}.csv or statistics_{scenario_id}.json
    """
    try:
        # Handle both messages_ and statistics_ prefixes
        if filename.startswith('messages_'):
            scenario_id = filename[9:]  # Remove 'messages_' prefix
        elif filename.startswith('statistics_'):
            scenario_id = filename[11:]  # Remove 'statistics_' prefix
        else:
            return None

        # Remove file extension
        if scenario_id.endswith('.csv'):
            scenario_id = scenario_id[:-4]
        if scenario_id.endswith('.json'):
            scenario_id = scenario_id[:-5]

        return ScenarioConfiguration.from_scenario_id(scenario_id)
    except Exception as e:
        print(f"Warning: Could not parse filename '{filename}': {e}")
        return None


def load_simulation_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load simulation data from CSV file."""
    try:
        df = pd.read_csv(file_path)

        # Check if this is a detailed meta-model file (has prediction columns)
        if 'actual_delay_ms' in df.columns and 'predicted_delay_ms' in df.columns:
            return None
        else:
            # Standard message file
            # Validate required columns
            required_columns = ['msg_id', 'sender', 'receiver', 'delay_ms']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Warning: File {file_path} missing required columns: {missing_columns}")
                return None

        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_matching_detailed_simulation(config: ScenarioConfiguration, detailed_results: Dict[str, pd.DataFrame]) -> \
        Optional[pd.DataFrame]:
    """Find the detailed simulation result that matches the given configuration."""
    # Create a detailed version of the config
    detailed_config = ScenarioConfiguration(
        payload_size=config.payload_size,
        num_devices=config.num_devices,
        model_type=ModelType.detailed,
        scenario_duration=config.scenario_duration,
        traffic_configuration=config.traffic_configuration,
        network_type=config.network_type
    )

    detailed_scenario_id = detailed_config.scenario_id
    return detailed_results.get(detailed_scenario_id)


def calculate_metrics(detailed_df: pd.DataFrame, model_df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    Calculate RMSE and MAE between detailed and model simulation results.

    Returns:
        Tuple of (RMSE, MAE, number_of_messages)
    """
    # Align messages by msg_id, sender, receiver for comparison
    detailed_delays = detailed_df.set_index(['msg_id', 'sender', 'receiver'])['delay_ms']
    model_delays = model_df.set_index(['msg_id', 'sender', 'receiver'])['delay_ms']

    # Find common messages
    common_indices = detailed_delays.index.intersection(model_delays.index)

    if len(common_indices) == 0:
        raise ValueError("No common messages found between detailed and model simulations")

    detailed_common = detailed_delays.loc[common_indices]
    model_common = model_delays.loc[common_indices]

    # Calculate metrics
    differences = model_common - detailed_common
    rmse = np.sqrt(np.mean(differences ** 2))
    mae = np.mean(np.abs(differences))

    return rmse, mae, len(common_indices)


def analyze_results(results_folder: str, output_file: Optional[str] = None) -> List[EvaluationResult]:
    """
    Analyze all simulation results in the given folder.

    Args:
        results_folder: Path to folder containing CSV result files and JSON statistics files
        output_file: Optional path to save detailed results CSV

    Returns:
        List of EvaluationResult objects
    """
    results_path = Path(results_folder)
    if not results_path.exists():
        raise ValueError(f"Results folder does not exist: {results_folder}")

    # Find all CSV and JSON files
    csv_files = list(results_path.glob("messages_*.csv"))
    json_files = list(results_path.glob("statistics_*.json"))

    print(
        f"Found {len(csv_files)} standard CSV files, and {len(json_files)} JSON files in {results_folder}")

    # Load all simulation data and performance data
    all_results = {}
    detailed_results = {}
    execution_runtime_data = {}
    substitution_occurred_data = {}
    substitution_message_index_data = {}

    # Load standard CSV files (message data)
    for csv_file in csv_files:
        config = parse_filename_to_config(csv_file.name)
        if config is None:
            continue

        df = load_simulation_data(csv_file)
        if df is None:
            continue

        scenario_id = config.scenario_id
        all_results[scenario_id] = df

        # Store detailed simulations separately
        if config.model_type == ModelType.detailed:
            detailed_results[scenario_id] = df

    # Load JSON files (performance and substitution data)
    for json_file in json_files:
        config = parse_filename_to_config(json_file.name)
        if config is None:
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        scenario_id = config.scenario_id
        # Load performance metrics
        execution_runtime_data[scenario_id] = data.get('execution_run_time_s', 0.0)

        # Load substitution info
        substitution_data = data.get('meta_model_substitution', {})
        substitution_occurred_data[scenario_id] = substitution_data.get('substitution_occurred', False)
        substitution_message_index_data[scenario_id] = substitution_data.get('substitution_message_index')

    # Process all simulations (including detailed ones)
    evaluation_results = []

    for scenario_id, model_df in all_results.items():
        try:
            config = ScenarioConfiguration.from_scenario_id(scenario_id)

            if config.model_type == ModelType.detailed:
                # For detailed simulations, we don't calculate RMSE/MAE (they are the baseline)
                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    rmse=None,  # No RMSE for baseline
                    mae=None,  # No MAE for baseline
                    num_messages=len(model_df),
                    mean_delay_detailed=None,  # Not applicable
                    mean_delay_model=model_df['delay_ms'].mean(),
                    std_delay_detailed=None,  # Not applicable
                    std_delay_model=model_df['delay_ms'].std(),
                    execution_time_s=execution_runtime_data[scenario_id],
                    substitution_occurred=substitution_occurred_data[scenario_id],
                    substitution_message_index=substitution_message_index_data[scenario_id],
                )

            else:
                # For non-detailed simulations, calculate accuracy metrics
                detailed_df = find_matching_detailed_simulation(config, detailed_results)
                if detailed_df is None:
                    print(f"Warning: No matching detailed simulation found for {scenario_id}")
                    continue

                # Calculate overall metrics
                rmse, mae, num_messages = calculate_metrics(detailed_df, model_df)

                # Calculate mean delays for reference
                mean_delay_detailed = detailed_df['delay_ms'].mean()
                mean_delay_model = model_df['delay_ms'].mean()

                std_delay_detailed = detailed_df['delay_ms'].std()
                std_delay_model = model_df['delay_ms'].std()

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    rmse=rmse,
                    mae=mae,
                    num_messages=num_messages,
                    mean_delay_detailed=mean_delay_detailed,
                    mean_delay_model=mean_delay_model,
                    std_delay_detailed=std_delay_detailed,
                    std_delay_model=std_delay_model,
                    execution_time_s=execution_runtime_data[scenario_id],
                    substitution_occurred=substitution_occurred_data[scenario_id],
                    substitution_message_index=substitution_message_index_data[scenario_id],
                )

            evaluation_results.append(result)

        except Exception as e:
            print(f"Error processing {scenario_id}: {e}")
            continue
    return evaluation_results


def save_evaluation_results_to_csv(
        evaluation_results: List[EvaluationResult],
        output_file: str,
        include_scenario_details: bool = True
) -> None:
    """
    Save evaluation results to CSV file.

    Args:
        evaluation_results: List of EvaluationResult objects
        output_file: Path to output CSV file
        include_scenario_details: Whether to include detailed scenario configuration columns
    """
    if not evaluation_results:
        print("Warning: No evaluation results to save")
        return

    # Convert results to list of dictionaries
    rows = []

    for result in evaluation_results:
        row = {
            # Basic identifiers
            'scenario_id': result.scenario_config.scenario_id,
            'model_type': result.model_type.value,

            # Accuracy metrics
            'rmse': result.rmse,
            'mae': result.mae,
            'num_messages': result.num_messages,

            # Delay statistics
            'mean_delay_detailed': result.mean_delay_detailed,
            'mean_delay_model': result.mean_delay_model,
            'std_delay_detailed': result.std_delay_detailed,
            'std_delay_model': result.std_delay_model,

            # Performance metrics
            'execution_time_s': result.execution_time_s,

            # Meta-model specific metrics
            'substitution_occurred': result.substitution_occurred,
            'substitution_message_index': result.substitution_message_index,
        }

        # Add scenario configuration details if requested
        if include_scenario_details:
            row.update({
                'payload_size': result.scenario_config.payload_size,
                'num_devices': result.scenario_config.num_devices,
                'scenario_duration': result.scenario_config.scenario_duration,
                'traffic_configuration': result.scenario_config.traffic_configuration,
                'network_type': result.scenario_config.network_type,

                # Factor values (raw)
                'cluster_distance_threshold': result.scenario_config.cluster_distance_threshold.value if result.scenario_config.cluster_distance_threshold else None,
                'batch_size_ipupa': result.scenario_config.i_pupa.value if result.scenario_config.i_pupa else None,
                'learning_rate_weighting': result.scenario_config.learning_rate_weighting.value if result.scenario_config.learning_rate_weighting else None,
                'butterfly_threshold_value': result.scenario_config.butterfly_threshold_value.value if result.scenario_config.butterfly_threshold_value else None,
                'substitution_priority': result.scenario_config.substitution_priority.name if result.scenario_config.substitution_priority else None,

                # Factor names (for analysis)
                'cluster_distance_threshold_name': result.scenario_config.cluster_distance_threshold.name if result.scenario_config.cluster_distance_threshold else None,
                'batch_size_ipupa_name': result.scenario_config.i_pupa.name if result.scenario_config.i_pupa else None,
                'learning_rate_weighting_name': result.scenario_config.learning_rate_weighting.name if result.scenario_config.learning_rate_weighting else None,
                'butterfly_threshold_value_name': result.scenario_config.butterfly_threshold_value.name if result.scenario_config.butterfly_threshold_value else None,

            })

        rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(rows)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(evaluation_results)} evaluation results to {output_file}")


def analyze_results_with_csv_export(results_folder: str, output_file: Optional[str] = None) -> List[EvaluationResult]:
    """
    Analyze all simulation results and automatically save to CSV.

    Args:
        results_folder: Path to folder containing result files
        output_file: Optional path to save detailed results CSV

    Returns:
        List of EvaluationResult objects
    """
    # Use the existing analyze_results function (assuming it's imported or defined)
    evaluation_results = analyze_results(results_folder, output_file)

    # Automatically save results to CSV
    if output_file:
        save_evaluation_results_to_csv(evaluation_results, output_file)

    return evaluation_results


# Example usage (add this to the end of your existing script):
if __name__ == "__main__":
    phase = 1

    # Create output directory
    Path('analysis_results').mkdir(exist_ok=True)

    # Analyze results and save to CSV
    results = analyze_results_with_csv_export(
        f'results/phase{phase}',
        f'analysis_results/aggregated_results{phase}.csv'
    )

    print(f"Analysis complete. Processed {len(results)} scenarios.")

    # Print some basic statistics
    detailed_count = sum(1 for r in results if r.model_type.value == 'detailed')
    model_count = len(results) - detailed_count

    print(f"- {detailed_count} detailed simulations (baseline)")
    print(f"- {model_count} model simulations")