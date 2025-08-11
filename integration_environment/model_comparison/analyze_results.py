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

from integration_environment.scenario_configuration import *


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    scenario_config: ScenarioConfiguration
    model_type: ModelType

    # accuracy metrics
    nrmse_mean: Optional[float]
    nrmse_std: Optional[float]
    mean_in_sigma_interval: Optional[float]

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


def find_matching_detailed_simulations(config: ScenarioConfiguration, detailed_results: Dict[str, pd.DataFrame]) -> \
        List[pd.DataFrame]:
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
    matching_results = []
    run = 0
    while True:
        detailed_config.run = run
        detailed_scenario_id = detailed_config.scenario_id
        detailed_result = detailed_results.get(detailed_scenario_id)
        if detailed_result is not None:
            matching_results.append(detailed_result)
            run += 1
        else:
            return matching_results


def calculate_metrics_grouped(detailed_dfs: List[pd.DataFrame], model_dfs: List[pd.DataFrame]) -> Tuple[
    float, float, float]:
    if len(detailed_dfs) != len(model_dfs):
        raise ValueError(f"Mismatch in number of runs: {len(detailed_dfs)} detailed vs {len(model_dfs)} model")

    # Collect delays for each message across all runs
    detailed_delays_by_message = {}
    model_delays_by_message = {}

    for detailed_df, model_df in zip(detailed_dfs, model_dfs):
        # Process detailed simulation
        detailed_indexed = detailed_df.set_index(['msg_id', 'sender', 'receiver'])['delay_ms']
        for msg_key, delay in detailed_indexed.items():
            if msg_key not in detailed_delays_by_message:
                detailed_delays_by_message[msg_key] = []
            detailed_delays_by_message[msg_key].append(delay)

        # Process model simulation
        model_indexed = model_df.set_index(['msg_id', 'sender', 'receiver'])['delay_ms']
        for msg_key, delay in model_indexed.items():
            if msg_key not in model_delays_by_message:
                model_delays_by_message[msg_key] = []
            model_delays_by_message[msg_key].append(delay)

    # Find common messages across both detailed and model simulations
    common_messages = set(detailed_delays_by_message.keys()).intersection(set(model_delays_by_message.keys()))

    if len(common_messages) == 0:
        raise ValueError("No common messages found between detailed and model simulations across all runs")

    # Calculate mean delay for each message across runs
    detailed_mean_delays = []
    model_mean_delays = []

    detailed_std_delays = []
    model_std_delays = []

    means_in_sigma_interval = []

    for msg_key in common_messages:
        # calculate mean value of same messages
        detailed_mean = np.mean(detailed_delays_by_message[msg_key])
        model_mean = np.mean(model_delays_by_message[msg_key])
        detailed_mean_delays.append(detailed_mean)
        model_mean_delays.append(model_mean)

        # calculate std value of same messages
        detailed_std = np.std(detailed_delays_by_message[msg_key])
        model_std = np.std(model_delays_by_message[msg_key])
        detailed_std_delays.append(detailed_std)
        model_std_delays.append(model_std)

        # calculate the one-sigma-interval
        interval_lower = detailed_mean - abs(detailed_std)
        interval_upper = detailed_mean + abs(detailed_std)
        messages_in_interval = [interval_lower <= m <= interval_upper for m in model_delays_by_message[msg_key]]
        mean_in_interval = np.mean(messages_in_interval)
        means_in_sigma_interval.append(mean_in_interval)

    # Convert to numpy arrays for calculations
    detailed_mean_delays = np.array(detailed_mean_delays)
    model_mean_delays = np.array(model_mean_delays)
    detailed_std_delays = np.array(detailed_std_delays)
    model_std_delays = np.array(model_std_delays)

    # Calculate RMSE and MAE between the mean delays
    differences = model_mean_delays - detailed_mean_delays
    rmse_means = np.sqrt(np.mean(differences ** 2))
    nrmse_means = rmse_means / np.mean(detailed_mean_delays)

    # Calculate RMSE and MAE between the mean delays
    differences_std = model_std_delays - detailed_std_delays
    rmse_std = np.sqrt(np.mean(differences_std ** 2))
    nrmse_std = rmse_std / np.mean(detailed_std_delays)

    return nrmse_means, nrmse_std, np.mean(means_in_sigma_interval)


def analyze_results(results_folder: str) -> List[EvaluationResult]:
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

    # Group scenarios by configuration (excluding run ID)
    scenario_groups = {}
    for scenario_id, model_df in all_results.items():
        config = ScenarioConfiguration.from_scenario_id(scenario_id)
        # Create a config key without the run ID
        config_without_run = ScenarioConfiguration(
            payload_size=config.payload_size,
            num_devices=config.num_devices,
            model_type=config.model_type,
            scenario_duration=config.scenario_duration,
            traffic_configuration=config.traffic_configuration,
            network_type=config.network_type,
            cluster_distance_threshold=config.cluster_distance_threshold,
            i_pupa=config.i_pupa,
            learning_rate_weighting=config.learning_rate_weighting,
            butterfly_threshold_value=config.butterfly_threshold_value,
            substitution_priority=config.substitution_priority,
            run=None  # Exclude run from grouping
        )

        # Use scenario_id without run for grouping
        base_scenario_id = config_without_run.scenario_id
        if base_scenario_id not in scenario_groups:
            scenario_groups[base_scenario_id] = {
                'config': config_without_run,
                'scenario_ids': [],
                'dataframes': [],
                'execution_times': [],
                'substitution_data': []
            }

        scenario_groups[base_scenario_id]['scenario_ids'].append(scenario_id)
        scenario_groups[base_scenario_id]['dataframes'].append(model_df)
        scenario_groups[base_scenario_id]['execution_times'].append(execution_runtime_data.get(scenario_id, 0.0))
        scenario_groups[base_scenario_id]['substitution_data'].append({
            'substitution_occurred': substitution_occurred_data.get(scenario_id, False),
            'substitution_message_index': substitution_message_index_data.get(scenario_id, None)
        })

    # Process grouped simulations
    evaluation_results = []

    for base_scenario_id, group_data in scenario_groups.items():
        try:
            config = group_data['config']
            dataframes = group_data['dataframes']
            execution_times = group_data['execution_times']
            substitution_data = group_data['substitution_data']

            if config.model_type == ModelType.detailed:
                mean_execution_time = float(np.mean(execution_times))

                # For substitution data, use the first run's data (or aggregate as needed)
                substitution_occurred = any(sub['substitution_occurred'] for sub in substitution_data)
                substitution_message_index = next((sub['substitution_message_index'] for sub in substitution_data if
                                                   sub['substitution_message_index'] is not None), None)

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    nrmse_mean=None,  # No RMSE for baseline
                    nrmse_std=None,  # No MAE for baseline
                    mean_in_sigma_interval=None,
                    execution_time_s=mean_execution_time,
                    substitution_occurred=substitution_occurred,
                    substitution_message_index=substitution_message_index,
                )

            else:
                # For non-detailed simulations, find matching detailed simulations and calculate accuracy metrics
                detailed_dfs = find_matching_detailed_simulations(config, detailed_results)
                if len(detailed_dfs) == 0:
                    print(f"Warning: No matching detailed simulation found for {base_scenario_id}")
                    continue

                # Calculate overall metrics across all runs
                nrmse_means, nrmse_std, mean_in_one_sigma_interval = calculate_metrics_grouped(detailed_dfs, dataframes)

                mean_execution_time = float(np.mean(execution_times))

                # For substitution data, use aggregated information
                substitution_occurred = any(sub['substitution_occurred'] for sub in substitution_data)
                substitution_message_index = next((sub['substitution_message_index'] for sub in substitution_data if
                                                   sub['substitution_message_index'] is not None), None)

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    nrmse_mean=nrmse_means,
                    nrmse_std=nrmse_std,
                    mean_in_sigma_interval=mean_in_one_sigma_interval,
                    execution_time_s=mean_execution_time,
                    substitution_occurred=substitution_occurred,
                    substitution_message_index=substitution_message_index,
                )

            evaluation_results.append(result)

        except Exception as e:
            print(f"Error processing {base_scenario_id}: {e}")
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
            'nrmse_mean': result.nrmse_mean,
            'nrmse_std': result.nrmse_std,
            'mean_in_one_sigma_interval': result.mean_in_sigma_interval,

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
    evaluation_results = analyze_results(results_folder)

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
