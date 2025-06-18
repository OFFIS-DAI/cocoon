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
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from integration_environment.scenario_configuration import (
    ScenarioConfiguration,
    ModelType,
    NetworkModelType,
    PayloadSizeConfig,
    ScenarioDuration,
    NumDevices,
    TrafficConfig
)


@dataclass
class PerformanceMetrics:
    """Container for runtime and memory performance metrics."""
    execution_time_s: float
    memory_peak_mb: float
    memory_avg_mb: float
    memory_change_mb: float


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    scenario_config: ScenarioConfiguration
    model_type: ModelType
    rmse: Optional[float]  # None for detailed simulations (baseline)
    mae: Optional[float]   # None for detailed simulations (baseline)
    num_messages: int
    mean_delay_detailed: Optional[float]  # None for non-detailed simulations
    mean_delay_model: float
    performance_metrics: Optional[PerformanceMetrics] = None


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

        # Validate required columns
        required_columns = ['msg_id', 'sender', 'receiver', 'delay_ms']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: File {file_path} missing required columns: {missing_columns}")
            return None

        return df
    except Exception as e:
        return None


def load_performance_data(file_path: Path) -> Optional[PerformanceMetrics]:
    """Load performance statistics from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        return PerformanceMetrics(
            execution_time_s=data.get('execution_run_time_s', 0.0),
            memory_peak_mb=data.get('memory_statistics', {}).get('peak_memory_mb', 0.0),
            memory_avg_mb=data.get('memory_statistics', {}).get('avg_memory_mb', 0.0),
            memory_change_mb=data.get('memory_statistics', {}).get('memory_change_mb', 0.0)
        )
    except Exception as e:
        print(f"Warning: Could not load performance data from {file_path}: {e}")
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

    print(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files in {results_folder}")

    # Load all simulation data and performance data
    all_results = {}
    detailed_results = {}
    performance_data = {}

    # Load CSV files (message data)
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

    # Load JSON files (performance data)
    for json_file in json_files:
        config = parse_filename_to_config(json_file.name)
        if config is None:
            continue

        perf_data = load_performance_data(json_file)
        if perf_data is None:
            continue

        scenario_id = config.scenario_id
        performance_data[scenario_id] = perf_data

    print(f"Successfully loaded {len(all_results)} simulation results")
    print(f"Found {len(detailed_results)} detailed simulation baselines")
    print(f"Loaded {len(performance_data)} performance statistics")

    # Process all simulations (including detailed ones)
    evaluation_results = []

    for scenario_id, model_df in all_results.items():
        try:
            config = ScenarioConfiguration.from_scenario_id(scenario_id)

            # Get performance metrics if available
            perf_metrics = performance_data.get(scenario_id)

            if config.model_type == ModelType.detailed:
                # For detailed simulations, we don't calculate RMSE/MAE (they are the baseline)
                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    rmse=None,  # No RMSE for baseline
                    mae=None,   # No MAE for baseline
                    num_messages=len(model_df),
                    mean_delay_detailed=None,  # Not applicable
                    mean_delay_model=model_df['delay_ms'].mean(),
                    performance_metrics=perf_metrics
                )

                perf_str = ""
                if perf_metrics:
                    perf_str = f", Runtime={perf_metrics.execution_time_s:.2f}s, Memory={perf_metrics.memory_peak_mb:.1f}MB"

                print(f"✓ {scenario_id} (BASELINE): Messages={len(model_df)}, Mean Delay={model_df['delay_ms'].mean():.2f}ms{perf_str}")

            else:
                # For non-detailed simulations, calculate accuracy metrics
                detailed_df = find_matching_detailed_simulation(config, detailed_results)
                if detailed_df is None:
                    print(f"Warning: No matching detailed simulation found for {scenario_id}")
                    continue

                # Calculate metrics
                rmse, mae, num_messages = calculate_metrics(detailed_df, model_df)

                # Calculate mean delays for reference
                mean_delay_detailed = detailed_df['delay_ms'].mean()
                mean_delay_model = model_df['delay_ms'].mean()

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    rmse=rmse,
                    mae=mae,
                    num_messages=num_messages,
                    mean_delay_detailed=mean_delay_detailed,
                    mean_delay_model=mean_delay_model,
                    performance_metrics=perf_metrics
                )

                perf_str = ""
                if perf_metrics:
                    perf_str = f", Runtime={perf_metrics.execution_time_s:.2f}s, Memory={perf_metrics.memory_peak_mb:.1f}MB"

                print(f"✓ {scenario_id}: RMSE={rmse:.2f}ms, MAE={mae:.2f}ms, Messages={num_messages}{perf_str}")

            evaluation_results.append(result)

        except Exception as e:
            print(f"Error processing {scenario_id}: {e}")
            continue

    # Save detailed results if requested
    if output_file and evaluation_results:
        save_detailed_results(evaluation_results, output_file)

    return evaluation_results


def save_detailed_results(results: List[EvaluationResult], output_file: str):
    """Save detailed evaluation results to CSV."""
    data = []
    for result in results:
        row = {
            'scenario_id': result.scenario_config.scenario_id,
            'model_type': result.model_type.name,
            'network_type': result.scenario_config.network_type.name,
            'num_devices': result.scenario_config.num_devices.name,
            'payload_size': result.scenario_config.payload_size.name,
            'scenario_duration': result.scenario_config.scenario_duration.name,
            'traffic_config': result.scenario_config.traffic_configuration.name,
            'rmse_ms': result.rmse,
            'mae_ms': result.mae,
            'num_messages': result.num_messages,
            'mean_delay_detailed_ms': result.mean_delay_detailed,
            'mean_delay_model_ms': result.mean_delay_model,
            'relative_error_percent': ((result.mean_delay_model - result.mean_delay_detailed) / result.mean_delay_detailed * 100) if result.mean_delay_detailed and result.mean_delay_detailed > 0 else None
        }

        # Add performance metrics if available
        if result.performance_metrics:
            row.update({
                'execution_time_s': result.performance_metrics.execution_time_s,
                'memory_peak_mb': result.performance_metrics.memory_peak_mb,
                'memory_avg_mb': result.performance_metrics.memory_avg_mb,
                'memory_change_mb': result.performance_metrics.memory_change_mb
            })
        else:
            row.update({
                'execution_time_s': None,
                'memory_peak_mb': None,
                'memory_avg_mb': None,
                'memory_change_mb': None
            })

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")


def print_performance_summary(results: List[EvaluationResult]):
    """Print performance comparison summary."""
    print(f"\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Get results with performance data
    perf_results = [r for r in results if r.performance_metrics is not None]

    if not perf_results:
        print("No performance data available for analysis.")
        return

    print(f"Performance data available for {len(perf_results)} out of {len(results)} scenarios")

    # Group by model type for performance comparison
    by_model = {}
    for result in perf_results:
        model_name = result.model_type.name
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(result)

    print(f"\nRUNTIME COMPARISON BY MODEL TYPE")
    print("-" * 50)

    for model_name, model_results in by_model.items():
        runtimes = [r.performance_metrics.execution_time_s for r in model_results]

        print(f"\n{model_name.upper()}:")
        print(f"  Scenarios: {len(model_results)}")
        print(f"  Runtime (s): Mean={np.mean(runtimes):.3f}, Std={np.std(runtimes):.3f}")
        print(f"              Min={np.min(runtimes):.3f}, Max={np.max(runtimes):.3f}")

    print(f"\nMEMORY USAGE COMPARISON BY MODEL TYPE")
    print("-" * 50)

    for model_name, model_results in by_model.items():
        peak_memory = [r.performance_metrics.memory_peak_mb for r in model_results]
        avg_memory = [r.performance_metrics.memory_avg_mb for r in model_results]

        print(f"\n{model_name.upper()}:")
        print(f"  Peak Memory (MB): Mean={np.mean(peak_memory):.1f}, Std={np.std(peak_memory):.1f}")
        print(f"                    Min={np.min(peak_memory):.1f}, Max={np.max(peak_memory):.1f}")
        print(f"  Avg Memory (MB):  Mean={np.mean(avg_memory):.1f}, Std={np.std(avg_memory):.1f}")

    # Performance vs Accuracy Trade-off Analysis (excluding detailed simulations)
    print(f"\nPERFORMANCE vs ACCURACY TRADE-OFF")
    print("-" * 50)

    # Get baseline performance from detailed simulations
    detailed_results = [r for r in perf_results if r.model_type == ModelType.detailed]
    if detailed_results:
        detailed_runtime = np.mean([r.performance_metrics.execution_time_s for r in detailed_results])
        detailed_memory = np.mean([r.performance_metrics.memory_peak_mb for r in detailed_results])
        print(f"\nDETAILED (BASELINE):")
        print(f"  Avg Runtime: {detailed_runtime:.3f}s")
        print(f"  Avg Peak Memory: {detailed_memory:.1f}MB")
        print(f"  Accuracy: Perfect (baseline)")

    for model_name, model_results in by_model.items():
        if len(model_results) == 0 or model_name == 'detailed':
            continue

        # Only calculate accuracy metrics for non-detailed models
        accuracy_results = [r for r in model_results if r.rmse is not None and r.mae is not None]
        if not accuracy_results:
            continue

        avg_runtime = np.mean([r.performance_metrics.execution_time_s for r in model_results])
        avg_rmse = np.mean([r.rmse for r in accuracy_results])
        avg_mae = np.mean([r.mae for r in accuracy_results])
        avg_memory = np.mean([r.performance_metrics.memory_peak_mb for r in model_results])

        # Calculate speedup compared to detailed simulation
        speedup = detailed_runtime / avg_runtime if detailed_results and avg_runtime > 0 else None
        memory_ratio = avg_memory / detailed_memory if detailed_results and detailed_memory > 0 else None

        print(f"\n{model_name.upper()}:")
        print(f"  Avg Runtime: {avg_runtime:.3f}s" + (f" ({speedup:.1f}x faster)" if speedup else ""))
        print(f"  Avg Peak Memory: {avg_memory:.1f}MB" + (f" ({memory_ratio:.2f}x baseline)" if memory_ratio else ""))
        print(f"  Avg RMSE: {avg_rmse:.2f}ms")
        print(f"  Avg MAE: {avg_mae:.2f}ms")


def print_summary(results: List[EvaluationResult]):
    """Print summary statistics."""
    if not results:
        print("No results to summarize.")
        return

    # Separate detailed and non-detailed results for accuracy analysis
    accuracy_results = [r for r in results if r.rmse is not None and r.mae is not None]
    detailed_results = [r for r in results if r.model_type == ModelType.detailed]

    print(f"\n" + "=" * 80)
    print("ACCURACY EVALUATION SUMMARY")
    print("=" * 80)

    if detailed_results:
        print(f"\nDETAILED SIMULATIONS (BASELINE):")
        print(f"  Scenarios: {len(detailed_results)}")
        print(f"  Used as ground truth for accuracy evaluation")

    if accuracy_results:
        # Group by model type
        by_model = {}
        for result in accuracy_results:
            model_name = result.model_type.name
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(result)

        for model_name, model_results in by_model.items():
            print(f"\nModel Type: {model_name}")
            print("-" * 40)

            rmse_values = [r.rmse for r in model_results]
            mae_values = [r.mae for r in model_results]

            print(f"  Scenarios evaluated: {len(model_results)}")
            print(
                f"  RMSE (ms): Mean={np.mean(rmse_values):.2f}, Std={np.std(rmse_values):.2f}, Min={np.min(rmse_values):.2f}, Max={np.max(rmse_values):.2f}")
            print(
                f"  MAE (ms):  Mean={np.mean(mae_values):.2f}, Std={np.std(mae_values):.2f}, Min={np.min(mae_values):.2f}, Max={np.max(mae_values):.2f}")

    # Additional analysis by network type and other factors
    print(f"\n" + "=" * 80)
    print("ANALYSIS BY NETWORK TYPE")
    print("=" * 80)

    if accuracy_results:
        by_network = {}
        for result in accuracy_results:
            network_name = result.scenario_config.network_type.name
            if network_name not in by_network:
                by_network[network_name] = []
            by_network[network_name].append(result)

        for network_name, network_results in by_network.items():
            print(f"\nNetwork Type: {network_name}")
            print("-" * 40)

            rmse_values = [r.rmse for r in network_results]
            mae_values = [r.mae for r in network_results]

            print(f"  Scenarios evaluated: {len(network_results)}")
            print(f"  RMSE (ms): Mean={np.mean(rmse_values):.2f}, Std={np.std(rmse_values):.2f}")
            print(f"  MAE (ms):  Mean={np.mean(mae_values):.2f}, Std={np.std(mae_values):.2f}")

    print(f"\nTotal scenarios: {len(results)} ({len(detailed_results)} detailed, {len(accuracy_results)} accuracy-evaluated)")

    # Add performance summary
    print_performance_summary(results)


def main():
    """Main function."""

    try:
        # Analyze results
        results = analyze_results('results', 'analysis_results.csv')

        # Print summary
        print_summary(results)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())