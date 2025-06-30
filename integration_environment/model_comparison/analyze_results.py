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

from integration_environment.scenario_configuration import (
    ScenarioConfiguration,
    ModelType, ClusterDistanceThreshold
)


@dataclass
class PerformanceMetrics:
    """Container for runtime and memory performance metrics."""
    execution_time_s: float
    memory_peak_mb: float
    memory_avg_mb: float
    memory_change_mb: float


@dataclass
class SubstitutionInfo:
    """Container for meta-model substitution information."""
    substitution_occurred: bool
    substitution_time_s: Optional[float] = None
    substitution_message_index: Optional[int] = None
    substitution_runtime_s: Optional[float] = None
    confidence_score: Optional[float] = None
    additional_info: Optional[dict] = None


@dataclass
class MetaModelPredictionMetrics:
    """Container for meta-model prediction analysis."""
    prediction_rmse: float
    prediction_mae: float
    prediction_correlation: float
    num_predictions: int
    mean_actual_delay: float
    mean_predicted_delay: float
    prediction_bias: float  # mean(predicted - actual)
    before_substitution_prediction_rmse: Optional[float] = None
    before_substitution_prediction_mae: Optional[float] = None
    after_substitution_prediction_rmse: Optional[float] = None
    after_substitution_prediction_mae: Optional[float] = None


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    scenario_config: ScenarioConfiguration
    model_type: ModelType
    rmse: Optional[float]  # None for detailed simulations (baseline)
    mae: Optional[float]  # None for detailed simulations (baseline)
    num_messages: int
    mean_delay_detailed: Optional[float]  # None for non-detailed simulations
    mean_delay_model: float
    performance_metrics: Optional[PerformanceMetrics] = None
    substitution_info: Optional[SubstitutionInfo] = None
    # Before/after substitution metrics
    before_substitution_rmse: Optional[float] = None
    before_substitution_mae: Optional[float] = None
    before_substitution_messages: Optional[int] = None
    after_substitution_rmse: Optional[float] = None
    after_substitution_mae: Optional[float] = None
    after_substitution_messages: Optional[int] = None
    # Meta-model prediction metrics
    meta_model_metrics: Optional[MetaModelPredictionMetrics] = None


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
            # This is a meta-model prediction file
            # Rename columns to match standard format for compatibility
            df['delay_ms'] = df['actual_delay_ms']
            # Keep both actual and predicted for analysis
            return df
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


def analyze_meta_model_predictions(df: pd.DataFrame,
                                   substitution_message_index: Optional[int] = None) -> MetaModelPredictionMetrics:
    """
    Analyze meta-model prediction accuracy.

    Args:
        df: DataFrame with actual_delay_ms and predicted_delay_ms columns
        substitution_message_index: Optional message index for before/after analysis

    Returns:
        MetaModelPredictionMetrics object
    """
    # Remove rows with missing predictions or actual values
    valid_data = df.dropna(subset=['actual_delay_ms', 'predicted_delay_ms'])

    if len(valid_data) == 0:
        raise ValueError("No valid prediction data found")

    actual = valid_data['actual_delay_ms'].values
    predicted = valid_data['predicted_delay_ms'].values

    # Calculate overall prediction metrics
    prediction_errors = predicted - actual
    prediction_rmse = np.sqrt(np.mean(prediction_errors ** 2))
    prediction_mae = np.mean(np.abs(prediction_errors))
    prediction_correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0.0
    prediction_bias = np.mean(prediction_errors)

    # Initialize before/after metrics
    before_pred_rmse = before_pred_mae = None
    after_pred_rmse = after_pred_mae = None

    # Calculate before/after substitution prediction metrics if applicable
    if substitution_message_index is not None:
        try:
            # Convert msg_id to integer for comparison
            valid_data_copy = valid_data.copy()
            # Split data based on substitution
            before_sub = valid_data_copy[:substitution_message_index]
            after_sub = valid_data_copy[substitution_message_index:]

            if len(before_sub) > 0:
                before_errors = before_sub['predicted_delay_ms'] - before_sub['actual_delay_ms']
                before_pred_rmse = np.sqrt(np.mean(before_errors ** 2))
                before_pred_mae = np.mean(np.abs(before_errors))

            if len(after_sub) > 0:
                after_errors = after_sub['predicted_delay_ms'] - after_sub['actual_delay_ms']
                after_pred_rmse = np.sqrt(np.mean(after_errors ** 2))
                after_pred_mae = np.mean(np.abs(after_errors))

        except Exception as e:
            print(f"Warning: Could not calculate before/after prediction metrics: {e}")

    return MetaModelPredictionMetrics(
        prediction_rmse=prediction_rmse,
        prediction_mae=prediction_mae,
        prediction_correlation=prediction_correlation,
        num_predictions=len(valid_data),
        mean_actual_delay=np.mean(actual),
        mean_predicted_delay=np.mean(predicted),
        prediction_bias=prediction_bias,
        before_substitution_prediction_rmse=before_pred_rmse,
        before_substitution_prediction_mae=before_pred_mae,
        after_substitution_prediction_rmse=after_pred_rmse,
        after_substitution_prediction_mae=after_pred_mae
    )


def load_performance_and_substitution_data(file_path: Path) -> Tuple[
    Optional[PerformanceMetrics], Optional[SubstitutionInfo]]:
    """Load performance statistics and substitution info from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load performance metrics
        perf_metrics = PerformanceMetrics(
            execution_time_s=data.get('execution_run_time_s', 0.0),
            memory_peak_mb=data.get('memory_statistics', {}).get('peak_memory_mb', 0.0),
            memory_avg_mb=data.get('memory_statistics', {}).get('avg_memory_mb', 0.0),
            memory_change_mb=data.get('memory_statistics', {}).get('memory_change_mb', 0.0)
        )

        # Load substitution info
        substitution_data = data.get('meta_model_substitution', {})
        substitution_info = SubstitutionInfo(
            substitution_occurred=substitution_data.get('substitution_occurred', False),
            substitution_time_s=substitution_data.get('substitution_time_s'),
            substitution_message_index=substitution_data.get('substitution_message_index'),
            substitution_runtime_s=substitution_data.get('substitution_runtime_s'),
            confidence_score=substitution_data.get('confidence_score'),
            additional_info={k: v for k, v in substitution_data.items()
                             if k not in ['substitution_occurred', 'substitution_time_s',
                                          'substitution_message_index', 'substitution_runtime_s', 'confidence_score']}
        )

        return perf_metrics, substitution_info
    except Exception as e:
        print(f"Warning: Could not load performance/substitution data from {file_path}: {e}")
        return None, None


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


def calculate_metrics_before_after_substitution(detailed_df: pd.DataFrame, model_df: pd.DataFrame,
                                                substitution_message_index: int) -> Tuple[dict, dict]:
    """
    Calculate RMSE and MAE before and after substitution based on message index.

    Note: msg_id is a string, so we need to convert substitution_message_index to string for comparison.

    Returns:
        Tuple of (before_metrics, after_metrics) where each is a dict with 'rmse', 'mae', 'num_messages'
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

    # Create a temporary dataframe to work with message ordering
    temp_df = pd.DataFrame({
        'detailed': detailed_common,
        'model': model_common
    }).reset_index()

    # Split data based on substitution message index
    before_substitution = temp_df[:substitution_message_index]
    after_substitution = temp_df[substitution_message_index:]

    # Calculate metrics for before substitution
    before_metrics = {'rmse': None, 'mae': None, 'num_messages': 0}
    if len(before_substitution) > 0:
        before_diff = before_substitution['model'] - before_substitution['detailed']
        before_metrics = {
            'rmse': np.sqrt(np.mean(before_diff ** 2)),
            'mae': np.mean(np.abs(before_diff)),
            'num_messages': len(before_substitution)
        }

    # Calculate metrics for after substitution
    after_metrics = {'rmse': None, 'mae': None, 'num_messages': 0}
    if len(after_substitution) > 0:
        after_diff = after_substitution['model'] - after_substitution['detailed']
        after_metrics = {
            'rmse': np.sqrt(np.mean(after_diff ** 2)),
            'mae': np.mean(np.abs(after_diff)),
            'num_messages': len(after_substitution)
        }

    return before_metrics, after_metrics


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

    # Also look for meta-model detail files (different naming pattern)
    meta_model_files = list(results_path.glob("cocoon_meta_model*.csv"))

    print(
        f"Found {len(csv_files)} standard CSV files, {len(meta_model_files)} meta-model detail files, and {len(json_files)} JSON files in {results_folder}")

    # Load all simulation data and performance data
    all_results = {}
    detailed_results = {}
    performance_data = {}
    substitution_data = {}
    meta_model_detailed_data = {}

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

    # Load meta-model detail files
    for meta_file in meta_model_files:
        # Parse filename to get scenario info - these have different naming pattern
        filename = meta_file.name
        try:
            # Extract scenario ID from filename like "cocoon_meta_modeltensmallone_mincbr_broadcast_1_mpssimbench_lte450half0.csv"
            if filename.startswith('cocoon_meta_model'):
                scenario_part = filename[18:]  # Remove 'cocoon_meta_model' prefix
                if scenario_part.endswith('.csv'):
                    scenario_part = scenario_part[:-4]  # Remove .csv extension

                # Try to find matching scenario in all_results
                matching_scenario = None
                for existing_scenario_id in all_results.keys():
                    if ('meta_model' in existing_scenario_id and
                            scenario_part in existing_scenario_id):
                        matching_scenario = existing_scenario_id
                        break

                if matching_scenario:
                    df = load_simulation_data(meta_file)
                    if df is not None:
                        meta_model_detailed_data[matching_scenario] = df
                        print(f"Loaded meta-model detail file for {matching_scenario}")

        except Exception as e:
            print(f"Warning: Could not parse meta-model file {filename}: {e}")
            continue

    # Load JSON files (performance and substitution data)
    for json_file in json_files:
        config = parse_filename_to_config(json_file.name)
        if config is None:
            continue

        perf_data, sub_data = load_performance_and_substitution_data(json_file)
        if perf_data is None and sub_data is None:
            continue

        scenario_id = config.scenario_id
        if perf_data:
            performance_data[scenario_id] = perf_data
        if sub_data:
            substitution_data[scenario_id] = sub_data

    print(f"Successfully loaded {len(all_results)} simulation results")
    print(f"Found {len(detailed_results)} detailed simulation baselines")
    print(f"Loaded {len(performance_data)} performance statistics")
    print(f"Loaded {len(substitution_data)} substitution records")
    print(f"Loaded {len(meta_model_detailed_data)} meta-model detail files")

    # Process all simulations (including detailed ones)
    evaluation_results = []

    for scenario_id, model_df in all_results.items():
        try:
            config = ScenarioConfiguration.from_scenario_id(scenario_id)

            # Get performance metrics and substitution info if available
            perf_metrics = performance_data.get(scenario_id)
            sub_info = substitution_data.get(scenario_id)

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
                    performance_metrics=perf_metrics,
                    substitution_info=sub_info
                )

                perf_str = ""
                if perf_metrics:
                    perf_str = f", Runtime={perf_metrics.execution_time_s:.2f}s, Memory={perf_metrics.memory_peak_mb:.1f}MB"

                print(
                    f"✓ {scenario_id} (BASELINE): Messages={len(model_df)}, Mean Delay={model_df['delay_ms'].mean():.2f}ms{perf_str}")

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

                # Initialize before/after metrics
                before_sub_rmse = before_sub_mae = before_sub_msgs = None
                after_sub_rmse = after_sub_mae = after_sub_msgs = None

                # Calculate before/after substitution metrics if substitution occurred
                if (sub_info and sub_info.substitution_occurred and
                        sub_info.substitution_message_index is not None):
                    try:
                        before_metrics, after_metrics = calculate_metrics_before_after_substitution(
                            detailed_df, model_df, sub_info.substitution_message_index
                        )
                        before_sub_rmse = before_metrics['rmse']
                        before_sub_mae = before_metrics['mae']
                        before_sub_msgs = before_metrics['num_messages']
                        after_sub_rmse = after_metrics['rmse']
                        after_sub_mae = after_metrics['mae']
                        after_sub_msgs = after_metrics['num_messages']
                    except Exception as e:
                        print(f"Warning: Could not calculate before/after metrics for {scenario_id}: {e}")

                # Analyze meta-model predictions if available
                meta_model_pred_metrics = None
                if scenario_id in meta_model_detailed_data:
                    try:
                        meta_df = meta_model_detailed_data[scenario_id]
                        substitution_idx = sub_info.substitution_message_index if (
                                sub_info and sub_info.substitution_occurred) else None
                        meta_model_pred_metrics = analyze_meta_model_predictions(meta_df, substitution_idx)
                    except Exception as e:
                        print(f"Warning: Could not analyze meta-model predictions for {scenario_id}: {e}")

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    rmse=rmse,
                    mae=mae,
                    num_messages=num_messages,
                    mean_delay_detailed=mean_delay_detailed,
                    mean_delay_model=mean_delay_model,
                    performance_metrics=perf_metrics,
                    substitution_info=sub_info,
                    before_substitution_rmse=before_sub_rmse,
                    before_substitution_mae=before_sub_mae,
                    before_substitution_messages=before_sub_msgs,
                    after_substitution_rmse=after_sub_rmse,
                    after_substitution_mae=after_sub_mae,
                    after_substitution_messages=after_sub_msgs,
                    meta_model_metrics=meta_model_pred_metrics
                )

                perf_str = ""
                if perf_metrics:
                    perf_str = f", Runtime={perf_metrics.execution_time_s:.2f}s, Memory={perf_metrics.memory_peak_mb:.1f}MB"

                sub_str = ""
                if sub_info and sub_info.substitution_occurred:
                    sub_str = f", SUBSTITUTION at {sub_info.substitution_time_s:.1f}s (msg #{sub_info.substitution_message_index})"
                    if before_sub_rmse is not None and after_sub_rmse is not None:
                        sub_str += f", Before RMSE={before_sub_rmse:.2f}ms, After RMSE={after_sub_rmse:.2f}ms"

                pred_str = ""
                if meta_model_pred_metrics:
                    pred_str = f", Pred RMSE={meta_model_pred_metrics.prediction_rmse:.2f}ms, Corr={meta_model_pred_metrics.prediction_correlation:.3f}"

                print(
                    f"✓ {scenario_id}: RMSE={rmse:.2f}ms, MAE={mae:.2f}ms, Messages={num_messages}{perf_str}{sub_str}{pred_str}")

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
            'relative_error_percent': ((
                                               result.mean_delay_model - result.mean_delay_detailed) / result.mean_delay_detailed * 100) if result.mean_delay_detailed and result.mean_delay_detailed > 0 else None
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

        # Add substitution info if available
        if result.substitution_info:
            row.update({
                'substitution_occurred': result.substitution_info.substitution_occurred,
                'substitution_time_s': result.substitution_info.substitution_time_s,
                'substitution_message_index': result.substitution_info.substitution_message_index,
                'substitution_runtime_s': result.substitution_info.substitution_runtime_s,
                'confidence_score': result.substitution_info.confidence_score,
                'before_substitution_rmse_ms': result.before_substitution_rmse,
                'before_substitution_mae_ms': result.before_substitution_mae,
                'before_substitution_messages': result.before_substitution_messages,
                'after_substitution_rmse_ms': result.after_substitution_rmse,
                'after_substitution_mae_ms': result.after_substitution_mae,
                'after_substitution_messages': result.after_substitution_messages
            })
        else:
            row.update({
                'substitution_occurred': False,
                'substitution_time_s': None,
                'substitution_message_index': None,
                'substitution_runtime_s': None,
                'confidence_score': None,
                'before_substitution_rmse_ms': None,
                'before_substitution_mae_ms': None,
                'before_substitution_messages': None,
                'after_substitution_rmse_ms': None,
                'after_substitution_mae_ms': None,
                'after_substitution_messages': None
            })

        # Add meta-model prediction metrics if available
        if result.meta_model_metrics:
            row.update({
                'prediction_rmse_ms': result.meta_model_metrics.prediction_rmse,
                'prediction_mae_ms': result.meta_model_metrics.prediction_mae,
                'prediction_correlation': result.meta_model_metrics.prediction_correlation,
                'prediction_bias_ms': result.meta_model_metrics.prediction_bias,
                'num_predictions': result.meta_model_metrics.num_predictions,
                'mean_actual_delay_ms': result.meta_model_metrics.mean_actual_delay,
                'mean_predicted_delay_ms': result.meta_model_metrics.mean_predicted_delay,
                'before_sub_prediction_rmse_ms': result.meta_model_metrics.before_substitution_prediction_rmse,
                'before_sub_prediction_mae_ms': result.meta_model_metrics.before_substitution_prediction_mae,
                'after_sub_prediction_rmse_ms': result.meta_model_metrics.after_substitution_prediction_rmse,
                'after_sub_prediction_mae_ms': result.meta_model_metrics.after_substitution_prediction_mae
            })
        else:
            row.update({
                'prediction_rmse_ms': None,
                'prediction_mae_ms': None,
                'prediction_correlation': None,
                'prediction_bias_ms': None,
                'num_predictions': None,
                'mean_actual_delay_ms': None,
                'mean_predicted_delay_ms': None,
                'before_sub_prediction_rmse_ms': None,
                'before_sub_prediction_mae_ms': None,
                'after_sub_prediction_rmse_ms': None,
                'after_sub_prediction_mae_ms': None
            })

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")


def print_meta_model_prediction_analysis(results: List[EvaluationResult]):
    """Print meta-model prediction analysis."""
    print(f"\n" + "=" * 80)
    print("META-MODEL PREDICTION ANALYSIS")
    print("=" * 80)

    # Filter meta-model results with prediction data
    meta_model_results = [r for r in results if r.model_type == ModelType.meta_model and r.meta_model_metrics]

    if not meta_model_results:
        print("No meta-model results with prediction data found.")
        return

    print(f"Analyzing {len(meta_model_results)} meta-model scenarios with prediction data")

    # Overall prediction accuracy
    pred_rmse_values = [r.meta_model_metrics.prediction_rmse for r in meta_model_results]
    pred_mae_values = [r.meta_model_metrics.prediction_mae for r in meta_model_results]
    pred_corr_values = [r.meta_model_metrics.prediction_correlation for r in meta_model_results]
    pred_bias_values = [r.meta_model_metrics.prediction_bias for r in meta_model_results]

    print(f"\nOVERALL PREDICTION ACCURACY:")
    print("-" * 50)
    print(f"  Prediction RMSE (ms): Mean={np.mean(pred_rmse_values):.2f}, Std={np.std(pred_rmse_values):.2f}")
    print(f"                        Min={np.min(pred_rmse_values):.2f}, Max={np.max(pred_rmse_values):.2f}")
    print(f"  Prediction MAE (ms):  Mean={np.mean(pred_mae_values):.2f}, Std={np.std(pred_mae_values):.2f}")
    print(f"                        Min={np.min(pred_mae_values):.2f}, Max={np.max(pred_mae_values):.2f}")
    print(f"  Correlation:          Mean={np.mean(pred_corr_values):.3f}, Std={np.std(pred_corr_values):.3f}")
    print(f"                        Min={np.min(pred_corr_values):.3f}, Max={np.max(pred_corr_values):.3f}")
    print(f"  Bias (ms):            Mean={np.mean(pred_bias_values):.2f}, Std={np.std(pred_bias_values):.2f}")

    # Compare prediction accuracy vs simulation accuracy
    print(f"\nPREDICTION vs SIMULATION ACCURACY:")
    print("-" * 50)

    simulation_rmse = [r.rmse for r in meta_model_results if r.rmse is not None]
    simulation_mae = [r.mae for r in meta_model_results if r.mae is not None]

    if simulation_rmse and pred_rmse_values:
        print(f"  Simulation RMSE vs Detailed: Mean={np.mean(simulation_rmse):.2f}ms")
        print(f"  Prediction RMSE vs Actual:   Mean={np.mean(pred_rmse_values):.2f}ms")
        print(f"  Ratio (Pred/Sim):            {np.mean(pred_rmse_values) / np.mean(simulation_rmse):.2f}")

    # Analyze before/after substitution prediction accuracy
    before_pred_results = [r for r in meta_model_results
                           if (r.meta_model_metrics.before_substitution_prediction_rmse is not None)]
    after_pred_results = [r for r in meta_model_results
                          if (r.meta_model_metrics.after_substitution_prediction_rmse is not None)]

    if before_pred_results and after_pred_results:
        print(f"\nBEFORE vs AFTER SUBSTITUTION PREDICTION ACCURACY:")
        print("-" * 50)

        before_pred_rmse = [r.meta_model_metrics.before_substitution_prediction_rmse for r in before_pred_results]
        after_pred_rmse = [r.meta_model_metrics.after_substitution_prediction_rmse for r in after_pred_results]
        before_pred_mae = [r.meta_model_metrics.before_substitution_prediction_mae for r in before_pred_results]
        after_pred_mae = [r.meta_model_metrics.after_substitution_prediction_mae for r in after_pred_results]

        print(f"  BEFORE substitution (learning phase):")
        print(f"    Prediction RMSE: Mean={np.mean(before_pred_rmse):.2f}ms, Std={np.std(before_pred_rmse):.2f}ms")
        print(f"    Prediction MAE:  Mean={np.mean(before_pred_mae):.2f}ms, Std={np.std(before_pred_mae):.2f}ms")

        print(f"  AFTER substitution (pure prediction phase):")
        print(f"    Prediction RMSE: Mean={np.mean(after_pred_rmse):.2f}ms, Std={np.std(after_pred_rmse):.2f}ms")
        print(f"    Prediction MAE:  Mean={np.mean(after_pred_mae):.2f}ms, Std={np.std(after_pred_mae):.2f}ms")

        # Analyze prediction quality degradation
        if len(before_pred_rmse) == len(after_pred_rmse):
            pred_rmse_degradation = np.mean(after_pred_rmse) - np.mean(before_pred_rmse)
            pred_mae_degradation = np.mean(after_pred_mae) - np.mean(before_pred_mae)

            print(f"  PREDICTION QUALITY CHANGE:")
            print(
                f"    RMSE change: {pred_rmse_degradation:.2f}ms ({pred_rmse_degradation / np.mean(before_pred_rmse) * 100:.1f}%)")
            print(
                f"    MAE change:  {pred_mae_degradation:.2f}ms ({pred_mae_degradation / np.mean(before_pred_mae) * 100:.1f}%)")

    # Analyze prediction accuracy by network type
    print(f"\nPREDICTION ACCURACY BY NETWORK TYPE:")
    print("-" * 50)

    by_network = {}
    for result in meta_model_results:
        network_name = result.scenario_config.network_type.name
        if network_name not in by_network:
            by_network[network_name] = []
        by_network[network_name].append(result)

    for network_name, network_results in by_network.items():
        network_pred_rmse = [r.meta_model_metrics.prediction_rmse for r in network_results]
        network_pred_corr = [r.meta_model_metrics.prediction_correlation for r in network_results]

        print(
            f"  {network_name}: RMSE={np.mean(network_pred_rmse):.2f}ms, Correlation={np.mean(network_pred_corr):.3f} ({len(network_results)} scenarios)")


def print_substitution_analysis(results: List[EvaluationResult]):
    """Print meta-model substitution analysis."""
    print(f"\n" + "=" * 80)
    print("META-MODEL SUBSTITUTION ANALYSIS")
    print("=" * 80)

    # Filter meta-model results with substitution info
    meta_model_results = [r for r in results if r.model_type == ModelType.meta_model and r.substitution_info]

    if not meta_model_results:
        print("No meta-model results with substitution information found.")
        return

    print(f"Analyzing {len(meta_model_results)} meta-model scenarios")

    # Count substitutions
    substitution_results = [r for r in meta_model_results if r.substitution_info.substitution_occurred]
    no_substitution_results = [r for r in meta_model_results if not r.substitution_info.substitution_occurred]

    print(f"\nSUBSTITUTION OCCURRENCE:")
    print(
        f"  Scenarios with substitution: {len(substitution_results)} ({len(substitution_results) / len(meta_model_results) * 100:.1f}%)")
    print(
        f"  Scenarios without substitution: {len(no_substitution_results)} ({len(no_substitution_results) / len(meta_model_results) * 100:.1f}%)")

    if substitution_results:
        print(f"\nSUBSTITUTION TIMING ANALYSIS:")
        print("-" * 50)

        # Analyze substitution timing
        sub_times = [r.substitution_info.substitution_time_s for r in substitution_results if
                     r.substitution_info.substitution_time_s is not None]
        sub_runtimes = [r.substitution_info.substitution_runtime_s for r in substitution_results if
                        r.substitution_info.substitution_runtime_s is not None]
        sub_msg_indices = [r.substitution_info.substitution_message_index for r in substitution_results if
                           r.substitution_info.substitution_message_index is not None]
        confidence_scores = [r.substitution_info.confidence_score for r in substitution_results if
                             r.substitution_info.confidence_score is not None]

        if sub_times:
            print(f"  Simulation time (s): Mean={np.mean(sub_times):.2f}, Std={np.std(sub_times):.2f}")
            print(f"                      Min={np.min(sub_times):.2f}, Max={np.max(sub_times):.2f}")

        if sub_runtimes:
            print(f"  Real runtime (s):   Mean={np.mean(sub_runtimes):.2f}, Std={np.std(sub_runtimes):.2f}")
            print(f"                      Min={np.min(sub_runtimes):.2f}, Max={np.max(sub_runtimes):.2f}")

        if sub_msg_indices:
            print(f"  Message index:      Mean={np.mean(sub_msg_indices):.0f}, Std={np.std(sub_msg_indices):.0f}")
            print(f"                      Min={np.min(sub_msg_indices)}, Max={np.max(sub_msg_indices)}")

        if confidence_scores:
            print(f"  Confidence score:   Mean={np.mean(confidence_scores):.3f}, Std={np.std(confidence_scores):.3f}")
            print(f"                      Min={np.min(confidence_scores):.3f}, Max={np.max(confidence_scores):.3f}")

        # Analyze accuracy impact of substitution
        print(f"\nACCURACY IMPACT OF SUBSTITUTION:")
        print("-" * 50)

        # Compare accuracy between scenarios with and without substitution
        if no_substitution_results:
            sub_rmse = [r.rmse for r in substitution_results if r.rmse is not None]
            no_sub_rmse = [r.rmse for r in no_substitution_results if r.rmse is not None]

            if sub_rmse and no_sub_rmse:
                print(f"  RMSE with substitution:    Mean={np.mean(sub_rmse):.2f}ms, Std={np.std(sub_rmse):.2f}ms")
                print(
                    f"  RMSE without substitution: Mean={np.mean(no_sub_rmse):.2f}ms, Std={np.std(no_sub_rmse):.2f}ms")
                print(f"  RMSE difference:           {np.mean(sub_rmse) - np.mean(no_sub_rmse):.2f}ms")

            sub_mae = [r.mae for r in substitution_results if r.mae is not None]
            no_sub_mae = [r.mae for r in no_substitution_results if r.mae is not None]

            if sub_mae and no_sub_mae:
                print(f"  MAE with substitution:     Mean={np.mean(sub_mae):.2f}ms, Std={np.std(sub_mae):.2f}ms")
                print(f"  MAE without substitution:  Mean={np.mean(no_sub_mae):.2f}ms, Std={np.std(no_sub_mae):.2f}ms")
                print(f"  MAE difference:            {np.mean(sub_mae) - np.mean(no_sub_mae):.2f}ms")

        # Analyze before vs after substitution accuracy
        print(f"\nBEFORE vs AFTER SUBSTITUTION ACCURACY:")
        print("-" * 50)

        # Get results with before/after metrics
        before_after_results = [r for r in substitution_results
                                if (r.before_substitution_rmse is not None and
                                    r.after_substitution_rmse is not None)]

        if before_after_results:
            before_rmse = [r.before_substitution_rmse for r in before_after_results]
            after_rmse = [r.after_substitution_rmse for r in before_after_results]
            before_mae = [r.before_substitution_mae for r in before_after_results]
            after_mae = [r.after_substitution_mae for r in before_after_results]
            before_msgs = [r.before_substitution_messages for r in before_after_results]
            after_msgs = [r.after_substitution_messages for r in before_after_results]

            print(f"  Scenarios with before/after analysis: {len(before_after_results)}")
            print(f"  BEFORE substitution (detailed phase):")
            print(f"    RMSE: Mean={np.mean(before_rmse):.2f}ms, Std={np.std(before_rmse):.2f}ms")
            print(f"    MAE:  Mean={np.mean(before_mae):.2f}ms, Std={np.std(before_mae):.2f}ms")
            print(f"    Messages: Mean={np.mean(before_msgs):.0f}, Total={np.sum(before_msgs)}")

            print(f"  AFTER substitution (meta-model phase):")
            print(f"    RMSE: Mean={np.mean(after_rmse):.2f}ms, Std={np.std(after_rmse):.2f}ms")
            print(f"    MAE:  Mean={np.mean(after_mae):.2f}ms, Std={np.std(after_mae):.2f}ms")
            print(f"    Messages: Mean={np.mean(after_msgs):.0f}, Total={np.sum(after_msgs)}")

            # Analyze message distribution
            total_before = np.sum(before_msgs)
            total_after = np.sum(after_msgs)
            total_messages = total_before + total_after

            print(f"  MESSAGE DISTRIBUTION:")
            print(f"    Before substitution: {total_before} messages ({total_before / total_messages * 100:.1f}%)")
            print(f"    After substitution:  {total_after} messages ({total_after / total_messages * 100:.1f}%)")

        else:
            print("  No scenarios with complete before/after substitution data found.")

        # Analyze performance impact
        print(f"\nPERFORMANCE IMPACT OF SUBSTITUTION:")
        print("-" * 50)

        sub_perf = [r.performance_metrics for r in substitution_results if r.performance_metrics]
        no_sub_perf = [r.performance_metrics for r in no_substitution_results if r.performance_metrics]

        if sub_perf and no_sub_perf:
            sub_runtimes_perf = [p.execution_time_s for p in sub_perf]
            no_sub_runtimes_perf = [p.execution_time_s for p in no_sub_perf]

            print(f"  Runtime with substitution:    Mean={np.mean(sub_runtimes_perf):.2f}s")
            print(f"  Runtime without substitution: Mean={np.mean(no_sub_runtimes_perf):.2f}s")
            print(f"  Runtime speedup:              {np.mean(no_sub_runtimes_perf) / np.mean(sub_runtimes_perf):.2f}x")

        # Group by network type for substitution analysis
        print(f"\nSUBSTITUTION BY NETWORK TYPE:")
        print("-" * 50)

        by_network = {}
        for result in meta_model_results:
            network_name = result.scenario_config.network_type.name
            if network_name not in by_network:
                by_network[network_name] = {'total': 0, 'substituted': 0}
            by_network[network_name]['total'] += 1
            if result.substitution_info.substitution_occurred:
                by_network[network_name]['substituted'] += 1

        for network_name, counts in by_network.items():
            substitution_rate = counts['substituted'] / counts['total'] * 100
            print(f"  {network_name}: {counts['substituted']}/{counts['total']} scenarios ({substitution_rate:.1f}%)")


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

    print(
        f"\nTotal scenarios: {len(results)} ({len(detailed_results)} detailed, {len(accuracy_results)} accuracy-evaluated)")

    # Add performance summary
    print_performance_summary(results)

    # Add substitution analysis
    print_substitution_analysis(results)

    # Add meta-model prediction analysis
    print_meta_model_prediction_analysis(results)


def analyze_error_scenarios(results_folder: str) -> dict:
    """
    Analyze error scenarios from simulation results.

    Args:
        results_folder: Path to folder containing JSON statistics files

    Returns:
        Dictionary containing error analysis results
    """
    results_path = Path(results_folder)
    if not results_path.exists():
        raise ValueError(f"Results folder does not exist: {results_folder}")

    # Find all JSON statistics files
    json_files = list(results_path.glob("statistics_*.json"))

    if not json_files:
        print("No statistics files found for error analysis.")
        return {}

    print(f"Analyzing {len(json_files)} statistics files for errors...")

    # Error tracking
    error_scenarios = []
    timeout_scenarios = []
    successful_scenarios = []
    total_scenarios = 0

    # Error pattern tracking
    error_types = {}
    timeout_patterns = {}
    error_by_model_type = {}
    error_by_network_type = {}
    error_by_scenario_complexity = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            total_scenarios += 1
            scenario_id = data.get('scenario_id', json_file.stem)

            # Parse scenario configuration for pattern analysis
            try:
                config = ScenarioConfiguration.from_scenario_id(scenario_id)
                model_type = config.model_type.name
                network_type = config.network_type.name
                num_devices = config.num_devices.name
                payload_size = config.payload_size.name
            except Exception as e:
                print(f"Warning: Could not parse scenario config for {scenario_id}: {e}")
                model_type = network_type = num_devices = payload_size = "unknown"

            # Check for errors
            if data.get('error_occurred', False):
                error_info = data.get('error_info', {})
                error_scenario = {
                    'scenario_id': scenario_id,
                    'model_type': model_type,
                    'network_type': network_type,
                    'num_devices': num_devices,
                    'payload_size': payload_size,
                    'execution_time_s': data.get('execution_run_time_s', 0),
                    'error_type': error_info.get('error_type', 'Unknown'),
                    'error_message': error_info.get('error_message', ''),
                    'error_details': error_info.get('error_details', ''),
                    'messages_sent': data.get('messages_sent', 0),
                    'messages_received': data.get('messages_received', 0),
                    'messages_completed': data.get('messages_completed', 0),
                    'memory_peak_mb': data.get('memory_statistics', {}).get('peak_memory_mb', 0),
                }
                error_scenarios.append(error_scenario)

                # Track error patterns
                error_type = error_info.get('error_type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1

                if model_type not in error_by_model_type:
                    error_by_model_type[model_type] = {'total': 0, 'errors': 0}
                error_by_model_type[model_type]['errors'] += 1

                if network_type not in error_by_network_type:
                    error_by_network_type[network_type] = {'total': 0, 'errors': 0}
                error_by_network_type[network_type]['errors'] += 1

            # Check for timeouts
            elif data.get('timeout_occurred', False):
                timeout_scenario = {
                    'scenario_id': scenario_id,
                    'model_type': model_type,
                    'network_type': network_type,
                    'num_devices': num_devices,
                    'payload_size': payload_size,
                    'execution_time_s': data.get('execution_run_time_s', 0),
                    'timeout_limit_s': data.get('timeout_limit_s', 0),
                    'timeout_error_message': data.get('timeout_error_message', ''),
                    'messages_sent': data.get('messages_sent', 0),
                    'messages_received': data.get('messages_received', 0),
                    'messages_completed': data.get('messages_completed', 0),
                    'memory_peak_mb': data.get('memory_statistics', {}).get('peak_memory_mb', 0),
                }
                timeout_scenarios.append(timeout_scenario)

                # Track timeout patterns
                timeout_limit = data.get('timeout_limit_s', 0)
                timeout_patterns[timeout_limit] = timeout_patterns.get(timeout_limit, 0) + 1

                if model_type not in error_by_model_type:
                    error_by_model_type[model_type] = {'total': 0, 'errors': 0}
                error_by_model_type[model_type]['errors'] += 1

                if network_type not in error_by_network_type:
                    error_by_network_type[network_type] = {'total': 0, 'errors': 0}
                error_by_network_type[network_type]['errors'] += 1

            else:
                # Successful scenario
                successful_scenario = {
                    'scenario_id': scenario_id,
                    'model_type': model_type,
                    'network_type': network_type,
                    'execution_time_s': data.get('execution_run_time_s', 0),
                    'messages_sent': len([]) if 'messages_sent' not in data else data['messages_sent'],
                    'memory_peak_mb': data.get('memory_statistics', {}).get('peak_memory_mb', 0),
                }
                successful_scenarios.append(successful_scenario)

            # Track totals for error rate calculation
            if model_type not in error_by_model_type:
                error_by_model_type[model_type] = {'total': 0, 'errors': 0}
            error_by_model_type[model_type]['total'] += 1

            if network_type not in error_by_network_type:
                error_by_network_type[network_type] = {'total': 0, 'errors': 0}
            error_by_network_type[network_type]['total'] += 1

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    # Calculate error rates
    error_rate_by_model = {}
    for model, counts in error_by_model_type.items():
        if counts['total'] > 0:
            error_rate_by_model[model] = {
                'error_count': counts['errors'],
                'total_count': counts['total'],
                'error_rate': counts['errors'] / counts['total']
            }

    error_rate_by_network = {}
    for network, counts in error_by_network_type.items():
        if counts['total'] > 0:
            error_rate_by_network[network] = {
                'error_count': counts['errors'],
                'total_count': counts['total'],
                'error_rate': counts['errors'] / counts['total']
            }

    # Compile results
    analysis_results = {
        'summary': {
            'total_scenarios': total_scenarios,
            'successful_scenarios': len(successful_scenarios),
            'error_scenarios': len(error_scenarios),
            'timeout_scenarios': len(timeout_scenarios),
            'overall_success_rate': len(successful_scenarios) / total_scenarios if total_scenarios > 0 else 0,
            'overall_error_rate': len(error_scenarios) / total_scenarios if total_scenarios > 0 else 0,
            'overall_timeout_rate': len(timeout_scenarios) / total_scenarios if total_scenarios > 0 else 0,
        },
        'error_details': {
            'error_scenarios': error_scenarios,
            'timeout_scenarios': timeout_scenarios,
            'error_types': error_types,
            'timeout_patterns': timeout_patterns,
        },
        'error_patterns': {
            'by_model_type': error_rate_by_model,
            'by_network_type': error_rate_by_network,
        },
        'successful_scenarios': successful_scenarios,
    }

    return analysis_results


def print_error_analysis(error_analysis: dict):
    """
    Print a comprehensive error analysis report.

    Args:
        error_analysis: Dictionary returned by analyze_error_scenarios()
    """
    if not error_analysis:
        print("No error analysis data available.")
        return

    summary = error_analysis['summary']
    error_details = error_analysis['error_details']
    error_patterns = error_analysis['error_patterns']

    print(f"\n" + "=" * 80)
    print("ERROR SCENARIO ANALYSIS")
    print("=" * 80)

    # Overall summary
    print(f"\nOVERALL SUMMARY:")
    print("-" * 50)
    print(f"  Total scenarios analyzed: {summary['total_scenarios']}")
    print(f"  Successful scenarios: {summary['successful_scenarios']} ({summary['overall_success_rate'] * 100:.1f}%)")
    print(f"  Error scenarios: {summary['error_scenarios']} ({summary['overall_error_rate'] * 100:.1f}%)")
    print(f"  Timeout scenarios: {summary['timeout_scenarios']} ({summary['overall_timeout_rate'] * 100:.1f}%)")

    # Error type breakdown
    if error_details['error_types']:
        print(f"\nERROR TYPES:")
        print("-" * 50)
        for error_type, count in sorted(error_details['error_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / summary['error_scenarios'] * 100 if summary['error_scenarios'] > 0 else 0
            print(f"  {error_type}: {count} scenarios ({percentage:.1f}% of errors)")

    # Timeout patterns
    if error_details['timeout_patterns']:
        print(f"\nTIMEOUT PATTERNS:")
        print("-" * 50)
        for timeout_limit, count in sorted(error_details['timeout_patterns'].items()):
            percentage = count / summary['timeout_scenarios'] * 100 if summary['timeout_scenarios'] > 0 else 0
            print(f"  {timeout_limit}s timeout: {count} scenarios ({percentage:.1f}% of timeouts)")

    # Error rates by model type
    if error_patterns['by_model_type']:
        print(f"\nERROR RATES BY MODEL TYPE:")
        print("-" * 50)
        for model_type, stats in sorted(error_patterns['by_model_type'].items(),
                                        key=lambda x: x[1]['error_rate'], reverse=True):
            print(f"  {model_type}: {stats['error_count']}/{stats['total_count']} scenarios "
                  f"({stats['error_rate'] * 100:.1f}% error rate)")

    # Error rates by network type
    if error_patterns['by_network_type']:
        print(f"\nERROR RATES BY NETWORK TYPE:")
        print("-" * 50)
        for network_type, stats in sorted(error_patterns['by_network_type'].items(),
                                          key=lambda x: x[1]['error_rate'], reverse=True):
            print(f"  {network_type}: {stats['error_count']}/{stats['total_count']} scenarios "
                  f"({stats['error_rate'] * 100:.1f}% error rate)")

    # Detailed error scenarios
    if error_details['error_scenarios']:
        print(f"\nDETAILED ERROR SCENARIOS:")
        print("-" * 50)
        for i, error in enumerate(error_details['error_scenarios'][:10], 1):  # Show first 10
            print(f"  {i}. {error['scenario_id']} ({error['model_type']}, {error['network_type']})")
            print(f"     Error: {error['error_type']} - {error['error_message']}")
            print(
                f"     Runtime: {error['execution_time_s']:.2f}s, Messages: {error['messages_completed']}/{error['messages_sent']}")
            if error['error_details']:
                print(f"     Details: {error['error_details'][:100]}...")
            print()

        if len(error_details['error_scenarios']) > 10:
            print(f"  ... and {len(error_details['error_scenarios']) - 10} more error scenarios")

    # Detailed timeout scenarios
    if error_details['timeout_scenarios']:
        print(f"\nDETAILED TIMEOUT SCENARIOS:")
        print("-" * 50)
        for i, timeout in enumerate(error_details['timeout_scenarios'][:5], 1):  # Show first 5
            print(f"  {i}. {timeout['scenario_id']} ({timeout['model_type']}, {timeout['network_type']})")
            print(f"     Timeout: {timeout['timeout_limit_s']}s limit, ran for {timeout['execution_time_s']:.2f}s")
            print(f"     Messages: {timeout['messages_completed']}/{timeout['messages_sent']}")
            if timeout['timeout_error_message']:
                print(f"     Message: {timeout['timeout_error_message']}")
            print()

        if len(error_details['timeout_scenarios']) > 5:
            print(f"  ... and {len(error_details['timeout_scenarios']) - 5} more timeout scenarios")

    # Performance analysis of failed vs successful scenarios
    successful_scenarios = error_analysis['successful_scenarios']
    if successful_scenarios and (error_details['error_scenarios'] or error_details['timeout_scenarios']):
        print(f"\nPERFORMANCE COMPARISON (SUCCESSFUL vs FAILED):")
        print("-" * 50)

        # Successful scenario stats
        success_runtimes = [s['execution_time_s'] for s in successful_scenarios]
        success_memory = [s['memory_peak_mb'] for s in successful_scenarios if s['memory_peak_mb'] > 0]

        # Failed scenario stats
        failed_runtimes = ([e['execution_time_s'] for e in error_details['error_scenarios']] +
                           [t['execution_time_s'] for t in error_details['timeout_scenarios']])
        failed_memory = ([e['memory_peak_mb'] for e in error_details['error_scenarios'] if e['memory_peak_mb'] > 0] +
                         [t['memory_peak_mb'] for t in error_details['timeout_scenarios'] if t['memory_peak_mb'] > 0])

        if success_runtimes:
            print(f"  Successful scenarios runtime: Mean={np.mean(success_runtimes):.2f}s, "
                  f"Std={np.std(success_runtimes):.2f}s")
        if failed_runtimes:
            print(f"  Failed scenarios runtime: Mean={np.mean(failed_runtimes):.2f}s, "
                  f"Std={np.std(failed_runtimes):.2f}s")

        if success_memory:
            print(f"  Successful scenarios memory: Mean={np.mean(success_memory):.1f}MB, "
                  f"Std={np.std(success_memory):.1f}MB")
        if failed_memory:
            print(f"  Failed scenarios memory: Mean={np.mean(failed_memory):.1f}MB, "
                  f"Std={np.std(failed_memory):.1f}MB")


def save_error_analysis(error_analysis: dict, output_file: str):
    """
    Save error analysis results to CSV files.

    Args:
        error_analysis: Dictionary returned by analyze_error_scenarios()
        output_file: Base filename for output files (without extension)
    """
    if not error_analysis:
        print("No error analysis data to save.")
        return

    # Save error scenarios
    if error_analysis['error_details']['error_scenarios']:
        error_df = pd.DataFrame(error_analysis['error_details']['error_scenarios'])
        error_file = f"{output_file}_errors.csv"
        error_df.to_csv(error_file, index=False)
        print(f"Error scenarios saved to: {error_file}")

    # Save timeout scenarios
    if error_analysis['error_details']['timeout_scenarios']:
        timeout_df = pd.DataFrame(error_analysis['error_details']['timeout_scenarios'])
        timeout_file = f"{output_file}_timeouts.csv"
        timeout_df.to_csv(timeout_file, index=False)
        print(f"Timeout scenarios saved to: {timeout_file}")

    # Save summary statistics
    summary_data = []

    # Overall summary
    summary = error_analysis['summary']
    summary_data.append({
        'category': 'overall',
        'subcategory': 'total',
        'metric': 'scenarios',
        'value': summary['total_scenarios']
    })
    summary_data.append({
        'category': 'overall',
        'subcategory': 'success',
        'metric': 'rate',
        'value': summary['overall_success_rate']
    })
    summary_data.append({
        'category': 'overall',
        'subcategory': 'error',
        'metric': 'rate',
        'value': summary['overall_error_rate']
    })
    summary_data.append({
        'category': 'overall',
        'subcategory': 'timeout',
        'metric': 'rate',
        'value': summary['overall_timeout_rate']
    })

    # Error rates by model type
    for model_type, stats in error_analysis['error_patterns']['by_model_type'].items():
        summary_data.append({
            'category': 'model_type',
            'subcategory': model_type,
            'metric': 'error_rate',
            'value': stats['error_rate']
        })

    # Error rates by network type
    for network_type, stats in error_analysis['error_patterns']['by_network_type'].items():
        summary_data.append({
            'category': 'network_type',
            'subcategory': network_type,
            'metric': 'error_rate',
            'value': stats['error_rate']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_file}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Error analysis summary saved to: {summary_file}")


@dataclass
class StochasticAnalysisResult:
    """Container for stochastic analysis metrics."""
    base_scenario_id: str  # scenario_id without run number
    model_type: str  # Changed to str to avoid ModelType import issues
    network_type: str
    num_runs: int

    # Delay variability metrics
    delay_mean_across_runs: float
    delay_std_across_runs: float
    delay_cv_across_runs: float  # coefficient of variation
    delay_min_across_runs: float
    delay_max_across_runs: float
    delay_range_across_runs: float

    # Message count variability
    num_messages_mean: float
    num_messages_std: float
    num_messages_consistent: bool  # True if all runs have same number of messages

    # RMSE/MAE variability (for non-detailed models)
    rmse_mean_across_runs: Optional[float] = None
    rmse_std_across_runs: Optional[float] = None
    rmse_cv_across_runs: Optional[float] = None
    mae_mean_across_runs: Optional[float] = None
    mae_std_across_runs: Optional[float] = None
    mae_cv_across_runs: Optional[float] = None

    # Performance variability
    runtime_mean_across_runs: Optional[float] = None
    runtime_std_across_runs: Optional[float] = None
    runtime_cv_across_runs: Optional[float] = None
    memory_mean_across_runs: Optional[float] = None
    memory_std_across_runs: Optional[float] = None
    memory_cv_across_runs: Optional[float] = None

    # Substitution variability (for meta-model)
    substitution_consistency: Optional[float] = None  # fraction of runs that substituted
    substitution_time_std: Optional[float] = None  # std of substitution times when it occurred

    # Statistical significance tests
    delay_anova_p_value: Optional[float] = None  # ANOVA p-value for delay differences
    is_stochastic_delay: Optional[bool] = None  # True if significant delay variation found


def extract_run_number_from_scenario_id(scenario_id: str) -> int:
    """
    Extract run number from scenario ID.

    Expected format: detailed-two-small-one_day-cbr_broadcast_4_mph-simbench_ethernet-none-none-0
    Returns the final number (0 in this example).
    """
    # Find the last number after the final dash
    match = re.search(r'-(\d+)$', scenario_id)
    if match:
        return int(match.group(1))
    else:
        # If no run number found, assume run 0
        return 0


def get_base_scenario_id(scenario_id: str) -> str:
    """
    Get base scenario ID by removing the run number.

    Args:
        scenario_id: Full scenario ID like "detailed-two-small-one_day-cbr_broadcast_4_mph-simbench_ethernet-none-none-0"

    Returns:
        Base scenario ID like "detailed-two-small-one_day-cbr_broadcast_4_mph-simbench_ethernet-none-none"
    """
    # Remove run number from the end (pattern: -\d+$ meaning dash followed by digits at end)
    base_scenario_id = re.sub(r'-\d+$', '', scenario_id)
    return base_scenario_id


def group_results_by_run(results: List) -> dict:
    """
    Group results by run number for debugging purposes.

    Returns:
        Dictionary mapping run numbers to lists of results
    """
    by_run = {}

    for result in results:
        run_number = extract_run_number_from_scenario_id(result.scenario_config.scenario_id)
        if run_number not in by_run:
            by_run[run_number] = []
        by_run[run_number].append(result)

    return by_run


def analyze_stochastic_behavior(results: List) -> List[StochasticAnalysisResult]:
    """
    Analyze run-to-run variability for scenarios with multiple runs.

    Args:
        results: List of EvaluationResult objects

    Returns:
        List of StochasticAnalysisResult objects for scenarios with multiple runs
    """
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not available. Statistical tests will be skipped.")
        stats = None

    # Group results by base scenario (without run number)
    base_scenario_groups = {}

    for result in results:
        # Extract base scenario ID (remove run suffix)
        scenario_id = result.scenario_config.scenario_id

        # Use the helper function to get base scenario ID
        base_scenario_id = get_base_scenario_id(scenario_id)
        run_number = extract_run_number_from_scenario_id(scenario_id)

        # Debug print for first few scenarios
        if len(base_scenario_groups) < 5:
            print(f"Scenario ID: {scenario_id} -> Base: {base_scenario_id}, Run: {run_number}")

        if base_scenario_id not in base_scenario_groups:
            base_scenario_groups[base_scenario_id] = []
        base_scenario_groups[base_scenario_id].append(result)

    print(f"Found {len(base_scenario_groups)} unique base scenarios")

    stochastic_results = []

    for base_scenario_id, run_results in base_scenario_groups.items():
        # Only analyze scenarios with multiple runs
        if len(run_results) < 1:
            continue

        print(f"Analyzing stochastic behavior for {base_scenario_id} ({len(run_results)} runs)")

        try:
            # Extract metrics from all runs
            model_type = run_results[0].model_type.name if hasattr(run_results[0].model_type, 'name') else str(
                run_results[0].model_type)
            network_type = run_results[0].scenario_config.network_type.name if hasattr(
                run_results[0].scenario_config.network_type, 'name') else str(
                run_results[0].scenario_config.network_type)

            # Delay metrics
            delay_means = [r.mean_delay_model for r in run_results]
            delay_mean = np.mean(delay_means)
            delay_std = np.std(delay_means, ddof=1) if len(delay_means) > 1 else 0.0
            delay_cv = delay_std / delay_mean if delay_mean > 0 else 0.0

            # RMSE/MAE metrics (for non-detailed models)
            rmse_values = [r.rmse for r in run_results if r.rmse is not None]
            mae_values = [r.mae for r in run_results if r.mae is not None]

            rmse_mean = rmse_std = rmse_cv = None
            mae_mean = mae_std = mae_cv = None

            if rmse_values:
                rmse_mean = np.mean(rmse_values)
                rmse_std = np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0
                rmse_cv = rmse_std / rmse_mean if rmse_mean > 0 else 0.0

            if mae_values:
                mae_mean = np.mean(mae_values)
                mae_std = np.std(mae_values, ddof=1) if len(mae_values) > 1 else 0.0
                mae_cv = mae_std / mae_mean if mae_mean > 0 else 0.0

            # Performance metrics
            runtime_values = [r.performance_metrics.execution_time_s for r in run_results
                              if r.performance_metrics is not None]
            memory_values = [r.performance_metrics.memory_peak_mb for r in run_results
                             if r.performance_metrics is not None]

            runtime_mean = runtime_std = runtime_cv = None
            memory_mean = memory_std = memory_cv = None

            if runtime_values:
                runtime_mean = np.mean(runtime_values)
                runtime_std = np.std(runtime_values, ddof=1) if len(runtime_values) > 1 else 0.0
                runtime_cv = runtime_std / runtime_mean if runtime_mean > 0 else 0.0

            if memory_values:
                memory_mean = np.mean(memory_values)
                memory_std = np.std(memory_values, ddof=1) if len(memory_values) > 1 else 0.0
                memory_cv = memory_std / memory_mean if memory_mean > 0 else 0.0

            # Message count consistency
            message_counts = [r.num_messages for r in run_results]
            num_messages_mean = np.mean(message_counts)
            num_messages_std = np.std(message_counts, ddof=1) if len(message_counts) > 1 else 0.0
            num_messages_consistent = len(set(message_counts)) == 1

            # Substitution analysis (for meta-model)
            substitution_consistency = substitution_time_std = None
            if model_type == 'meta_model':
                substitution_occurred = [r.substitution_info.substitution_occurred for r in run_results
                                         if r.substitution_info is not None]
                if substitution_occurred:
                    substitution_consistency = np.mean(substitution_occurred)

                    substitution_times = [r.substitution_info.substitution_time_s for r in run_results
                                          if (r.substitution_info is not None and
                                              r.substitution_info.substitution_occurred and
                                              r.substitution_info.substitution_time_s is not None)]
                    if len(substitution_times) > 1:
                        substitution_time_std = np.std(substitution_times, ddof=1)

            # Statistical tests for delay variation
            delay_anova_p_value = is_stochastic_delay = None
            if stats and len(run_results) >= 3:  # Need at least 3 runs for meaningful ANOVA
                try:
                    # Perform one-way ANOVA on delay means
                    # This is a simplified test - in practice you might want to compare
                    # the full delay distributions from each run
                    _, delay_anova_p_value = stats.f_oneway(*[np.array([r.mean_delay_model]) for r in run_results])
                    is_stochastic_delay = delay_anova_p_value < 0.05  # Significant at 5% level
                except Exception as e:
                    print(f"Warning: Could not perform ANOVA test: {e}")
                    delay_anova_p_value = None
                    is_stochastic_delay = None

            # Create result object
            stochastic_result = StochasticAnalysisResult(
                base_scenario_id=base_scenario_id,
                model_type=model_type,
                network_type=network_type,
                num_runs=len(run_results),

                delay_mean_across_runs=delay_mean,
                delay_std_across_runs=delay_std,
                delay_cv_across_runs=delay_cv,
                delay_min_across_runs=np.min(delay_means),
                delay_max_across_runs=np.max(delay_means),
                delay_range_across_runs=np.max(delay_means) - np.min(delay_means),

                rmse_mean_across_runs=rmse_mean,
                rmse_std_across_runs=rmse_std,
                rmse_cv_across_runs=rmse_cv,
                mae_mean_across_runs=mae_mean,
                mae_std_across_runs=mae_std,
                mae_cv_across_runs=mae_cv,

                runtime_mean_across_runs=runtime_mean,
                runtime_std_across_runs=runtime_std,
                runtime_cv_across_runs=runtime_cv,
                memory_mean_across_runs=memory_mean,
                memory_std_across_runs=memory_std,
                memory_cv_across_runs=memory_cv,

                num_messages_mean=num_messages_mean,
                num_messages_std=num_messages_std,
                num_messages_consistent=num_messages_consistent,

                substitution_consistency=substitution_consistency,
                substitution_time_std=substitution_time_std,

                delay_anova_p_value=delay_anova_p_value,
                is_stochastic_delay=is_stochastic_delay
            )

            stochastic_results.append(stochastic_result)

        except Exception as e:
            print(f"Error analyzing {base_scenario_id}: {e}")
            continue

    return stochastic_results


def print_stochastic_analysis(stochastic_results: List[StochasticAnalysisResult]):
    """Print stochastic behavior analysis."""
    if not stochastic_results:
        print("No scenarios with multiple runs found for stochastic analysis.")
        return

    print(f"\n" + "=" * 80)
    print("STOCHASTIC BEHAVIOR ANALYSIS")
    print("=" * 80)

    print(f"Analyzed {len(stochastic_results)} scenarios with multiple runs")

    # Overall stochasticity summary
    print(f"\nOVERALL STOCHASTICITY SUMMARY:")
    print("-" * 50)

    # Delay variability
    delay_cvs = [r.delay_cv_across_runs for r in stochastic_results]
    high_delay_variability = len([cv for cv in delay_cvs if cv > 0.05])  # CV > 5%
    moderate_delay_variability = len([cv for cv in delay_cvs if 0.01 < cv <= 0.05])  # 1% < CV <= 5%
    low_delay_variability = len([cv for cv in delay_cvs if cv <= 0.01])  # CV <= 1%

    print(f"  Delay Variability:")
    print(
        f"    High (CV > 5%): {high_delay_variability} scenarios ({high_delay_variability / len(stochastic_results) * 100:.1f}%)")
    print(
        f"    Moderate (1% < CV <= 5%): {moderate_delay_variability} scenarios ({moderate_delay_variability / len(stochastic_results) * 100:.1f}%)")
    print(
        f"    Low (CV <= 1%): {low_delay_variability} scenarios ({low_delay_variability / len(stochastic_results) * 100:.1f}%)")

    # Message count consistency
    consistent_message_counts = len([r for r in stochastic_results if r.num_messages_consistent])
    print(
        f"  Message Count Consistency: {consistent_message_counts}/{len(stochastic_results)} scenarios ({consistent_message_counts / len(stochastic_results) * 100:.1f}%)")

    # Statistical significance
    statistically_significant = [r for r in stochastic_results if r.is_stochastic_delay is True]
    if statistically_significant:
        print(f"  Statistically Significant Variation: {len(statistically_significant)} scenarios")

    # Analysis by model type
    print(f"\nSTOCHASTICITY BY MODEL TYPE:")
    print("-" * 50)

    by_model = {}
    for result in stochastic_results:
        model_name = result.model_type
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(result)

    for model_name, model_results in by_model.items():
        delay_cvs = [r.delay_cv_across_runs for r in model_results]
        runtime_cvs = [r.runtime_cv_across_runs for r in model_results if r.runtime_cv_across_runs is not None]

        print(f"\n  {model_name.upper()}:")
        print(f"    Scenarios: {len(model_results)}")
        print(f"    Delay CV: Mean={np.mean(delay_cvs):.4f}, Std={np.std(delay_cvs):.4f}")
        print(f"             Min={np.min(delay_cvs):.4f}, Max={np.max(delay_cvs):.4f}")

        if runtime_cvs:
            print(f"    Runtime CV: Mean={np.mean(runtime_cvs):.4f}, Std={np.std(runtime_cvs):.4f}")

        # Message consistency
        consistent_count = len([r for r in model_results if r.num_messages_consistent])
        print(
            f"    Message Consistency: {consistent_count}/{len(model_results)} ({consistent_count / len(model_results) * 100:.1f}%)")

        # RMSE/MAE variability for non-detailed models
        rmse_cvs = [r.rmse_cv_across_runs for r in model_results if r.rmse_cv_across_runs is not None]
        if rmse_cvs:
            print(f"    RMSE CV: Mean={np.mean(rmse_cvs):.4f}, Std={np.std(rmse_cvs):.4f}")

    # Detailed analysis of high-variability scenarios
    print(f"\nHIGH-VARIABILITY SCENARIOS (CV > 5%):")
    print("-" * 50)

    high_variability_scenarios = [r for r in stochastic_results if r.delay_cv_across_runs > 0.05]
    if high_variability_scenarios:
        # Sort by CV descending
        high_variability_scenarios.sort(key=lambda x: x.delay_cv_across_runs, reverse=True)

        for i, result in enumerate(high_variability_scenarios[:10], 1):  # Show top 10
            print(f"  {i}. {result.base_scenario_id}")
            print(f"     Model: {result.model_type}, Network: {result.network_type}")
            print(f"     Runs: {result.num_runs}, Delay CV: {result.delay_cv_across_runs:.4f}")
            print(f"     Delay Range: {result.delay_min_across_runs:.2f} - {result.delay_max_across_runs:.2f}ms")
            print(f"     Messages: {result.num_messages_mean:.0f} ± {result.num_messages_std:.1f}")

            if result.rmse_cv_across_runs is not None:
                print(f"     RMSE CV: {result.rmse_cv_across_runs:.4f}")

            if result.substitution_consistency is not None:
                print(f"     Substitution Rate: {result.substitution_consistency:.2f}")

            print()

        if len(high_variability_scenarios) > 10:
            print(f"  ... and {len(high_variability_scenarios) - 10} more high-variability scenarios")
    else:
        print("  No scenarios with high delay variability found.")

    # Meta-model substitution analysis
    meta_model_results = [r for r in stochastic_results if r.model_type == 'meta_model']
    if meta_model_results:
        print(f"\nMETA-MODEL SUBSTITUTION VARIABILITY:")
        print("-" * 50)

        substitution_consistencies = [r.substitution_consistency for r in meta_model_results
                                      if r.substitution_consistency is not None]

        if substitution_consistencies:
            print(f"  Substitution Rate Across Runs:")
            print(f"    Mean: {np.mean(substitution_consistencies):.3f}")
            print(f"    Std:  {np.std(substitution_consistencies):.3f}")
            print(f"    Range: {np.min(substitution_consistencies):.3f} - {np.max(substitution_consistencies):.3f}")

            # Classify substitution consistency
            always_substitute = len([c for c in substitution_consistencies if c == 1.0])
            never_substitute = len([c for c in substitution_consistencies if c == 0.0])
            sometimes_substitute = len(substitution_consistencies) - always_substitute - never_substitute

            print(f"  Substitution Patterns:")
            print(f"    Always substitute: {always_substitute} scenarios")
            print(f"    Never substitute: {never_substitute} scenarios")
            print(f"    Sometimes substitute: {sometimes_substitute} scenarios")

        substitution_time_stds = [r.substitution_time_std for r in meta_model_results
                                  if r.substitution_time_std is not None]
        if substitution_time_stds:
            print(f"  Substitution Time Variability (when it occurs):")
            print(f"    Mean Std: {np.mean(substitution_time_stds):.2f}s")
            print(f"    Range: {np.min(substitution_time_stds):.2f} - {np.max(substitution_time_stds):.2f}s")


def save_stochastic_analysis(stochastic_results: List[StochasticAnalysisResult], output_file: str):
    """Save stochastic analysis results to CSV."""
    if not stochastic_results:
        print("No stochastic analysis results to save.")
        return

    data = []
    for result in stochastic_results:
        row = {
            'base_scenario_id': result.base_scenario_id,
            'model_type': result.model_type,
            'network_type': result.network_type,
            'num_runs': result.num_runs,

            # Delay metrics
            'delay_mean_ms': result.delay_mean_across_runs,
            'delay_std_ms': result.delay_std_across_runs,
            'delay_cv': result.delay_cv_across_runs,
            'delay_min_ms': result.delay_min_across_runs,
            'delay_max_ms': result.delay_max_across_runs,
            'delay_range_ms': result.delay_range_across_runs,

            # RMSE/MAE metrics
            'rmse_mean_ms': result.rmse_mean_across_runs,
            'rmse_std_ms': result.rmse_std_across_runs,
            'rmse_cv': result.rmse_cv_across_runs,
            'mae_mean_ms': result.mae_mean_across_runs,
            'mae_std_ms': result.mae_std_across_runs,
            'mae_cv': result.mae_cv_across_runs,

            # Performance metrics
            'runtime_mean_s': result.runtime_mean_across_runs,
            'runtime_std_s': result.runtime_std_across_runs,
            'runtime_cv': result.runtime_cv_across_runs,
            'memory_mean_mb': result.memory_mean_across_runs,
            'memory_std_mb': result.memory_std_across_runs,
            'memory_cv': result.memory_cv_across_runs,

            # Message metrics
            'num_messages_mean': result.num_messages_mean,
            'num_messages_std': result.num_messages_std,
            'num_messages_consistent': result.num_messages_consistent,

            # Substitution metrics
            'substitution_consistency': result.substitution_consistency,
            'substitution_time_std_s': result.substitution_time_std,

            # Statistical tests
            'delay_anova_p_value': result.delay_anova_p_value,
            'is_stochastic_delay': result.is_stochastic_delay,

            # Derived metrics
            'high_delay_variability': result.delay_cv_across_runs > 0.05,
            'moderate_delay_variability': 0.01 < result.delay_cv_across_runs <= 0.05,
            'low_delay_variability': result.delay_cv_across_runs <= 0.01,
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Stochastic analysis results saved to: {output_file}")


def updated_parse_filename_to_config(filename: str):
    """
    Updated parse function to handle run numbers correctly.
    Add this to replace the existing parse_filename_to_config function.
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

        # Parse the scenario ID using the from_scenario_id method
        # This should handle the run number correctly now
        from integration_environment.scenario_configuration import ScenarioConfiguration
        return ScenarioConfiguration.from_scenario_id(scenario_id)

    except Exception as e:
        print(f"Warning: Could not parse filename '{filename}': {e}")
        return None


# Analyze regular results
results = analyze_results('results', 'analysis_results.csv')

# NEW: Analyze stochastic behavior
stochastic_results = analyze_stochastic_behavior(results)

# Analyze error scenarios
error_analysis = analyze_error_scenarios('results')

# Print all summaries
print_summary(results)

# NEW: Print stochastic analysis
if stochastic_results:
    print_stochastic_analysis(stochastic_results)
    save_stochastic_analysis(stochastic_results, 'stochastic_analysis.csv')
else:
    print("No stochastic analysis performed - no scenarios with multiple runs found.")

print_error_analysis(error_analysis)

# Save error analysis
save_error_analysis(error_analysis, 'error_analysis')
