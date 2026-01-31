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
from typing import Dict, List, Tuple, Optional, Any

from scipy.stats import wasserstein_distance

from integration_environment.scenario_configuration import *


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    scenario_config: ScenarioConfiguration
    model_type: ModelType

    # data properties
    mean: Optional[float]
    std: Optional[float]
    mean_message_cv: Optional[float]
    mean_num_messages: Optional[int]

    # accuracy metrics
    nrmse_mean: Optional[float]
    nrmse_std: Optional[float]
    mean_in_sigma_interval: Optional[float]
    wasserstein_dist: Optional[float]

    # performance metric
    execution_time_s: float
    timeout_occurred: bool = False

    # substitution info (for meta-model)
    substitution_occurred: bool = False
    substitution_message_index: Optional[int] = None

    # score that indicates how well the hyper-parameter performed in this scenario
    score: Optional[int] = None


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
        network_type=config.network_type,
        prediction_model_type=PredictionModelType.none
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


def find_matching_ideal_simulations(config: ScenarioConfiguration, ideal_results: Dict[str, pd.DataFrame]) -> \
        List[pd.DataFrame]:
    """Find the ideal simulation result that matches the given configuration."""
    # Create an ideal version of the config
    ideal_config = ScenarioConfiguration(
        payload_size=config.payload_size,
        num_devices=config.num_devices,
        model_type=ModelType.ideal,
        scenario_duration=config.scenario_duration,
        traffic_configuration=config.traffic_configuration,
        network_type=NetworkModelType.none  # Ideal simulations use 'none' network type
    )
    matching_results = []
    run = 0
    while True:
        ideal_config.run = run
        ideal_scenario_id = ideal_config.scenario_id
        ideal_result = ideal_results.get(ideal_scenario_id)
        if ideal_result is not None:
            matching_results.append(ideal_result)
            run += 1
        else:
            return matching_results


def calculate_metrics_grouped(
        baseline_dfs: List[pd.DataFrame],
        model_dfs: List[pd.DataFrame],
        delay_cap_ms: int = 3000,
) -> Tuple[float, float, float, float]:
    """
    Compares baseline vs model delay distributions per message key, even if the number
    of runs differs between baseline_dfs and model_dfs.

    Returns:
      (nrmse_means, nrmse_std, mean_in_sigma_interval, mean_wasserstein)
    """
    MsgKey = Tuple[Any, Any, Any]  # (msg_id, sender, receiver)

    def collect_delays(dfs: List[pd.DataFrame]) -> Dict[MsgKey, List[float]]:
        delays_by_msg: Dict[MsgKey, List[float]] = {}
        for df in dfs:
            if df is None or df.empty:
                continue
            # avoid in-place mutation of caller's dataframes
            d = df.dropna(subset=["delay_ms"]).copy()
            d = d[d["delay_ms"] <= delay_cap_ms]

            if d.empty:
                continue

            indexed = d.set_index(["msg_id", "sender", "receiver"])["delay_ms"]
            for msg_key, delay in indexed.items():
                delays_by_msg.setdefault(msg_key, []).append(float(delay))
        return delays_by_msg

    baseline_delays_by_message = collect_delays(baseline_dfs)
    model_delays_by_message = collect_delays(model_dfs)

    common_messages = set(baseline_delays_by_message).intersection(model_delays_by_message)
    if not common_messages:
        raise ValueError("No common messages found between baseline and model across all runs")

    baseline_mean_delays, model_mean_delays = [], []
    baseline_std_delays, model_std_delays = [], []
    means_in_sigma_interval, wasserstein_distances = [], []

    for msg_key in common_messages:
        b = np.asarray(baseline_delays_by_message[msg_key], dtype=float)
        m = np.asarray(model_delays_by_message[msg_key], dtype=float)

        # Means / stds
        b_mean = float(np.mean(b))
        m_mean = float(np.mean(m))
        b_std = float(np.std(b, ddof=0))
        m_std = float(np.std(m, ddof=0))

        baseline_mean_delays.append(b_mean)
        model_mean_delays.append(m_mean)
        baseline_std_delays.append(b_std)
        model_std_delays.append(m_std)

        # One-sigma interval coverage (fraction of model samples within baseline mean±std)
        lo, hi = b_mean - abs(b_std), b_mean + abs(b_std)
        means_in_sigma_interval.append(float(np.mean((m >= lo) & (m <= hi))))

        # 1-Wasserstein (Earth Mover's Distance) between the two empirical samples
        wasserstein_distances.append(float(wasserstein_distance(b, m)))

    baseline_mean_delays = np.asarray(baseline_mean_delays, dtype=float)
    model_mean_delays = np.asarray(model_mean_delays, dtype=float)
    baseline_std_delays = np.asarray(baseline_std_delays, dtype=float)
    model_std_delays = np.asarray(model_std_delays, dtype=float)

    # NRMSE on means
    diff = model_mean_delays - baseline_mean_delays
    rmse_means = float(np.sqrt(np.mean(diff ** 2)))
    denom_means = float(np.mean(baseline_mean_delays))
    nrmse_means = rmse_means / denom_means if denom_means > 0 else float("inf")

    # NRMSE on stds (guard against denom==0)
    diff_std = model_std_delays - baseline_std_delays
    rmse_std = float(np.sqrt(np.mean(diff_std ** 2)))
    denom_std = float(np.mean(baseline_std_delays))
    nrmse_std = rmse_std / denom_std if denom_std > 0 else float("inf")

    return (
        nrmse_means,
        nrmse_std,
        float(np.mean(means_in_sigma_interval)),
        float(np.mean(wasserstein_distances)),
    )


def calculate_delay_statistics(dataframes: List[pd.DataFrame]) -> Tuple[float, float, float]:
    """Calculate mean, std, and mean coefficient of variation across all messages in all runs."""
    # Overall statistics across all messages
    all_delays = []
    for df in dataframes:
        all_delays.extend(df['delay_ms'].tolist())

    all_delays = np.array(all_delays)
    mean_delay = np.mean(all_delays)
    std_delay = np.std(all_delays)

    # Per-message coefficient of variation
    message_delays = {}
    for df in dataframes:
        indexed = df.set_index(['msg_id', 'sender', 'receiver'])['delay_ms']
        for msg_key, delay in indexed.items():
            if msg_key not in message_delays:
                message_delays[msg_key] = []
            message_delays[msg_key].append(delay)

    # Calculate CV for each message and take mean
    cvs = []
    for msg_key, delays in message_delays.items():
        delays = np.array(delays)
        msg_mean = np.mean(delays)
        msg_std = np.std(delays)
        if msg_mean > 0:
            cvs.append(msg_std / msg_mean)

    mean_cv = np.mean(cvs) if cvs else 0

    return mean_delay, std_delay, mean_cv


def calculate_hyperparameter_scores(evaluation_results: List[EvaluationResult]) -> None:
    """
    Calculate scores for hyperparameter configurations based on performance metrics.

    Score = (n-idx(nrmse_mean) + n-idx(nrmse_std) + idx(intv) + n-idx(execution_time)) * (substitution success)
    """
    # Filter for meta-model results with valid metrics
    meta_model_results = [r for r in evaluation_results if r.model_type == ModelType.meta_model]
    valid_results = [r for r in meta_model_results if
                     r.nrmse_mean is not None and r.nrmse_std is not None and
                     r.mean_in_sigma_interval is not None and r.execution_time_s is not None]

    if len(valid_results) == 0:
        print("No valid meta-model results for scoring")
        return

    # Create DataFrame for easier ranking
    df_data = []
    for result in valid_results:
        df_data.append({
            'result_obj': result,
            'nrmse_mean': result.nrmse_mean,
            'wasserstein_dist': result.wasserstein_dist,
            'mean_in_sigma_interval': result.mean_in_sigma_interval,
            'execution_time_s': result.execution_time_s,
            'substitution_occurred': result.substitution_occurred
        })

    df = pd.DataFrame(df_data)

    # Rank metrics (0-based indexing, 0 = best)
    # For nrmse_mean and nrmse_std: lower is better (ascending=True)
    # For mean_in_sigma_interval: higher is better (ascending=False)
    # For execution_time_s: lower is better (ascending=True)

    df['rank_nrmse_mean'] = df['nrmse_mean'].rank(method='min', ascending=True) - 1
    df['rank_interval'] = df['mean_in_sigma_interval'].rank(method='min', ascending=False) - 1
    df['rank_execution_time'] = df['execution_time_s'].rank(method='min', ascending=True) - 1
    df['rank_wasserstein_dist'] = df['wasserstein_dist'].rank(method='min', ascending=True) - 1

    max_rank = len(valid_results) - 1

    df['score_component'] = (
            (max_rank - df['rank_nrmse_mean']) +
            df['rank_interval'] +
            (max_rank - df['rank_wasserstein_dist']) +
            (max_rank - df['rank_execution_time'])
    )

    # Apply substitution multiplier (1 if successful, 0 if not)
    df['substitution_multiplier'] = df['substitution_occurred'].apply(lambda x: 1.0 if x else 0)
    df['final_score'] = df['score_component'] * df['substitution_multiplier']

    # Assign scores back to result objects, higher score is better
    for _, row in df.iterrows():
        row['result_obj'].score = int(round(row['final_score']))

    print(f"Calculated scores for {len(valid_results)} meta-model configurations")


def analyze_results(results_folder: str) -> List[EvaluationResult]:
    """
    Analyze all simulation results in the given folder.

    Args:
        results_folder: Path to folder containing CSV result files and JSON statistics files

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
    ideal_results = {}
    execution_runtime_data = {}
    timeout_data = {}
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

        if config.model_type == ModelType.meta_model and config.substitution == Substitution.disabled:
            continue

        # Store ideal simulations separately
        elif config.model_type == ModelType.ideal:
            ideal_results[scenario_id] = df

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
        timeout_data[scenario_id] = data.get('timeout_occurred', False)

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
            test_train_split=config.test_train_split,
            amount_of_scenarios_in_training_data=config.amount_of_scenarios_in_training_data,
            substitution=config.substitution,
            prediction_model_type=config.prediction_model_type,
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
                'timeout_occurred_data': [],
                'substitution_data': []
            }

        scenario_groups[base_scenario_id]['scenario_ids'].append(scenario_id)
        scenario_groups[base_scenario_id]['dataframes'].append(model_df)
        scenario_groups[base_scenario_id]['execution_times'].append(execution_runtime_data.get(scenario_id, 0.0))
        scenario_groups[base_scenario_id]['timeout_occurred_data'].append(timeout_data.get(scenario_id, False))
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
            timeout_data = group_data['timeout_occurred_data']
            substitution_data = group_data['substitution_data']

            # Calculate delay statistics for all scenarios
            mean_delay, std_delay, cv = calculate_delay_statistics(dataframes)

            if config.model_type in [ModelType.detailed, ModelType.ideal]:
                mean_execution_time = float(np.mean(execution_times))

                # For substitution data, use the first run's data (or aggregate as needed)
                substitution_occurred = any(sub['substitution_occurred'] for sub in substitution_data)
                substitution_message_index = next((sub['substitution_message_index'] for sub in substitution_data if
                                                   sub['substitution_message_index'] is not None), None)

                timeout_occurred = any(timeout_data)

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    mean=mean_delay,
                    std=std_delay,
                    mean_message_cv=cv,
                    mean_num_messages=np.mean([len(df) for df in dataframes]),
                    nrmse_mean=None,  # No RMSE for baseline models
                    nrmse_std=None,  # No MAE for baseline models
                    mean_in_sigma_interval=None,
                    wasserstein_dist=None,
                    execution_time_s=mean_execution_time,
                    timeout_occurred=timeout_occurred,
                    substitution_occurred=substitution_occurred,
                    substitution_message_index=substitution_message_index,
                )

            else:
                # For other model types, find matching baseline simulations and calculate accuracy metrics
                # First try to find detailed simulations as baseline
                baseline_dfs = find_matching_detailed_simulations(config, detailed_results)

                if len(baseline_dfs) == 0:
                    print(f"Warning: No matching baseline simulation found for {base_scenario_id}")
                    continue

                # Calculate overall metrics across all runs
                nrmse_means, nrmse_std, mean_in_one_sigma_interval, wasserstein_dist = calculate_metrics_grouped(
                    baseline_dfs, dataframes)

                mean_execution_time = float(np.mean(execution_times))

                # For substitution data, use aggregated information
                substitution_occurred = any(sub['substitution_occurred'] for sub in substitution_data)
                substitution_message_index = next((sub['substitution_message_index'] for sub in substitution_data if
                                                   sub['substitution_message_index'] is not None), None)

                timeout_occurred = any(timeout_data)

                result = EvaluationResult(
                    scenario_config=config,
                    model_type=config.model_type,
                    mean=mean_delay,
                    std=std_delay,
                    mean_message_cv=cv,
                    nrmse_mean=nrmse_means,
                    nrmse_std=nrmse_std,
                    mean_num_messages=np.mean([len(df) for df in dataframes]),
                    mean_in_sigma_interval=mean_in_one_sigma_interval,
                    wasserstein_dist=wasserstein_dist,
                    execution_time_s=mean_execution_time,
                    timeout_occurred=timeout_occurred,
                    substitution_occurred=substitution_occurred,
                    substitution_message_index=substitution_message_index,
                )

            evaluation_results.append(result)

        except Exception as e:
            print(f"Error processing {base_scenario_id}: {e}")
            continue

    if phase == 1 or phase == 3:
        # Calculate hyperparameter scores
        calculate_hyperparameter_scores(evaluation_results)

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

            # Data properties
            'mean': result.mean,
            'std': result.std,
            'mean_message_cv': result.mean_message_cv,

            # Accuracy metrics
            'nrmse_mean': result.nrmse_mean,
            'nrmse_std': result.nrmse_std,
            'mean_in_one_sigma_interval': result.mean_in_sigma_interval,
            'wasserstein_distance': result.wasserstein_dist,

            # Performance metrics
            'execution_time_s': result.execution_time_s,
            'timeout_occurred': result.timeout_occurred,

            # Meta-model specific metrics
            'substitution_occurred': result.substitution_occurred,
            'substitution_message_index': result.substitution_message_index,
            'score': result.score,
        }

        # Add scenario configuration details if requested
        if include_scenario_details:
            row.update({
                'payload_size': result.scenario_config.payload_size,
                'num_devices': result.scenario_config.num_devices,
                'scenario_duration': result.scenario_config.scenario_duration,
                'traffic_configuration': result.scenario_config.traffic_configuration,
                'network_type': result.scenario_config.network_type,
                'mean_number_of_messages': result.mean_num_messages,
                'prediction_model_type': result.scenario_config.prediction_model_type,

                # Factor values (raw)
                'cluster_distance_threshold': result.scenario_config.cluster_distance_threshold.value if result.scenario_config.cluster_distance_threshold else None,
                'batch_size_ipupa': result.scenario_config.i_pupa.value if result.scenario_config.i_pupa else None,
                'learning_rate_weighting': result.scenario_config.learning_rate_weighting.value if result.scenario_config.learning_rate_weighting else None,
                'butterfly_threshold_value': result.scenario_config.butterfly_threshold_value.value if result.scenario_config.butterfly_threshold_value else None,
                'substitution_priority': result.scenario_config.substitution_priority.value if result.scenario_config.substitution_priority else None,
                'test_train': result.scenario_config.test_train_split.value if result.scenario_config.test_train_split else None,

                # Factor names (for analysis)
                'cluster_distance_threshold_name': result.scenario_config.cluster_distance_threshold.name if result.scenario_config.cluster_distance_threshold else None,
                'batch_size_ipupa_name': result.scenario_config.i_pupa.name if result.scenario_config.i_pupa else None,
                'learning_rate_weighting_name': result.scenario_config.learning_rate_weighting.name if result.scenario_config.learning_rate_weighting else None,
                'butterfly_threshold_value_name': result.scenario_config.butterfly_threshold_value.name if result.scenario_config.butterfly_threshold_value else None,
                'substitution_priority_name': result.scenario_config.substitution_priority.name if result.scenario_config.substitution_priority else None,
                'test_train_name': result.scenario_config.test_train_split.name if result.scenario_config.test_train_split else None,

                # Additional meta-model flags
                'substitution': result.scenario_config.substitution.value if result.scenario_config.substitution else None,
                'substitution_name': result.scenario_config.substitution.name if result.scenario_config.substitution else None,
                'amount_of_scenarios_in_training_data': result.scenario_config.amount_of_scenarios_in_training_data.value if result.scenario_config.amount_of_scenarios_in_training_data else None,
                'amount_of_scenarios_in_training_data_name': result.scenario_config.amount_of_scenarios_in_training_data.name if result.scenario_config.amount_of_scenarios_in_training_data else None,
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
    phase = 2

    # Create output directory
    Path('analysis_results').mkdir(exist_ok=True)

    if phase is not None:
        # Analyze results and save to CSV
        results = analyze_results_with_csv_export(
            f'results/phase{phase}',
            f'analysis_results/aggregated_results{phase}.csv'
        )

        print(f"Analysis complete. Processed {len(results)} scenarios.")

        # Print basic statistics
        model_counts = {}
        for r in results:
            model_type = r.model_type.value
            model_counts[model_type] = model_counts.get(model_type, 0) + 1

        print(f"\nModel distribution:")
        for model_type, count in model_counts.items():
            print(f"- {count} {model_type} simulations")

        if phase == 1 or phase==3:
            # Print top scoring hyperparameter configurations
            metamodel_results = [r for r in results if r.model_type == ModelType.meta_model and r.score is not None]
            if metamodel_results:
                top_configs = sorted(metamodel_results, key=lambda x: x.score, reverse=True)[:5]
                print(f"\nTop 5 hyperparameter configurations by score:")
                for i, config in enumerate(top_configs, 1):
                    print(f"{i}. Score: {config.score}")
                    print(f"   Config: {config.scenario_config.cluster_distance_threshold}-"
                          f"{config.scenario_config.i_pupa}-"
                          f"{config.scenario_config.learning_rate_weighting}-"
                          f"{config.scenario_config.butterfly_threshold_value}-"
                          f"{config.scenario_config.substitution_priority}-"
                          f"{config.scenario_config.test_train_split}")
                    print(f"   NRMSE: {config.nrmse_mean:.4f}, Interval: {config.mean_in_sigma_interval:.3f}, "
                          f"Time: {config.execution_time_s:.1f}s, Substitution: {config.substitution_occurred}")

    if phase is None:
        eval_results = analyze_results('results/minimal', phase=1)
        save_evaluation_results_to_csv(evaluation_results=eval_results,
                                       output_file='analysis_results/minimal_analysis.csv',
                                       include_scenario_details=True)
