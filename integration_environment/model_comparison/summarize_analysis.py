#!/usr/bin/env python3
"""
Communication Model Analysis Summary Script

This script analyzes the aggregated results from communication model evaluation
and extracts key metrics for comparison between detailed and meta-model approaches.
"""

import pandas as pd
from pathlib import Path


def load_data(csv_file: str) -> pd.DataFrame:
    """Load and prepare the aggregated results data."""
    df = pd.read_csv(csv_file)

    # Clean traffic configuration names for better readability
    df['traffic_config_clean'] = df['traffic_configuration'].str.replace('TrafficConfig.', '')

    return df


def analyze_general_properties(df: pd.DataFrame):
    """Analyze general data properties."""
    print("=" * 60)
    print("GENERAL DATA PROPERTIES")
    print("=" * 60)

    total_scenarios = len(df)
    model_counts = df['model_type'].value_counts()

    print(f"Total scenarios: {total_scenarios}")
    print("Model type distribution:")
    for model_type, count in model_counts.items():
        print(f"  {model_type}: {count}")
    print()

    # Traffic configurations
    traffic_configs = df['traffic_config_clean'].unique()
    print(f"Traffic configurations: {len(traffic_configs)}")
    for config in sorted(traffic_configs):
        count = len(df[df['traffic_config_clean'] == config])
        print(f"  - {config}: {count} scenarios")
    print()


def analyze_delay_by_traffic(df: pd.DataFrame):
    """Analyze mean and std delay values by traffic configuration and network type."""
    print("=" * 60)
    print("MEAN AND STD DELAY VALUES BY TRAFFIC CONFIG AND NETWORK TYPE")
    print("=" * 60)

    model_types = df['model_type'].unique()

    for model_type in sorted(model_types):
        print(f"\n{model_type.upper()} SIMULATIONS:")
        model_df = df[df['model_type'] == model_type]

        if len(model_df) == 0:
            print(f"  No {model_type} simulations found")
            continue

        # Filter out rows where mean is NaN
        valid_model_df = model_df[model_df['mean'].notna()]
        if len(valid_model_df) == 0:
            print(f"  No valid delay data for {model_type} simulations")
            continue

        # Group by both traffic config and network type
        traffic_network_stats = valid_model_df.groupby(['traffic_config_clean', 'network_type']).agg({
            'mean': ['mean', 'std', 'min', 'max'],
            'std': ['mean', 'std', 'min', 'max']
        }).round(2)

        print(f"{'Traffic Config':<30} {'Network Type':<20} {'Mean Delay':<15} {'Std Delay':<15}")
        print(f"{'':^30} {'':^20} {'Avg±Std':<15} {'Avg±Std':<15}")
        print("-" * 85)

        for (traffic, network), stats in traffic_network_stats.iterrows():
            mean_avg = stats[('mean', 'mean')]
            mean_std = stats[('mean', 'std')]
            std_avg = stats[('std', 'mean')]
            std_std = stats[('std', 'std')]

            # Clean network type name for display
            network_clean = str(network).replace('NetworkModelType.', '')

            print(
                f"{traffic:<30} {network_clean:<20} {mean_avg:.1f}±{mean_std:.1f}{'ms':<8} {std_avg:.1f}±{std_std:.1f}ms")
    print()


def analyze_variation_properties(df: pd.DataFrame):
    """Analyze variation properties including CV comparison."""
    print("=" * 60)
    print("VARIATION PROPERTIES")
    print("=" * 60)

    model_types = df['model_type'].unique()

    for model_type in sorted(model_types):
        print(f"\nMEAN CV OF {model_type.upper()} SIMULATIONS (by traffic config):")
        model_df = df[df['model_type'] == model_type]

        if len(model_df) == 0:
            print(f"  No {model_type} simulations found")
            continue

        # Filter out rows where mean_message_cv is NaN
        valid_model_df = model_df[model_df['mean_message_cv'].notna()]
        if len(valid_model_df) == 0:
            print(f"  No valid CV data for {model_type} simulations")
            continue

        cv_stats = valid_model_df.groupby('traffic_config_clean')['mean_message_cv'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(6)

        print(f"{'Traffic Config':<30} {'Mean CV':<20}")
        print("-" * 55)
        for traffic in cv_stats.index:
            cv_mean = cv_stats.loc[traffic, 'mean']
            cv_std = cv_stats.loc[traffic, 'std']
            print(f"{traffic:<30} {cv_mean:.6f}±{cv_std:.6f}")
    print()


def analyze_metamodel_properties(df: pd.DataFrame):
    """Analyze meta-model specific properties."""
    print("=" * 60)
    print("META-MODEL PROPERTIES")
    print("=" * 60)

    metamodel_df = df[df['model_type'] == 'meta_model']

    if len(metamodel_df) == 0:
        print("No meta-model simulations found")
        return

    # Substitution analysis
    substitution_scenarios = metamodel_df[
        (metamodel_df['substitution_occurred'] == True) |
        (metamodel_df['substitution_occurred'] == 'True')
        ]

    substitution_percentage = len(substitution_scenarios) / len(metamodel_df) * 100

    print(f"PERCENTAGE OF SCENARIOS WITH SUBSTITUTION:")
    print(f"  Overall: {substitution_percentage:.1f}% ({len(substitution_scenarios)}/{len(metamodel_df)})")

    # By traffic config
    print(f"\nBy traffic configuration:")
    for traffic in metamodel_df['traffic_config_clean'].unique():
        traffic_df = metamodel_df[metamodel_df['traffic_config_clean'] == traffic]
        traffic_substitutions = traffic_df[
            (traffic_df['substitution_occurred'] == True) |
            (traffic_df['substitution_occurred'] == 'True')
            ]
        traffic_percentage = len(traffic_substitutions) / len(traffic_df) * 100
        print(f"  {traffic}: {traffic_percentage:.1f}% ({len(traffic_substitutions)}/{len(traffic_df)})")

    # Mean substitution message index
    if len(substitution_scenarios) > 0:
        valid_indices = substitution_scenarios['substitution_message_index'].dropna()
        if len(valid_indices) > 0:
            mean_index = valid_indices.mean()
            std_index = valid_indices.std()
            print(f"\nMEAN MESSAGE INDEX OF SUBSTITUTION:")
            print(f"  Overall: {mean_index:.1f}±{std_index:.1f}")

            print(f"\nBy traffic configuration:")
            for traffic in substitution_scenarios['traffic_config_clean'].unique():
                traffic_indices = substitution_scenarios[
                    substitution_scenarios['traffic_config_clean'] == traffic
                    ]['substitution_message_index'].dropna()
                if len(traffic_indices) > 0:
                    print(f"  {traffic}: {traffic_indices.mean():.1f}±{traffic_indices.std():.1f}")
    print()


def analyze_accuracy_comparison(df: pd.DataFrame):
    """Analyze accuracy comparison metrics for all approximation models."""
    print("=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)

    # Find all models that have accuracy metrics (non-baseline models)
    approximation_models = df[df['nrmse_mean'].notna()]

    if len(approximation_models) == 0:
        print("No valid accuracy data found")
        return

    model_types = approximation_models['model_type'].unique()

    for model_type in sorted(model_types):
        model_df = approximation_models[approximation_models['model_type'] == model_type]

        print(f"\n{model_type.upper()} - NRMSE VALUES BY TRAFFIC CONFIG:")
        print(f"{'Traffic Config':<30} {'NRMSE Mean':<15} {'NRMSE Std':<15}")
        print("-" * 65)

        nrmse_stats = model_df.groupby('traffic_config_clean').agg({
            'nrmse_mean': ['mean', 'std', 'min', 'max'],
            'nrmse_std': ['mean', 'std', 'min', 'max']
        }).round(4)

        for traffic in nrmse_stats.index:
            nrmse_mean_avg = nrmse_stats.loc[traffic, ('nrmse_mean', 'mean')]
            nrmse_mean_std = nrmse_stats.loc[traffic, ('nrmse_mean', 'std')]
            nrmse_std_avg = nrmse_stats.loc[traffic, ('nrmse_std', 'mean')]
            nrmse_std_std = nrmse_stats.loc[traffic, ('nrmse_std', 'std')]

            print(f"{traffic:<30} {nrmse_mean_avg:.4f}±{nrmse_mean_std:.4f}   {nrmse_std_avg:.4f}±{nrmse_std_std:.4f}")

        print(f"\n{model_type.upper()} - MEAN-IN-ONE-SIGMA VALUES BY TRAFFIC CONFIG:")
        print(f"{'Traffic Config':<30} {'Reliability %':<15}")
        print("-" * 50)

        sigma_stats = model_df.groupby('traffic_config_clean')['mean_in_one_sigma_interval'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        for traffic in sigma_stats.index:
            sigma_mean = sigma_stats.loc[traffic, 'mean'] * 100
            sigma_std = sigma_stats.loc[traffic, 'std'] * 100
            print(f"{traffic:<30} {sigma_mean:.1f}±{sigma_std:.1f}%")
    print()


def analyze_variability_vs_accuracy(df: pd.DataFrame):
    """Analyze dependency between variability of delays and prediction accuracy."""
    print("=" * 60)
    print("DEPENDENCY: VARIABILITY OF DELAYS VS. PREDICTION ACCURACY")
    print("=" * 60)

    # Analyze for each approximation model type
    approximation_models = df[df['nrmse_mean'].notna()]
    model_types = approximation_models['model_type'].unique()

    for model_type in sorted(model_types):
        print(f"\n{model_type.upper()} MODEL:")
        model_df = approximation_models[approximation_models['model_type'] == model_type]
        valid_data = model_df[
            model_df['nrmse_mean'].notna() &
            model_df['mean_message_cv'].notna()
            ]

        if len(valid_data) == 0:
            print("  No valid data for correlation analysis")
            continue

        # Calculate correlation
        cv_nrmse_corr = valid_data['mean_message_cv'].corr(valid_data['nrmse_mean'])
        cv_reliability_corr = valid_data['mean_message_cv'].corr(valid_data['mean_in_one_sigma_interval'])

        print(f"  Correlation between Message CV and NRMSE Mean: {cv_nrmse_corr:.4f}")
        print(f"  Correlation between Message CV and Reliability: {cv_reliability_corr:.4f}")
    print()


def analyze_best_hyperparameters(df: pd.DataFrame):
    """Analyze best hyperparameter configurations by score for different traffic configs."""
    print("=" * 60)
    print("BEST HYPERPARAMETERS BY TRAFFIC CONFIGURATION")
    print("=" * 60)

    metamodel_df = df[df['model_type'] == 'meta_model']
    scored_df = metamodel_df[metamodel_df['score'].notna()]

    if len(scored_df) == 0:
        print("No scored hyperparameter configurations found")
        return

    print("BEST SCORING HYPERPARAMETER CONFIGURATION PER TRAFFIC TYPE:")
    print()

    for traffic in sorted(scored_df['traffic_config_clean'].unique()):
        traffic_df = scored_df[scored_df['traffic_config_clean'] == traffic]

        if len(traffic_df) == 0:
            continue

        # Find best scoring configuration
        best_config = traffic_df.loc[traffic_df['score'].idxmax()]

        print(f"Traffic: {traffic}")
        print(f"  Best Score: {best_config['score']}")
        print(f"  Configuration:")
        print(f"    - Cluster Distance Threshold: {best_config.get('cluster_distance_threshold_name', 'N/A')}")
        print(f"    - Batch Size I-PUPA: {best_config.get('batch_size_ipupa_name', 'N/A')}")
        print(f"    - Learning Rate Weighting: {best_config.get('learning_rate_weighting_name', 'N/A')}")
        print(f"    - Butterfly Threshold: {best_config.get('butterfly_threshold_value_name', 'N/A')}")
        print(f"  Performance Metrics:")
        print(f"    - NRMSE Mean: {best_config.get('nrmse_mean', 'N/A'):.4f}")
        print(f"    - NRMSE Std: {best_config.get('nrmse_std', 'N/A'):.4f}")
        print(f"    - Reliability: {best_config.get('mean_in_one_sigma_interval', 0) * 100:.1f}%")
        print(f"    - Execution Time: {best_config.get('execution_time_s', 'N/A'):.1f}s")
        print(f"    - Substitution Occurred: {best_config.get('substitution_occurred', 'N/A')}")
        print()

    # Overall best configuration by mean score across traffic types
    print("BEST CONFIGURATION BY MEAN SCORE ACROSS TRAFFIC TYPES:")

    # Group by hyperparameter configuration and calculate mean scores
    config_columns = ['cluster_distance_threshold_name', 'batch_size_ipupa_name',
                      'learning_rate_weighting_name', 'butterfly_threshold_value_name', 'substitution_priority']

    # Create configuration identifier
    scored_df['config_id'] = scored_df[config_columns].fillna('N/A').agg('-'.join, axis=1)

    # Calculate mean score for each configuration across all traffic types
    config_mean_scores = scored_df.groupby('config_id').agg({
        'score': ['mean', 'std', 'count'],
        'nrmse_mean': 'mean',
        'nrmse_std': 'mean',
        'mean_in_one_sigma_interval': 'mean',
        'execution_time_s': 'mean',
        'substitution_occurred': lambda x: sum(x == True) / len(x)
    }).round(4)

    # Find configuration with highest mean score
    best_mean_config_id = config_mean_scores[('score', 'mean')].idxmax()
    best_mean_score = config_mean_scores.loc[best_mean_config_id, ('score', 'mean')]
    score_std = config_mean_scores.loc[best_mean_config_id, ('score', 'std')]

    print(f"  Configuration: {best_mean_config_id}")
    print(f"  Mean Score: {best_mean_score:.2f} ± {score_std:.2f}")
    print(f"  Average Performance Metrics:")
    print(f"    - NRMSE Mean: {config_mean_scores.loc[best_mean_config_id, ('nrmse_mean', 'mean')]:.4f}")
    print(f"    - NRMSE Std: {config_mean_scores.loc[best_mean_config_id, ('nrmse_std', 'mean')]:.4f}")
    print(
        f"    - Reliability: {config_mean_scores.loc[best_mean_config_id, ('mean_in_one_sigma_interval', 'mean')] * 100:.1f}%")
    print(f"    - Execution Time: {config_mean_scores.loc[best_mean_config_id, ('execution_time_s', 'mean')]:.1f}s")
    print(
        f"    - Substitution Success Rate: {config_mean_scores.loc[best_mean_config_id, ('substitution_occurred', '<lambda>')] * 100:.1f}%")
    print()


def analyze_performance_comparison(df: pd.DataFrame):
    """Analyze performance comparison between all model types."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    model_types = df['model_type'].unique()
    baseline_models = ['detailed', 'ideal']
    approximation_models = [m for m in model_types if m not in baseline_models]

    print("EXECUTION TIMES:")
    print(f"{'Model Type':<30} {'Count':<8} {'Mean Time (s)':<15} {'Std Time (s)':<15}")
    print("-" * 70)

    # Show baseline models first
    for model_type in sorted(baseline_models):
        if model_type in model_types:
            model_df = df[df['model_type'] == model_type]
            if len(model_df) > 0:
                mean_time = model_df['execution_time_s'].mean()
                std_time = model_df['execution_time_s'].std()
                print(f"{model_type:<30} {len(model_df):<8} {mean_time:<15.1f} {std_time:<15.1f}")

    # Show approximation models
    for model_type in sorted(approximation_models):
        model_df = df[df['model_type'] == model_type]
        if len(model_df) > 0:
            mean_time = model_df['execution_time_s'].mean()
            std_time = model_df['execution_time_s'].std()
            print(f"{model_type:<30} {len(model_df):<8} {mean_time:<15.1f} {std_time:<15.1f}")

    # Meta-model with substitution (if exists)
    if 'meta_model' in model_types:
        metamodel_df = df[df['model_type'] == 'meta_model']
        substitution_df = metamodel_df[
            (metamodel_df['substitution_occurred'] == True) |
            (metamodel_df['substitution_occurred'] == 'True')
            ]

        if len(substitution_df) > 0:
            substitution_mean = substitution_df['execution_time_s'].mean()
            substitution_std = substitution_df['execution_time_s'].std()
            print(
                f"{'meta_model (with substitution)':<30} {len(substitution_df):<8} {substitution_mean:<15.1f} {substitution_std:<15.1f}")

    # Calculate speedups relative to detailed simulations
    detailed_df = df[df['model_type'] == 'detailed']
    if len(detailed_df) > 0:
        detailed_time = detailed_df['execution_time_s'].mean()

        print(f"\nSPEEDUP ANALYSIS (vs. Detailed):")
        for model_type in sorted(approximation_models):
            model_df = df[df['model_type'] == model_type]
            if len(model_df) > 0:
                model_time = model_df['execution_time_s'].mean()
                speedup = detailed_time / model_time
                print(f"  {model_type}: {speedup:.2f}x faster")

    # Performance by traffic configuration
    print(f"\nPERFORMANCE BY TRAFFIC CONFIGURATION:")
    traffic_configs = df['traffic_config_clean'].unique()

    for traffic in sorted(traffic_configs):
        print(f"\nTraffic: {traffic}")
        print(f"{'Model Type':<20} {'Exec Time (s)':<15} {'Count':<8}")
        print("-" * 45)

        for model_type in sorted(model_types):
            model_traffic = df[(df['model_type'] == model_type) & (df['traffic_config_clean'] == traffic)]
            if len(model_traffic) > 0:
                mean_time = model_traffic['execution_time_s'].mean()
                print(f"{model_type:<20} {mean_time:<15.1f} {len(model_traffic):<8}")
    print()


def main():
    """Main analysis function."""
    PHASE = 2

    csv_file = f"analysis_results/aggregated_results{PHASE}.csv"

    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return

    # Load data
    df = load_data(csv_file)

    print("COMMUNICATION MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()

    # Run all analyses
    analyze_general_properties(df)
    analyze_delay_by_traffic(df)
    analyze_variation_properties(df)
    analyze_metamodel_properties(df)
    analyze_accuracy_comparison(df)
    analyze_variability_vs_accuracy(df)
    analyze_best_hyperparameters(df)
    analyze_performance_comparison(df)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()