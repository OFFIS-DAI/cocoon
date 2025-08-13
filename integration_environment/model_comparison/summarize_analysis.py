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
    detailed_count = len(df[df['model_type'] == 'detailed'])
    metamodel_count = len(df[df['model_type'] == 'meta_model'])

    print(f"Total scenarios: {total_scenarios}")
    print(f"Detailed simulations: {detailed_count}")
    print(f"Meta-model simulations: {metamodel_count}")
    print()

    # Traffic configurations
    traffic_configs = df['traffic_config_clean'].unique()
    print(f"Traffic configurations: {len(traffic_configs)}")
    for config in sorted(traffic_configs):
        count = len(df[df['traffic_config_clean'] == config])
        print(f"  - {config}: {count} scenarios")
    print()


def analyze_delay_by_traffic(df: pd.DataFrame):
    """Analyze mean and std delay values by traffic configuration."""
    print("=" * 60)
    print("MEAN AND STD DELAY VALUES BY TRAFFIC CONFIG")
    print("=" * 60)

    for model_type in ['detailed', 'meta_model']:
        print(f"\n{model_type.upper()} SIMULATIONS:")
        model_df = df[df['model_type'] == model_type]

        if len(model_df) == 0:
            print(f"  No {model_type} simulations found")
            continue

        traffic_stats = model_df.groupby('traffic_config_clean').agg({
            'mean': ['mean', 'std', 'min', 'max'],
            'std': ['mean', 'std', 'min', 'max']
        }).round(2)

        print(f"{'Traffic Config':<30} {'Mean Delay':<15} {'Std Delay':<15}")
        print(f"{'':^30} {'Avg±Std':<15} {'Avg±Std':<15}")
        print("-" * 65)

        for traffic in traffic_stats.index:
            mean_avg = traffic_stats.loc[traffic, ('mean', 'mean')]
            mean_std = traffic_stats.loc[traffic, ('mean', 'std')]
            std_avg = traffic_stats.loc[traffic, ('std', 'mean')]
            std_std = traffic_stats.loc[traffic, ('std', 'std')]

            print(f"{traffic:<30} {mean_avg:.1f}±{mean_std:.1f}{'ms':<8} {std_avg:.1f}±{std_std:.1f}ms")
    print()


def analyze_variation_properties(df: pd.DataFrame):
    """Analyze variation properties including CV comparison."""
    print("=" * 60)
    print("VARIATION PROPERTIES")
    print("=" * 60)

    print("MEAN CV OF DETAILED SIMULATIONS (by traffic config):")
    detailed_df = df[df['model_type'] == 'detailed']

    if len(detailed_df) > 0:
        detailed_cv_stats = detailed_df.groupby('traffic_config_clean')['mean_message_cv'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(6)

        print(f"{'Traffic Config':<30} {'Mean CV':<20}")
        print("-" * 55)
        for traffic in detailed_cv_stats.index:
            cv_mean = detailed_cv_stats.loc[traffic, 'mean']
            cv_std = detailed_cv_stats.loc[traffic, 'std']
            print(f"{traffic:<30} {cv_mean:.6f}±{cv_std:.6f}")
    else:
        print("  No detailed simulations found")

    print("\nMEAN CV OF META-MODEL SIMULATIONS (by traffic config):")
    metamodel_df = df[df['model_type'] == 'meta_model']

    if len(metamodel_df) > 0:
        metamodel_cv_stats = metamodel_df.groupby('traffic_config_clean')['mean_message_cv'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(6)

        print(f"{'Traffic Config':<30} {'Mean CV':<20}")
        print("-" * 55)
        for traffic in metamodel_cv_stats.index:
            cv_mean = metamodel_cv_stats.loc[traffic, 'mean']
            cv_std = metamodel_cv_stats.loc[traffic, 'std']
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
    """Analyze accuracy comparison metrics."""
    print("=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)

    metamodel_df = df[df['model_type'] == 'meta_model']

    # Filter for valid accuracy data
    valid_accuracy = metamodel_df[
        metamodel_df['nrmse_mean'].notna() &
        metamodel_df['mean_in_one_sigma_interval'].notna()
        ]

    if len(valid_accuracy) == 0:
        print("No valid accuracy data found")
        return

    print("NRMSE VALUES BY TRAFFIC CONFIG:")
    print(f"{'Traffic Config':<30} {'NRMSE Mean':<15} {'NRMSE Std':<15}")
    print("-" * 65)

    nrmse_stats = valid_accuracy.groupby('traffic_config_clean').agg({
        'nrmse_mean': ['mean', 'std', 'min', 'max'],
        'nrmse_std': ['mean', 'std', 'min', 'max']
    }).round(4)

    for traffic in nrmse_stats.index:
        nrmse_mean_avg = nrmse_stats.loc[traffic, ('nrmse_mean', 'mean')]
        nrmse_mean_std = nrmse_stats.loc[traffic, ('nrmse_mean', 'std')]
        nrmse_std_avg = nrmse_stats.loc[traffic, ('nrmse_std', 'mean')]
        nrmse_std_std = nrmse_stats.loc[traffic, ('nrmse_std', 'std')]

        print(f"{traffic:<30} {nrmse_mean_avg:.4f}±{nrmse_mean_std:.4f}   {nrmse_std_avg:.4f}±{nrmse_std_std:.4f}")

    print("\nMEAN-IN-ONE-SIGMA VALUES BY TRAFFIC CONFIG:")
    print(f"{'Traffic Config':<30} {'Reliability %':<15}")
    print("-" * 50)

    sigma_stats = valid_accuracy.groupby('traffic_config_clean')['mean_in_one_sigma_interval'].agg([
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

    metamodel_df = df[df['model_type'] == 'meta_model']
    valid_data = metamodel_df[
        metamodel_df['nrmse_mean'].notna() &
        metamodel_df['mean_message_cv'].notna()
        ]

    if len(valid_data) == 0:
        print("No valid data for correlation analysis")
        return

    # Calculate correlation
    cv_nrmse_corr = valid_data['mean_message_cv'].corr(valid_data['nrmse_mean'])
    cv_reliability_corr = valid_data['mean_message_cv'].corr(valid_data['mean_in_one_sigma_interval'])

    print(f"Correlation between Message CV and NRMSE Mean: {cv_nrmse_corr:.4f}")
    print(f"Correlation between Message CV and Reliability: {cv_reliability_corr:.4f}")

    # Binned analysis
    print("\nBINNED ANALYSIS (by CV quartiles):")
    valid_data['cv_quartile'] = pd.qcut(valid_data['mean_message_cv'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    quartile_stats = valid_data.groupby('cv_quartile').agg({
        'mean_message_cv': ['mean', 'std'],
        'nrmse_mean': ['mean', 'std'],
        'mean_in_one_sigma_interval': ['mean', 'std']
    }).round(4)

    print(f"{'Quartile':<10} {'CV Range':<15} {'NRMSE Mean':<15} {'Reliability %':<15}")
    print("-" * 60)

    for quartile in quartile_stats.index:
        cv_mean = quartile_stats.loc[quartile, ('mean_message_cv', 'mean')]
        nrmse_mean = quartile_stats.loc[quartile, ('nrmse_mean', 'mean')]
        reliability = quartile_stats.loc[quartile, ('mean_in_one_sigma_interval', 'mean')] * 100

        print(f"{quartile:<10} {cv_mean:.6f}      {nrmse_mean:.4f}        {reliability:.1f}%")
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
    """Analyze performance comparison between detailed and meta-model with substitution."""
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    detailed_df = df[df['model_type'] == 'detailed']
    metamodel_df = df[df['model_type'] == 'meta_model']

    # Meta-model scenarios with substitution
    substitution_df = metamodel_df[
        (metamodel_df['substitution_occurred'] == True) |
        (metamodel_df['substitution_occurred'] == 'True')
        ]

    print("EXECUTION TIMES:")
    print(f"{'Scenario Type':<30} {'Count':<8} {'Mean Time (s)':<15} {'Std Time (s)':<15}")
    print("-" * 70)

    if len(detailed_df) > 0:
        detailed_mean = detailed_df['execution_time_s'].mean()
        detailed_std = detailed_df['execution_time_s'].std()
        print(f"{'Detailed simulations':<30} {len(detailed_df):<8} {detailed_mean:<15.1f} {detailed_std:<15.1f}")

    if len(metamodel_df) > 0:
        metamodel_mean = metamodel_df['execution_time_s'].mean()
        metamodel_std = metamodel_df['execution_time_s'].std()
        print(f"{'Meta-model (all)':<30} {len(metamodel_df):<8} {metamodel_mean:<15.1f} {metamodel_std:<15.1f}")

    if len(substitution_df) > 0:
        substitution_mean = substitution_df['execution_time_s'].mean()
        substitution_std = substitution_df['execution_time_s'].std()
        print(
            f"{'Meta-model (with substitution)':<30} {len(substitution_df):<8} {substitution_mean:<15.1f} {substitution_std:<15.1f}")

    # Calculate speedup
    if len(detailed_df) > 0 and len(substitution_df) > 0:
        speedup = detailed_df['execution_time_s'].mean() / substitution_df['execution_time_s'].mean()
        print(f"\nSPEEDUP ANALYSIS:")
        print(f"Meta-model with substitution vs. Detailed: {speedup:.2f}x faster")

    if len(detailed_df) > 0 and len(metamodel_df) > 0:
        overall_speedup = detailed_df['execution_time_s'].mean() / metamodel_df['execution_time_s'].mean()
        print(f"Meta-model (overall) vs. Detailed: {overall_speedup:.2f}x faster")

    # Performance by traffic configuration
    print(f"\nPERFORMANCE BY TRAFFIC CONFIGURATION:")
    print(f"{'Traffic Config':<30} {'Detailed (s)':<15} {'Meta-model (s)':<15} {'Speedup':<10}")
    print("-" * 75)

    for traffic in df['traffic_config_clean'].unique():
        detailed_traffic = detailed_df[detailed_df['traffic_config_clean'] == traffic]
        metamodel_traffic = metamodel_df[metamodel_df['traffic_config_clean'] == traffic]

        if len(detailed_traffic) > 0 and len(metamodel_traffic) > 0:
            detailed_time = detailed_traffic['execution_time_s'].mean()
            metamodel_time = metamodel_traffic['execution_time_s'].mean()
            speedup = detailed_time / metamodel_time

            print(f"{traffic:<30} {detailed_time:<15.1f} {metamodel_time:<15.1f} {speedup:<10.2f}x")
        elif len(metamodel_traffic) > 0:
            metamodel_time = metamodel_traffic['execution_time_s'].mean()
            print(f"{traffic:<30} {'N/A':<15} {metamodel_time:<15.1f} {'N/A':<10}")
    print()


def main():
    """Main analysis function."""
    PHASE = 1

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