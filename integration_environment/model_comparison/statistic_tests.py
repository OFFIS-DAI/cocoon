import pandas as pd
from pathlib import Path
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def load_results(phase: int):
    """Load the aggregated results CSV file"""
    results_path = Path(f"analysis_results/aggregated_results{phase}.csv")
    df = pd.read_csv(results_path)
    return df


def create_factor_coding(df):
    """
    Create coded factor variables (-1, 0, +1) for analysis
    """
    df_coded = df.copy()

    # Factor coding mappings based on your FCCCD design
    factor_mappings = {
        'cluster_distance_threshold': {1.0: -1, 3.0: 0, 5.0: 1},
        'batch_size_ipupa': {50: -1, 100: 0, 150: 1},
        'learning_rate_weighting': {0.1: -1, 0.5: 0, 0.9: 1},
        'butterfly_threshold_value': {0.1: -1, 0.5: 0, 0.9: 1},
        'substitution_priority': {'error_level': -1, 'none': 0, 'error_trend': 1}
    }

    # Create coded variables
    for factor, mapping in factor_mappings.items():
        if factor in df.columns:
            coded_col = f"{factor}_coded"
            df_coded[coded_col] = df_coded[factor].map(mapping)

            # Check for unmapped values
            unmapped = df_coded[df_coded[coded_col].isna()][factor].unique()
            if len(unmapped) > 0:
                print(f"Warning: Unmapped values for {factor}: {unmapped}")

    return df_coded


def simple_factor_effects_analysis(df, phase=1):
    """
    Simple one-way ANOVA analysis to determine factor effects on key responses
    """
    print("=" * 80)
    print("SIMPLE FACTOR EFFECTS ANALYSIS")
    print("=" * 80)

    # Filter to meta-model data only
    meta_data = df[df['model_type'] == 'meta_model'].copy()
    print(f"Analyzing {len(meta_data)} meta-model scenarios")

    if len(meta_data) == 0:
        print("No meta-model data found!")
        return

    # Create coded variables
    meta_data = create_factor_coding(meta_data)

    # Key response variables
    responses = {
        'nrmse_mean': 'NRMSE (mean)',
        'nrmse_std': 'NRMSE (std)',
        'mean_in_one_sigma_interval': 'Mean in one-sigma-interval',
        'substitution_message_index': 'Substitution Message Index',
        'execution_time_s': 'Execution time (s)',
        'score': 'Score'
    }

    # Factor variables (coded)
    factors = {
        'cluster_distance_threshold_coded': 'Cluster Distance Threshold',
        'batch_size_ipupa_coded': 'Batch Size (I-Pupa)',
        'learning_rate_weighting_coded': 'Learning Rate Weighting',
        'butterfly_threshold_value_coded': 'Butterfly Threshold Value',
        'substitution_priority_coded': 'Substitution Priority'
    }

    # Results storage
    significance_results = {}
    effect_directions = {}

    print(f"\nAnalyzing {len(responses)} response variables across {len(factors)} factors...\n")

    # Analyze each response variable
    for response_col, response_name in responses.items():
        if response_col not in meta_data.columns:
            print(f"Warning: {response_name} not found in data")
            continue

        print("=" * 60)
        print(f"RESPONSE: {response_name}")
        print("=" * 60)

        # Clean data for this response
        clean_data = meta_data.dropna(subset=[response_col])
        if len(clean_data) < 10:
            print(f"Insufficient data for {response_name} (n={len(clean_data)})")
            continue

        response_results = {}
        response_directions = {}

        # Test each factor
        for factor_col, factor_name in factors.items():
            if factor_col not in clean_data.columns:
                continue

            # Remove rows with missing factor values
            factor_data = clean_data.dropna(subset=[factor_col])
            if len(factor_data) < 10:
                continue

            try:
                # One-way ANOVA
                formula = f"{response_col} ~ C({factor_col})"
                model = ols(formula, data=factor_data).fit()
                anova_result = anova_lm(model, typ=1)

                p_value = anova_result['PR(>F)'].iloc[0]
                f_stat = anova_result['F'].iloc[0]

                # Store results
                response_results[factor_name] = {
                    'p_value': p_value,
                    'f_stat': f_stat,
                    'significant': p_value < 0.05
                }

                # Calculate means by factor level and effect direction
                means = factor_data.groupby(factor_col)[response_col].agg(['mean', 'std', 'count'])

                # Determine effect direction (for minimization objectives like RMSE, MAE, Memory)
                if response_col in ['rmse_ms', 'mae_ms', 'memory_avg_mb']:
                    # Lower is better
                    best_level = means['mean'].idxmin()
                    worst_level = means['mean'].idxmax()
                    effect_direction = "Minimize" if best_level != worst_level else "No clear direction"
                else:
                    # For substitution_message_index, depends on interpretation
                    # Assuming earlier substitution (lower index) is better
                    best_level = means['mean'].idxmin()
                    worst_level = means['mean'].idxmax()
                    effect_direction = "Earlier substitution" if best_level != worst_level else "No clear direction"

                response_directions[factor_name] = {
                    'means': means,
                    'best_level': best_level,
                    'worst_level': worst_level,
                    'effect_direction': effect_direction
                }

                # Print results
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "NS"

                print(f"{factor_name:30s}: F={f_stat:6.2f}, p={p_value:.4f} {significance}")

                # Show means if significant
                if p_value < 0.05:
                    print(f"  Factor levels (coded -> mean ± std):")
                    for level in sorted(means.index):
                        level_desc = {-1: "Low (-1)", 0: "Center (0)", 1: "High (+1)"}.get(level, f"Level {level}")
                        print(
                            f"    {level_desc:12s}: {means.loc[level, 'mean']:8.3f} ± {means.loc[level, 'std']:6.3f} (n={means.loc[level, 'count']})")

                    print()

            except Exception as e:
                print(f"{factor_name:30s}: Error - {str(e)}")

        # Store results for this response
        significance_results[response_name] = response_results
        effect_directions[response_name] = response_directions
        print()

    # Create summary table
    create_significance_summary(significance_results, effect_directions)

    return significance_results, effect_directions


def create_significance_summary(significance_results, effect_directions):
    """
    Create a summary table showing which factors are significant for which responses
    """
    print("=" * 80)
    print("SIGNIFICANCE SUMMARY TABLE")
    print("=" * 80)

    # Get all factors
    all_factors = set()
    for response_results in significance_results.values():
        all_factors.update(response_results.keys())
    all_factors = sorted(all_factors)

    # Create summary table
    print(f"{'Factor':<30s}", end="")
    responses = sorted(significance_results.keys())
    for response in responses:
        print(f"{response[:12]:<15s}", end="")
    print()
    print("-" * (30 + 15 * len(responses)))

    for factor in all_factors:
        print(f"{factor:<30s}", end="")
        for response in responses:
            if factor in significance_results[response]:
                p_val = significance_results[response][factor]['p_value']
                if p_val < 0.001:
                    symbol = "***"
                elif p_val < 0.01:
                    symbol = "**"
                elif p_val < 0.05:
                    symbol = "*"
                else:
                    symbol = "NS"
                print(f"{symbol:<15s}", end="")
            else:
                print(f"{'--':<15s}", end="")
        print()

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, NS = not significant")


def main(phase=1):
    """
    Main function to run the simple factor effects analysis
    """
    print("Loading results...")
    df = load_results(phase)

    print(f"Loaded {len(df)} total records")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")

    if phase == 1:
        # Run simple factor effects analysis
        simple_factor_effects_analysis(df)
    elif phase == 2:
        pass

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    # Run analysis for Phase 1
    main(phase=1)
