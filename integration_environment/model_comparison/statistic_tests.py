import pandas as pd
from pathlib import Path
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import itertools


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


def two_way_anova_analysis(df):
    """
    Two-way ANOVA analysis for Phase 2: examining effects of scenario parameters and model_type
    """
    print("=" * 80)
    print("TWO-WAY ANOVA ANALYSIS - SCENARIO PARAMETERS vs MODEL TYPE")
    print("=" * 80)

    # Key response variables for Phase 2
    responses = {
        'mean': 'Mean Delay (ms)',
        'std': 'Std Delay (ms)',
        'mean_message_cv': 'Mean Message CV',
        'execution_time_s': 'Execution Time (s)',
        'nrmse_mean': 'NRMSE (mean)',
        'nrmse_std': 'NRMSE (std)',
        'mean_in_one_sigma_interval': 'Mean in one-sigma-interval'
    }

    # Scenario parameters
    scenario_factors = {
        'network_type': 'Network Type',
        'payload_size': 'Payload Size',
        'num_devices': 'Number of Devices',
        'traffic_configuration': 'Traffic Configuration'
    }

    # Clean column names for analysis
    df_clean = df.copy()
    for col in ['network_type', 'payload_size', 'num_devices', 'traffic_configuration']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(r'^[^.]*\.', '', regex=True)

    print(f"Analyzing {len(df_clean)} total scenarios")
    print(f"Model types: {df_clean['model_type'].value_counts().to_dict()}")
    print()

    # Results storage
    anova_results = {}

    # Analyze each response variable
    for response_col, response_name in responses.items():
        if response_col not in df_clean.columns:
            print(f"Warning: {response_name} not found in data")
            continue

        print("=" * 70)
        print(f"RESPONSE: {response_name}")
        print("=" * 70)

        # Clean data for this response
        clean_data = df_clean.dropna(subset=[response_col])
        if len(clean_data) < 20:
            print(f"Insufficient data for {response_name} (n={len(clean_data)})")
            continue

        response_results = {}

        # Test each scenario factor with model_type
        for factor_col, factor_name in scenario_factors.items():
            if factor_col not in clean_data.columns:
                continue

            # Remove rows with missing factor values
            factor_data = clean_data.dropna(subset=[factor_col, 'model_type'])
            if len(factor_data) < 20:
                print(f"Insufficient data for {factor_name} (n={len(factor_data)})")
                continue

            try:
                # Two-way ANOVA: Factor + Model_type + Interaction
                formula = f"{response_col} ~ C({factor_col}) + C(model_type) + C({factor_col}):C(model_type)"
                model = ols(formula, data=factor_data).fit()
                anova_result = anova_lm(model, typ=2)

                # Extract results for main effects and interaction
                results = {}
                for effect in anova_result.index:
                    if 'Residual' not in effect:
                        p_value = anova_result.loc[effect, 'PR(>F)']
                        f_stat = anova_result.loc[effect, 'F']

                        # Determine significance
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "NS"

                        results[effect] = {
                            'f_stat': f_stat,
                            'p_value': p_value,
                            'significance': significance
                        }

                response_results[factor_name] = results

                # Print results
                print(f"\n{factor_name}:")
                print(f"{'Effect':<40} {'F-stat':<10} {'p-value':<10} {'Sig':<5}")
                print("-" * 70)

                for effect, stats in results.items():
                    effect_name = effect.replace('C(', '').replace(')', '').replace(':', ' x ')
                    print(
                        f"{effect_name:<40} {stats['f_stat']:<10.2f} {stats['p_value']:<10.4f} {stats['significance']:<5}")

                # Show means for significant main effects
                if f"C({factor_col})" in results and results[f"C({factor_col})"]["p_value"] < 0.05:
                    print(f"\nMeans by {factor_name}:")
                    factor_means = factor_data.groupby(factor_col)[response_col].agg(['mean', 'std', 'count'])
                    for level in factor_means.index:
                        print(
                            f"  {level}: {factor_means.loc[level, 'mean']:.3f} ± {factor_means.loc[level, 'std']:.3f} (n={factor_means.loc[level, 'count']})")

                if "C(model_type)" in results and results["C(model_type)"]["p_value"] < 0.05:
                    print(f"\nMeans by Model Type:")
                    model_means = factor_data.groupby('model_type')[response_col].agg(['mean', 'std', 'count'])
                    for model in model_means.index:
                        print(
                            f"  {model}: {model_means.loc[model, 'mean']:.3f} ± {model_means.loc[model, 'std']:.3f} (n={model_means.loc[model, 'count']})")

                # Show interaction means if significant
                interaction_key = f"C({factor_col}):C(model_type)"
                if interaction_key in results and results[interaction_key]["p_value"] < 0.05:
                    print(f"\nInteraction means ({factor_name} x Model Type):")
                    interaction_means = factor_data.groupby([factor_col, 'model_type'])[response_col].agg(
                        ['mean', 'count'])
                    for (factor_level, model_type), stats in interaction_means.iterrows():
                        print(f"  {factor_level} x {model_type}: {stats['mean']:.3f} (n={stats['count']})")

            except Exception as e:
                print(f"Error analyzing {factor_name}: {str(e)}")

        # Store results for this response
        anova_results[response_name] = response_results
        print()

    # Create comprehensive summary
    create_two_way_anova_summary(anova_results)

    return anova_results


def create_two_way_anova_summary(anova_results):
    """
    Create a summary table for two-way ANOVA results
    """
    print("=" * 80)
    print("TWO-WAY ANOVA SUMMARY TABLE")
    print("=" * 80)

    # Get all scenario factors
    all_factors = set()
    for response_results in anova_results.values():
        all_factors.update(response_results.keys())
    all_factors = sorted(all_factors)

    responses = sorted(anova_results.keys())

    # Main effects summary
    print("\nMAIN EFFECTS SIGNIFICANCE:")
    print(f"{'Factor':<25}", end="")
    for response in responses:
        print(f"{response[:12]:<15}", end="")
    print()
    print("-" * (25 + 15 * len(responses)))

    for factor in all_factors:
        print(f"{factor:<25}", end="")
        for response in responses:
            if factor in anova_results[response]:
                factor_results = anova_results[response][factor]
                # Look for the main effect (not interaction)
                main_effect_key = None
                for key in factor_results.keys():
                    if ':' not in key and 'model_type' not in key and factor.lower().replace(' ', '_') in key.lower():
                        main_effect_key = key
                        break

                if main_effect_key and main_effect_key in factor_results:
                    sig = factor_results[main_effect_key]['significance']
                    print(f"{sig:<15}", end="")
                else:
                    print(f"{'--':<15}", end="")
            else:
                print(f"{'--':<15}", end="")
        print()

    # Model type effects summary
    print(f"\nMODEL TYPE EFFECTS:")
    print(f"{'Factor Context':<25}", end="")
    for response in responses:
        print(f"{response[:12]:<15}", end="")
    print()
    print("-" * (25 + 15 * len(responses)))

    for factor in all_factors:
        print(f"{factor:<25}", end="")
        for response in responses:
            if factor in anova_results[response]:
                factor_results = anova_results[response][factor]
                if "C(model_type)" in factor_results:
                    sig = factor_results["C(model_type)"]['significance']
                    print(f"{sig:<15}", end="")
                else:
                    print(f"{'--':<15}", end="")
            else:
                print(f"{'--':<15}", end="")
        print()

    # Interaction effects summary
    print(f"\nINTERACTION EFFECTS:")
    print(f"{'Factor x Model':<25}", end="")
    for response in responses:
        print(f"{response[:12]:<15}", end="")
    print()
    print("-" * (25 + 15 * len(responses)))

    for factor in all_factors:
        print(f"{factor}<25", end="")
        for response in responses:
            if factor in anova_results[response]:
                factor_results = anova_results[response][factor]
                # Look for interaction effect
                interaction_key = None
                for key in factor_results.keys():
                    if ':' in key and 'model_type' in key:
                        interaction_key = key
                        break

                if interaction_key and interaction_key in factor_results:
                    sig = factor_results[interaction_key]['significance']
                    print(f"{sig:<15}", end="")
                else:
                    print(f"{'--':<15}", end="")
            else:
                print(f"{'--':<15}", end="")
        print()

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, NS = not significant")


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
    Main function to run the appropriate statistical analysis
    """
    print("Loading results...")
    df = load_results(phase)

    print(f"Loaded {len(df)} total records")
    print(f"Model types: {df['model_type'].value_counts().to_dict()}")

    if phase == 1:
        # Run simple factor effects analysis for hyperparameters
        print("\nRunning hyperparameter effects analysis...")
        simple_factor_effects_analysis(df, phase)
    elif phase == 2:
        # Run two-way ANOVA for scenario parameters vs model types
        print("\nRunning two-way ANOVA analysis...")
        two_way_anova_analysis(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main(phase=1)