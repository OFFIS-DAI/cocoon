import asyncio

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from integration_environment.model_comparison.analyze_results import analyze_results, save_evaluation_results_to_csv
from integration_environment.model_comparison.execute_comparison import run_scenario_config, \
    get_central_composite_design
from integration_environment.scenario_configuration import *


async def run_minimal_comparison():
    num_repetitions = 2

    payload_size = PayloadSizeConfig.medium
    n_devices = NumDevices.five
    network = NetworkModelType.simbench_5g

    scenario_configs = []

    # use three different traffic configs, all other params stay constant
    for scenario_duration, traffic_config in [(ScenarioDuration.one_min, TrafficConfig.cbr_broadcast_1_mps),
                                              (ScenarioDuration.one_min, TrafficConfig.poisson_broadcast_1_mps_1),
                                              (ScenarioDuration.one_hour, TrafficConfig.central_dsb_1mpm_5s_50p)
                                              ]:
        for model_type in [#ModelType.detailed,
                           #ModelType.ideal,
                           ModelType.meta_model,
                           #ModelType.channel,
                           #ModelType.static_graph,
                           ]:
            if model_type == ModelType.meta_model:
                for lr in [LearningRateWeighting.small,
                           #LearningRateWeighting.large_medium
                           ]:
                    for btv in [ButterflyThresholdValue.small,
                                #ButterflyThresholdValue.center,
                                ButterflyThresholdValue.large]:
                        for cdt in [ClusterDistanceThreshold.three,
                                    #ClusterDistanceThreshold.five
                                    ]:
                            for tts in [TestTrainSplit.parametrization_split,
                                        TestTrainSplit.technology_split,
                                        TestTrainSplit.scale_split,
                                        TestTrainSplit.traffic_model_split,
                                        TestTrainSplit.traffic_load_split]:
                                scenario_configs.append(
                                    ScenarioConfiguration(payload_size=payload_size,
                                                          num_devices=n_devices,
                                                          model_type=ModelType.meta_model,
                                                          scenario_duration=scenario_duration,
                                                          traffic_configuration=traffic_config,
                                                          network_type=network,
                                                          cluster_distance_threshold=cdt,
                                                          i_pupa=BatchSizeIPupa.hundred,
                                                          learning_rate_weighting=lr,
                                                          butterfly_threshold_value=btv,
                                                          substitution_priority=SubstitutionPriority.none,
                                                          test_train_split=tts))
            else:
                scenario_configs.append(
                    ScenarioConfiguration(payload_size=payload_size,
                                          num_devices=n_devices,
                                          model_type=model_type,
                                          scenario_duration=scenario_duration,
                                          traffic_configuration=traffic_config,
                                          network_type=network))

    print(f'Run {num_repetitions * len(scenario_configs)} scenarios. ')
    for r in range(num_repetitions):
        for scenario_config in scenario_configs:
            await run_scenario_config(scenario_configuration=scenario_config, run=r,
                                      phase=None, timeout_seconds=60 * 20, output_dir='minimal')


def conduct_minimal_analysis():
    eval_results = analyze_results('results/minimal', phase=1)
    save_evaluation_results_to_csv(evaluation_results=eval_results, output_file='analysis_results/minimal_analysis.csv',
                                   include_scenario_details=True)


if __name__ == "__main__":
    asyncio.run(run_minimal_comparison())
    conduct_minimal_analysis()
