import os
from typing import Optional

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from variable import PATH

os.chdir(PATH)


def get_state_df(technology: Optional[str] = None,
                 object_variables: list[str] = None):
    state_data_path = os.path.join(PATH, 'state_data')

    if technology:
        # Load specific technology file
        file_path = os.path.join(state_data_path, f'NORMAL_Tech_{technology}.pkl')
        if os.path.exists(file_path):
            state_df = pd.read_pickle(file_path)
        else:
            raise FileNotFoundError(f"File {file_path} not found.")
    else:
        # Load all files in the folder and concatenate them
        all_files = [os.path.join(state_data_path, f) for f in os.listdir(state_data_path) if f.endswith('.pkl')]

        if not all_files:
            raise FileNotFoundError("No state data files found in the directory.")

        dataframes = [pd.read_pickle(f) for f in all_files]
        state_df = pd.concat(dataframes, ignore_index=True)
    if object_variables:
        state_df[object_variables] = state_df[object_variables].fillna(0)
    else:
        state_df.fillna(0, inplace=True)
    return state_df

def filter_by_delay_and_num_messages(state_df: pd.DataFrame,
                                     minimum_number_of_messages_per_scenario,
                                     delay_threshold):
    # Filter out scenarios with less than given minimum of messages
    scenario_counts = state_df['scenario_name'].value_counts()

    valid_scenarios = scenario_counts[scenario_counts >= minimum_number_of_messages_per_scenario].index
    state_df = state_df[state_df['scenario_name'].isin(valid_scenarios)]

    # Identify and remove scenarios with any message delay >= DELAY_THRESHOLD
    scenarios_to_exclude = state_df[state_df['delay_ms'] >= delay_threshold]['scenario_name'].unique()
    state_df = state_df[~state_df['scenario_name'].isin(scenarios_to_exclude)]

    return state_df


def get_scenario_files_and_example_scenario_as_state_dataframes(technology: Optional[str] = None,
                                                                minimum_number_of_messages_per_scenario: int = 20,
                                                                delay_threshold: float = 3000.0,
                                                                object_variables: list[str] = None,
                                                                sample_percent: int = None):
    state_df = get_state_df(technology, object_variables)

    state_df = filter_by_delay_and_num_messages(state_df,
                                                minimum_number_of_messages_per_scenario,
                                                delay_threshold)
    num_scenarios = len(state_df['scenario_name'].unique())

    train_df = None
    test_df = None

    if sample_percent:
        application_scenarios = state_df.groupby('application')['scenario_name'].unique().to_dict()

        scenario_names = list()
        i = 0
        while len(scenario_names) < num_scenarios * (sample_percent / 100):
            for application in application_scenarios.keys():
                if len(application_scenarios[application]) > i:
                    scenario_names.append(application_scenarios[application][i])
            i += 1
        train_df = state_df[state_df['scenario_name'].isin(scenario_names)]
        test_df = state_df[~state_df['scenario_name'].isin(scenario_names)]

    return train_df, test_df


def get_scenario_files_as_state_df(technology: Optional[str] = None,
                                   test_scenario_name: Optional[str] = None,
                                   minimum_number_of_messages_per_scenario: int = 20,
                                   delay_threshold: float = 3000.0,
                                   object_variables: list[str] = None,
                                   sample_percent: int = None) -> pd.DataFrame:
    """
    Extract data from the saved state data and store test scenario in a separate variable.
    Also filters the scenarios by:
        - number of messages in the scenario
        - delay threshold
    :param sample_percent: percent of the dataset.
    :param object_variables: object variables for clustering -> if provided will only be filled with 0 values.
    :param technology: communication technology (LTE, LTE450, 5G, Ethernet).
    :param delay_threshold: threshold above which scenarios are not included.
    :param test_scenario_name: name of test scenario, if provided. Else None.
    :param minimum_number_of_messages_per_scenario: minimum number of messages a valid scenario should contain.
    :return: dataframe with message data.
    """
    state_df = get_state_df(technology, object_variables)

    # Filter out test scenario if provided
    state_df = state_df[state_df['scenario_name'] != test_scenario_name]

    state_df = filter_by_delay_and_num_messages(state_df,
                                                minimum_number_of_messages_per_scenario,
                                                delay_threshold)
    num_scenarios = len(state_df['scenario_name'].unique())

    if sample_percent:
        application_scenarios = state_df.groupby('application')['scenario_name'].unique().to_dict()

        scenario_names = list()
        i = 0
        while len(scenario_names) < num_scenarios * (sample_percent / 100):
            for application in application_scenarios.keys():
                if len(application_scenarios[application]) > i:
                    scenario_names.append(application_scenarios[application][i])
            i += 1
        state_df = state_df[state_df['scenario_name'].isin(scenario_names)]

    print('Generate scenario dataframe with size ', state_df.size)

    return state_df


def cluster_messages(object_variables: list[str],
                     state_df: pd.DataFrame,
                     clustering_distance_threshold: float) -> pd.DataFrame:
    """
    Cluster objects (messages) from historical data.
    :param object_variables: list of object variable names.
    :param state_df: dataframe with state data.
    :param clustering_distance_threshold: threshold for clustering algorithm.
    :return: dataframe with message data.
    """
    # state_df = state_df[object_variables + ['delay_ms']]

    # calculate pairwise distances with squared Euclidean distance metric
    dis_matrix = pdist(state_df[object_variables], metric='seuclidean')

    # Calculate linkages with hierarchical clustering (average linkage)
    linkage_matrix_average = linkage(dis_matrix, method='average')  # average linkage
    # build cluster from the previously calculated distances between (message) objects
    label_av = fcluster(linkage_matrix_average, t=clustering_distance_threshold, criterion='distance')

    # Add cluster labels to the dataframe
    state_df['cluster_av'] = label_av.tolist()

    return state_df


def train_and_tune_decision_tree_regressors_on_clusters(state_df: pd.DataFrame,
                                                        model_features: list[str],
                                                        param_grid=None):
    model_for_cluster_id = {}
    # Train a regression model for each cluster
    for cluster_id in state_df['cluster_av'].unique():
        # Select historical data for the current cluster
        cluster_data = state_df[state_df['cluster_av'] == cluster_id]

        # Extract features (X) and target (y) for the current cluster
        X = cluster_data[model_features]
        y = cluster_data['delay_ms']
        rf = DecisionTreeRegressor(random_state=42)

        if param_grid:
            print('Param Grid is set, initialize. ')
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1,
                                       scoring='neg_mean_squared_error')
            print('Fit grid to data.')
            try:
                grid_search.fit(X, y)
                print('Get best model from grid search. ')
                best_model = grid_search.best_estimator_
                print(f'Best parameters for cluster {cluster_id}: {grid_search.best_params_}')
            except ValueError:
                best_model = rf.fit(X, y)
            model_for_cluster_id[cluster_id] = best_model
        else:
            # Initialize random forest regressor and grid search
            rf.fit(X, y)
            model_for_cluster_id[cluster_id] = rf

    return model_for_cluster_id


def train_decision_tree_regressors_on_clusters(state_df: pd.DataFrame,
                                               model_features: list[str]):
    """
    Trains decision tree regressors on each cluster.
    :param state_df: Dataframe with message data (state attributes and delay times) and cluster information.
    :param model_features: list of model features.
    :return: map containing {cluster id: trained model}
    """
    # Create a dictionary to store regression models for each cluster
    cluster_models = {}

    # Train a regression model for each cluster
    for cluster_id in state_df['cluster_av'].unique():
        # Select historical data for the current cluster
        cluster_data = state_df[state_df['cluster_av'] == cluster_id]

        # Extract features (X) and target (y)
        X_train = cluster_data[model_features]
        y_train = cluster_data['delay_ms']

        # Train a decision tree regression model
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

        cluster_models[cluster_id] = model
    return cluster_models
