import pandas as pd
from scipy.spatial.distance import cdist


def assign_message_to_cluster(row: pd.Series,
                              variables,
                              train_df: pd.DataFrame) -> (int, float):
    """
    Calculates distance of this message object towards all other message objects and returns cluster of clotest message
    and corresponding distance.
    :param row: Series object with message object data.
    :param variables: variables of message object.
    :param train_df: Dataframe with training data and cluster information.
    :return: tuple of (closest cluster id, distance towards closest cluster).
    """
    # Extract the feature values for the current test message
    test_features = row[variables].values.reshape(1, -1)

    # Compute distances between this test message and all historical messages
    test_distances = cdist(test_features, train_df[variables], metric='seuclidean')
    closest_historical_idx = test_distances.argmin()
    # Assign the test message to the cluster of the closest historical message
    closest_cluster = train_df.iloc[closest_historical_idx]['cluster_av']
    distance = test_distances.T[closest_historical_idx][0]
    return closest_cluster, distance
