import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor


def train_decision_tree_regressor_online(cluster_model: DecisionTreeRegressor,
                                         state_df: pd.DataFrame,
                                         model_features: list[str]):
    """
    Trains decision tree regressor on current scenario data.
    :param state_df: Dataframe with message data (state attributes and delay times) and cluster information.
    :param model_features: list of model features.
    :return: map containing {cluster id: trained model}
    """
    # Extract features (X) and target (y)
    X_train = state_df[model_features]
    y_train = state_df['delay_ms']

    # Train a decision tree regression model
    model = clone(cluster_model)
    model.fit(X_train, y_train)

    return model


def get_exponential_weighted_moving_average(errors: list,
                                            alpha: float):
    # EWMA(t) = alpha * x(t) + (1-alpha) * EWMA(t-1)
    exp_weigh_mov_av = None
    for i in range(len(errors)):
        if i == 0:
            exp_weigh_mov_av = errors[i]
        else:
            exp_weigh_mov_av = alpha * errors[i] + (1 - alpha) * exp_weigh_mov_av
    return exp_weigh_mov_av


def weighted_prediction(alpha: float,
                        abs_error_cluster_predictions: list,
                        abs_error_online_predictions: list,
                        cluster_prediction: float,
                        online_prediction: float) -> float:
    if len(abs_error_online_predictions) == 0:
        if not online_prediction:
            return cluster_prediction
        return np.mean([cluster_prediction, online_prediction])
    # calculate exponential weighted moving average in error values for weighting
    cl_pred_moving_average = get_exponential_weighted_moving_average(errors=abs_error_cluster_predictions, alpha=alpha)
    on_pred_moving_average = get_exponential_weighted_moving_average(errors=abs_error_online_predictions, alpha=alpha)
    return ((online_prediction * cl_pred_moving_average
             + cluster_prediction * on_pred_moving_average) /
            (on_pred_moving_average + cl_pred_moving_average))
