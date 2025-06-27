import copy
import logging
import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from numpy import floating
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone

logger = logging.getLogger(__name__)


@dataclass
class MessageObservation:
    sender: str
    receiver: str
    payload_size_B: int
    time_send_ms: int
    msg_id: str
    time_receive_ms: int = math.inf

    actual_delay_ms: int = math.inf
    cluster_predicted_delay_ms: int = math.inf
    online_predicted_delay_ms: int = math.inf
    weighted_predicted_delay_ms: int = math.inf

    sender_node_state: Optional['NodeState'] = None
    receiver_node_state: Optional['NodeState'] = None
    network_state: Optional['NetworkState'] = None


@dataclass
class SubstitutionInfo:
    occurred: bool = False
    message_index: int = 0
    confidence_score: float = 0.0
    current_time_s: float = 0.0
    additional_metrics: dict = None


@dataclass
class Message:
    sender: str
    receiver: str
    payload_size_B: int
    time_send_ms: int
    msg_id: str
    time_receive_ms: int = math.inf
    delay_ms: int = math.inf


@dataclass
class NodeState:
    """
    Data class representing the state of a network node in cocoon.
    """
    average_outgoing_delay_ms: float = math.inf
    average_incoming_delay_ms: float = math.inf
    num_messages_sent_simultaneously: int = 0

    def get_as_sender_node_dict(self):
        return {
            'sender_average_outgoing_delay_ms': self.average_outgoing_delay_ms if self.average_outgoing_delay_ms != math.inf else None,
            'sender_average_incoming_delay_ms': self.average_incoming_delay_ms if self.average_incoming_delay_ms != math.inf else None,
            'sender_num_messages_sent_simultaneously': self.num_messages_sent_simultaneously
        }

    def get_as_receiver_node_dict(self):
        return {
            'receiver_average_outgoing_delay_ms': self.average_outgoing_delay_ms if self.average_outgoing_delay_ms != math.inf else None,
            'receiver_average_incoming_delay_ms': self.average_incoming_delay_ms if self.average_incoming_delay_ms != math.inf else None,
            'receiver_num_messages_sent_simultaneously': self.num_messages_sent_simultaneously
        }


@dataclass
class NetworkState:
    """
    Data class representing the state of the network in cocoon.
    """
    average_delay_ms: float = math.inf
    num_messages_in_transit: int = 0
    num_busy_links: int = 0
    num_network_nodes: int = 0
    num_messages_sent_simultaneously: int = 0

    def get_as_dict(self):
        return {
            'network_average_delay_ms': self.average_delay_ms if self.average_delay_ms != math.inf else None,
            'network_num_messages_in_transit': self.num_messages_in_transit,
            'network_num_busy_links': self.num_busy_links,
            'network_num_network_nodes': self.num_network_nodes,
            'network_num_messages_sent_simultaneously': self.num_messages_sent_simultaneously
        }


class CocoonNetworkNode:
    def __init__(self, name: str):
        self.name = name
        self.messages_sent = []
        self.messages_received = []
        self.node_state = NodeState()

        # Track messages by ID for updates
        self.sent_messages_by_id: Dict[str, Message] = {}
        self.received_messages_by_id: Dict[str, Message] = {}

    def register_sent_message(self, receiver: str, payload_size_B: int, current_time_ms: int, msg_id: str):
        """Register a message that this node sent."""
        message = Message(
            sender=self.name,
            receiver=receiver,
            payload_size_B=payload_size_B,
            time_send_ms=current_time_ms,
            msg_id=msg_id
        )

        self.messages_sent.append(message)
        self.sent_messages_by_id[msg_id] = message

    def update_sent_message_received(self, msg_id: str, time_receive_ms: int):
        """Update a sent message when it's confirmed as received."""
        if msg_id in self.sent_messages_by_id:
            message = self.sent_messages_by_id[msg_id]
            message.time_receive_ms = time_receive_ms
            message.delay_ms = time_receive_ms - message.time_send_ms
            return True
        return False

    def add_received_message_from_global(self, message: Message):
        """Add a received message reference from global tracking."""
        if message.receiver == self.name and message.msg_id not in self.received_messages_by_id:
            self.messages_received.append(message)
            self.received_messages_by_id[message.msg_id] = message

    def get_sent_message_by_id(self, msg_id: str) -> Optional[Message]:
        """Get a sent message by its ID."""
        return self.sent_messages_by_id.get(msg_id)

    def get_received_message_by_id(self, msg_id: str) -> Optional[Message]:
        """Get a received message by its ID."""
        return self.received_messages_by_id.get(msg_id)

    def has_received_messages(self):
        return len(self.messages_received) > 0

    def has_sent_messages(self):
        return len(self.messages_sent) > 0

    def update_state(self, time_ms):
        """
        Updates state based on current knowledge.
        """
        # Handle incoming delays
        incoming_delays = [msg.delay_ms for msg in self.messages_received
                           if msg.delay_ms != math.inf and msg.delay_ms >= 0]
        average_incoming_delay_ms = statistics.mean(incoming_delays) if incoming_delays else math.inf

        # Handle outgoing delays
        outgoing_delays = [msg.delay_ms for msg in self.messages_sent
                           if msg.delay_ms != math.inf and msg.delay_ms >= 0]
        average_outgoing_delay_ms = statistics.mean(outgoing_delays) if outgoing_delays else math.inf

        # Count simultaneous messages
        num_messages_sent_simultaneously = len([msg for msg in self.messages_sent if msg.time_send_ms == time_ms])

        self.node_state = NodeState(average_incoming_delay_ms, average_outgoing_delay_ms,
                                    num_messages_sent_simultaneously)

        return copy.deepcopy(self.node_state)


class CocoonNetworkGraph:
    def __init__(self):
        self.nodes: Dict[str, CocoonNetworkNode] = {}
        self.network_state = NetworkState()

        # Global message tracking - this is the single source of truth
        self.all_messages_by_id: Dict[str, Message] = {}

    def get_or_initialize_node(self, node_name) -> CocoonNetworkNode:
        if node_name not in self.nodes:
            self.nodes[node_name] = CocoonNetworkNode(name=node_name)
        return self.nodes[node_name]

    def update_state(self, time_ms):
        """
        Update network state for a given simulation time.
        """
        completed_messages = self.get_completed_messages()
        average_delay_ms = statistics.mean([msg.delay_ms for msg in completed_messages]) \
            if len(completed_messages) > 0 else math.inf
        num_messages_in_transit = len(self.get_messages_in_transit())
        num_network_nodes = len(self.nodes)
        messages_in_transit = self.get_messages_in_transit()
        busy_links = []
        for m in messages_in_transit:
            if (m.sender, m.receiver) not in busy_links:
                busy_links.append((m.sender, m.receiver))
        num_busy_links = len(busy_links)
        num_messages_sent_simultaneously = len([msg for msg in self.all_messages_by_id.values()
                                                if msg.time_send_ms == time_ms])
        self.network_state = NetworkState(average_delay_ms, num_messages_in_transit, num_busy_links, num_network_nodes,
                                          num_messages_sent_simultaneously)
        return copy.deepcopy(self.network_state)

    def register_sent_message(self, sender: str, receiver: str,
                              payload_size_B: int, current_time_ms: int, msg_id: str):
        """Register that a message was sent from sender to receiver."""
        sender_node = self.get_or_initialize_node(node_name=sender)
        receiver_node = self.get_or_initialize_node(node_name=receiver)  # Ensure receiver node exists

        sender_node.register_sent_message(
            receiver=receiver,
            payload_size_B=payload_size_B,
            current_time_ms=current_time_ms,
            msg_id=msg_id
        )

        # Store in global tracking (single source of truth)
        message = sender_node.get_sent_message_by_id(msg_id)
        if message:
            self.all_messages_by_id[msg_id] = message

    def mark_message_received(self, msg_id: str, current_time_ms: int) -> bool:
        """Mark a message as received using only msg_id and current time."""
        # Find the message in global tracking
        if msg_id not in self.all_messages_by_id:
            print(f"Warning: Message {msg_id} not found in global tracking")
            return False

        message = self.all_messages_by_id[msg_id]

        # Update the message with receive time and delay
        message.time_receive_ms = current_time_ms
        message.delay_ms = current_time_ms - message.time_send_ms

        # Update the sender node's sent message
        sender_node = self.get_or_initialize_node(message.sender)
        sender_node.update_sent_message_received(msg_id, current_time_ms)

        # Add the message to receiver node's received messages list
        receiver_node = self.get_or_initialize_node(message.receiver)
        receiver_node.add_received_message_from_global(message)
        return True

    def get_message_by_id(self, msg_id: str) -> Optional[Message]:
        """Get any message by its ID from global tracking."""
        return self.all_messages_by_id.get(msg_id)

    def get_messages_in_transit(self) -> list[Message]:
        """Get all messages that have been sent but not yet received."""
        messages_in_transit = []
        for message in self.all_messages_by_id.values():
            if message.time_receive_ms == math.inf:
                messages_in_transit.append(message)
        return messages_in_transit

    def get_completed_messages(self) -> list[Message]:
        """Get all messages that have been both sent and received."""
        completed_messages = []
        for message in self.all_messages_by_id.values():
            if message.time_receive_ms != math.inf:
                completed_messages.append(message)
        return completed_messages


class CocoonMetaModel:
    """
    Meta-model called cocoon which is supposed to approximate the detailed simulation.
    """

    class Mode(Enum):
        TRAINING = 0
        PRODUCTION = 1

    def __init__(self, output_file_name: str, mode: Mode = Mode.TRAINING, cluster_distance_threshold: float = 5,
                 i_pupa: int = 10, alpha: float = 0.3):
        self.output_file_name = output_file_name
        self.mode = mode
        self.network_graph = CocoonNetworkGraph()

        self.substitution_info = SubstitutionInfo()

        # DataFrame containing training data for the regressors
        self.training_df = None
        # Dictionary containing all pre-trained regressors on historical data
        self.model_for_cluster_id = {}
        self.online_model = None  # TODO: add multiple models for different clusters

        # create empty dictionary in order track observations for prediction training
        self.message_observations: Dict[str, MessageObservation] = {}

        self.object_variables = ['network_num_messages_in_transit', 'network_num_busy_links',
                                 'network_num_network_nodes',
                                 'network_num_messages_sent_simultaneously', 'payload_size_B',
                                 'sender_num_messages_sent_simultaneously']
        self.emerging_variables = ['network_average_delay_ms', 'sender_average_outgoing_delay_ms',
                                   'receiver_average_incoming_delay_ms']
        self.model_features = self.object_variables + self.emerging_variables

        self.clustering_distance_threshold = cluster_distance_threshold

        self.message_index = 0
        self.i_pupa = i_pupa

        # Additional attributes for weighted predictions
        self.alpha = alpha  # EWMA smoothing parameter
        self.cluster_prediction_errors = []  # Track cluster prediction errors
        self.online_prediction_errors = []  # Track online prediction errors
        self.weighted_prediction_errors = []  # Track weighted prediction errors

        # OPTIMIZATION: Store cluster centroids for fast distance calculation
        self.cluster_centroids = {}  # {cluster_id: centroid_vector}

        # boolean variable indicating if substitution threshold has been reached
        self.substitution_threshold_reached = False
        self.previous_node_count = 0

    async def get_messages_in_transit(self):
        messages_in_transit = self.network_graph.get_messages_in_transit()
        messages_in_transit_with_delay = [{'msg_id': m.msg_id,
                                           'time_send_ms': m.time_send_ms,
                                           'delay_ms': self.message_observations[m.msg_id].weighted_predicted_delay_ms}
                                          for m in messages_in_transit]

        return messages_in_transit_with_delay

    def execute_egg_phase(self, training_df: pd.DataFrame):
        """
        ----------
        EGG phase
        ----------
        Initialization of the simulation set-up, pre-training of decision tree regressors for each cluster.
        Now with centroid computation for LARVA optimization.
        """
        # fill training dataframe with 0s
        training_df[self.object_variables] = training_df[self.object_variables].fillna(0)

        self.training_df = training_df
        # calculate pairwise distances with squared Euclidean distance metric
        dis_matrix = pdist(training_df[self.object_variables], metric='seuclidean')

        # Calculate linkages with hierarchical clustering (centroid linkage)
        linkage_matrix_centroid = linkage(dis_matrix, method='centroid')  # centroid linkage
        # build cluster from the previously calculated distances between (message) objects
        label_cen = fcluster(linkage_matrix_centroid, t=self.clustering_distance_threshold, criterion='distance')

        # Add cluster labels to the dataframe
        training_df['cluster_cen'] = label_cen.tolist()

        self.compute_cluster_centroids(training_df)

        # Train a regression model for each cluster
        for cluster_id in training_df['cluster_cen'].unique():
            # Select historical data for the current cluster
            cluster_data = training_df[training_df['cluster_cen'] == cluster_id]

            # Extract features (X) and target (y) for the current cluster
            X = cluster_data[self.model_features]
            y = cluster_data['actual_delay_ms']
            reg = DecisionTreeRegressor(random_state=42)

            reg.fit(X, y)

            # Store feature names for later use
            reg.feature_names_in_ = np.array(self.model_features)

            self.model_for_cluster_id[cluster_id] = reg

        logger.info(f'EGG phase done. Resulting in a number of {len(self.model_for_cluster_id)} '
                    f'distinct clusters with one regressor each. Computed {len(self.cluster_centroids)} centroids.')

    def compute_cluster_centroids(self, training_df: pd.DataFrame):
        """
        Compute the centroid (mean) of each cluster in object variable space
        """
        self.cluster_centroids = {}

        for cluster_id in training_df['cluster_cen'].unique():
            cluster_data = training_df[training_df['cluster_cen'] == cluster_id]
            cluster_features = cluster_data[self.object_variables]

            # Compute centroid (mean of all points in cluster)
            centroid = cluster_features.mean().values
            self.cluster_centroids[cluster_id] = centroid

        logger.info(f'Computed centroids for {len(self.cluster_centroids)} clusters')

    def get_all_state_variables_as_dict(self, sender_node: CocoonNetworkNode, receiver_node: CocoonNetworkNode):
        state_dict = {}
        state_dict.update(sender_node.node_state.get_as_sender_node_dict())
        state_dict.update(receiver_node.node_state.get_as_receiver_node_dict())
        state_dict.update(self.network_graph.network_state.get_as_dict())
        return state_dict

    def execute_larva_phase(self, sender_node: CocoonNetworkNode, receiver_node: CocoonNetworkNode,
                            payload_size_B: int) -> int:
        """
        ------------
        LARVA phase
        ------------
        Assign current message to closest cluster using centroid distance calculation only.
        This is much faster than calculating distances to all training samples.
        """
        variables_dict = self.get_all_state_variables_as_dict(sender_node, receiver_node)
        variables_dict.update({'payload_size_B': payload_size_B})

        # Create feature vector for current message (still use numpy for centroid distance)
        message_features = np.array([variables_dict[var] for var in self.object_variables])

        # OPTIMIZATION: Calculate distance only to cluster centroids (much faster!)
        closest_cluster = self.find_closest_centroid(message_features)

        logger.info(f'Assigned message to cluster {closest_cluster} using centroid optimization')

        # get regressor of closest cluster
        regressor_cluster = self.model_for_cluster_id[closest_cluster]

        # Create DataFrame correctly with feature values as a dictionary
        prediction_dict = {var: variables_dict[var] for var in self.model_features}
        prediction_data = pd.DataFrame([prediction_dict])

        # predict message delay time with cluster regressor
        d_cl_pred = regressor_cluster.predict(prediction_data)[0]
        logger.info(f'Predicted delay time with cluster regressor: d_cl_pred = {d_cl_pred}. ')
        return int(d_cl_pred)

    def find_closest_centroid(self, message_features: np.ndarray) -> int:
        """
        Find closest cluster by comparing only to centroids (FASTEST approach)
        """
        min_distance = float('inf')
        closest_cluster = None

        for cluster_id, centroid in self.cluster_centroids.items():
            # Calculate Euclidean distance to centroid
            distance = np.linalg.norm(message_features - centroid)

            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        return closest_cluster

    def execute_pupa_phase(self, sender_node: CocoonNetworkNode, receiver_node: CocoonNetworkNode,
                           payload_size_B: int, cluster_prediction: float = None):
        """
        Enhanced PUPA phase with weighted predictions
        """
        variables_dict = self.get_all_state_variables_as_dict(sender_node, receiver_node)
        variables_dict.update({'payload_size_B': payload_size_B})

        online_prediction = None

        # Train/update online model periodically
        if self.message_index % self.i_pupa == 0:
            message_observations_as_df = self.get_observations_as_df()
            message_observations_as_df.dropna(subset=['actual_delay_ms'], inplace=True)

            if len(message_observations_as_df) > 0:
                # Extract features (X) and target (y)
                X = message_observations_as_df[self.model_features]
                y = message_observations_as_df['actual_delay_ms']
                reg = DecisionTreeRegressor(random_state=42)
                reg.fit(X, y)
                self.online_model = reg

        # Make online prediction if model exists
        if self.online_model is not None:
            # Create DataFrame correctly with feature values as a dictionary
            prediction_dict = {var: variables_dict[var] for var in self.model_features}
            prediction_data = pd.DataFrame([prediction_dict])

            online_prediction = int(self.online_model.predict(prediction_data)[0])
            logger.info(f'Predicted delay time online: d_on_pred = {online_prediction}')

        weighted_pred = None
        # Calculate weighted prediction
        if cluster_prediction is not None and online_prediction is not None:
            weighted_pred = int(self.weighted_prediction(
                alpha=self.alpha,
                abs_error_cluster_predictions=self.cluster_prediction_errors,
                abs_error_online_predictions=self.online_prediction_errors,
                cluster_prediction=cluster_prediction,
                online_prediction=online_prediction
            ))
            logger.info(f'Weighted prediction: d_w_pred = {weighted_pred}')
        return online_prediction, weighted_pred

    def train_decision_tree_regressor_online(self, cluster_model: DecisionTreeRegressor,
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

    def get_exponential_weighted_moving_average(self, errors: list, alpha: float):
        """
        Calculation of the exponential moving average (EWMA) for weighting of the predictions.
        """
        if not errors:
            return 0

        exp_weigh_mov_av = None
        for i in range(len(errors)):
            if i == 0:
                exp_weigh_mov_av = errors[i]
            else:
                exp_weigh_mov_av = alpha * errors[i] + (1 - alpha) * exp_weigh_mov_av
        return exp_weigh_mov_av

    def weighted_prediction(self, alpha: float,
                            abs_error_cluster_predictions: list,
                            abs_error_online_predictions: list,
                            cluster_prediction: float,
                            online_prediction: float) -> float | floating[Any] | Any:
        """
        Calculate weighted prediction based on historical errors
        """
        if len(abs_error_online_predictions) == 0:
            if online_prediction is None:
                return cluster_prediction
            return np.mean([cluster_prediction, online_prediction])

        # Calculate exponential weighted moving average of errors
        cl_pred_moving_average = self.get_exponential_weighted_moving_average(
            errors=abs_error_cluster_predictions, alpha=alpha
        )
        on_pred_moving_average = self.get_exponential_weighted_moving_average(
            errors=abs_error_online_predictions, alpha=alpha
        )

        # Avoid division by zero
        if (on_pred_moving_average + cl_pred_moving_average) == 0:
            return np.mean([cluster_prediction, online_prediction])

        # Weight predictions inversely to their error rates
        weighted_pred = ((online_prediction * cl_pred_moving_average
                          + cluster_prediction * on_pred_moving_average) /
                         (on_pred_moving_average + cl_pred_moving_average))

        return weighted_pred

    def execute_butterfly_phase(self):
        """
        ----------
        BUTTERFLY phase
        ----------
        Calculate substitution threshold based on:
        - Error trend of weighted predictions (d_w_pred)
        - Distance towards nearest cluster (dist_c)
        - Network graph topology information (number of nodes)

        Sets self.substitution_threshold_reached to True if the meta-model has
        reached sufficient performance/stability.
        """
        # 1. Error trend of weighted predictions
        if len(self.weighted_prediction_errors) >= self.i_pupa:
            # Recent prediction errors
            recent_errors = self.weighted_prediction_errors[-self.i_pupa:]
            # 1. Error Trend Factor (slope over all points)
            x = np.arange(len(recent_errors))
            y = np.array(recent_errors)
            # Linear regression: y = a * x + b
            slope, _ = np.polyfit(x, y, 1)
            if slope >= 0:
                error_trend_factor = 0.0
            else:
                error_trend_factor = min(1.0, abs(slope) / 50.0)

            expected_max_error = 100.0
            recent_mean_error = np.mean(recent_errors)
            error_level_factor = 1.0 - min(1.0, recent_mean_error / expected_max_error)

        else:
            # Not enough history to determine trend
            error_trend_factor = 0.3
            error_level_factor = 0.3

        # 2. Distance towards nearest cluster
        # Calculate for the most recent message observation
        if self.message_observations and self.message_index > 0:
            # Get the most recent message
            recent_msg_id = list(self.message_observations.keys())[-1]
            observation = self.message_observations[recent_msg_id]

            # Extract object variables from the observation
            msg_features = []
            for var in self.object_variables:
                if var.startswith('network_'):
                    msg_features.append(observation.network_state.get_as_dict().get(var, 0))
                elif var.startswith('sender_'):
                    msg_features.append(observation.sender_node_state.get_as_sender_node_dict().get(var, 0))
                elif var.startswith('receiver_'):
                    msg_features.append(observation.receiver_node_state.get_as_receiver_node_dict().get(var, 0))
                elif var == 'payload_size_B':
                    msg_features.append(observation.payload_size_B)

            msg_features = np.array(msg_features)

            # Find closest cluster and its distance
            min_distance = float('inf')

            for cluster_id, centroid in self.cluster_centroids.items():
                distance = np.linalg.norm(msg_features - centroid)
                if distance < min_distance:
                    min_distance = distance

            # Normalize distance (closer is better)
            # Using a reasonable maximum expected distance
            max_expected_distance = self.clustering_distance_threshold * 100  # could be much higher than in training data
            cluster_distance_factor = 1.0 - min(1.0, min_distance / max_expected_distance)

        else:
            cluster_distance_factor = 0.3  # Neutral starting value

        # 3. Network graph topology (number of nodes)
        current_node_count = len(self.network_graph.nodes)

        # Calculate stability of network topology
        if self.previous_node_count > 0:
            # Lower change ratio means more stable topology
            node_change_ratio = abs(current_node_count - self.previous_node_count) / self.previous_node_count
            network_topology_factor = 1.0 - min(1.0, node_change_ratio * 2)
        else:
            network_topology_factor = 0.5

        # Update previous node count for next call
        self.previous_node_count = current_node_count

        # Calculate combined threshold using weighted factors
        weights = {
            'error_trend': 0.1,
            'error_level': 0.4,
            'cluster_distance': 0.3,
            'network_topology': 0.2
        }

        combined_score = (
                weights['error_trend'] * error_trend_factor +
                weights['error_level'] * error_level_factor +
                weights['cluster_distance'] * cluster_distance_factor +
                weights['network_topology'] * network_topology_factor
        )

        # Determine if substitution threshold is reached
        threshold_value = 0.75  # Required threshold to trigger substitution
        self.substitution_threshold_reached = combined_score >= threshold_value

        if self.substitution_threshold_reached and not self.substitution_info.occurred:
            print('--------------SUBSTITUTION--------------')
            self.substitution_info = SubstitutionInfo(
                occurred=True,
                message_index=self.message_index,
                confidence_score=combined_score,
                current_time_s=time.time(),
                additional_metrics={
                    'total_messages_processed': len(self.message_observations),
                    'cluster_prediction_errors_count': len(self.cluster_prediction_errors),
                    'online_prediction_errors_count': len(self.online_prediction_errors)
                }
            )

        # Log the results if logger is available
        if logger:
            logger.info(f"Butterfly Phase Results:")
            logger.info(f"  Error Trend Factor: {error_trend_factor:.4f}")
            logger.info(f"  Error Level Factor: {error_level_factor:.4f}")
            logger.info(f"  Cluster Distance Factor: {cluster_distance_factor:.4f}")
            logger.info(f"  Network Topology Factor: {network_topology_factor:.4f}")
            logger.info(f"  Combined Score: {combined_score:.4f} (required: {threshold_value})")
            logger.info(f"  Substitution Threshold Reached: {self.substitution_threshold_reached}")

        return self.substitution_threshold_reached

    def update_prediction_errors(self, msg_id: str, actual_delay: float):
        """
        Update prediction error tracking when actual delay becomes available
        """
        if msg_id in self.message_observations:
            observation = self.message_observations[msg_id]

            if observation.cluster_predicted_delay_ms and observation.cluster_predicted_delay_ms != math.inf:
                self.cluster_prediction_errors.append(abs(actual_delay - observation.cluster_predicted_delay_ms))

            if observation.online_predicted_delay_ms and observation.online_predicted_delay_ms != math.inf:
                self.online_prediction_errors.append(abs(actual_delay - observation.online_predicted_delay_ms))

            if observation.weighted_predicted_delay_ms and observation.weighted_predicted_delay_ms != math.inf:
                self.weighted_prediction_errors.append(abs(actual_delay - observation.weighted_predicted_delay_ms))

    async def process_sent_message(self, sender: str, receiver: str,
                                   payload_size_B: int, current_time_ms: int, msg_id: str):
        """Process a message that was sent."""
        self.network_graph.register_sent_message(
            sender=sender,
            receiver=receiver,
            payload_size_B=payload_size_B,
            current_time_ms=current_time_ms,
            msg_id=msg_id
        )

    def process_received_message(self, msg_id: str, current_time_ms: int):
        """Process a message that was received - only needs msg_id and current time."""
        success = self.network_graph.mark_message_received(msg_id, current_time_ms)
        if not success:
            logger.warning(f"Failed to process received message {msg_id}")
        if msg_id not in self.message_observations:
            return
        observation = self.message_observations[msg_id]
        observation.time_receive_ms = current_time_ms
        if not self.substitution_threshold_reached:
            observation.actual_delay_ms = current_time_ms - observation.time_send_ms
            if self.mode == self.Mode.PRODUCTION:
                # Update prediction errors for learning
                self.update_prediction_errors(msg_id, observation.actual_delay_ms)

    async def process_observations(self) -> bool:
        """
        Processes current observations in the detailed simulation.
        Returns boolean, indicating if substitution threshold has been reached.
        """
        messages_in_transit = self.network_graph.get_messages_in_transit()
        for message in messages_in_transit:
            self.message_index += 1
            sender_node = self.network_graph.get_or_initialize_node(node_name=message.sender)
            sender_node_state = sender_node.update_state(time_ms=message.time_send_ms)
            receiver_node = self.network_graph.get_or_initialize_node(node_name=message.receiver)
            receiver_node_state = receiver_node.update_state(time_ms=message.time_send_ms)
            network_state = self.network_graph.update_state(time_ms=message.time_send_ms)

            d_cl_pred = None
            d_on_pred = None
            d_w_pred = None

            if self.mode == self.Mode.PRODUCTION:
                # LARVA phase - get cluster prediction (now optimized with centroids)
                d_cl_pred = self.execute_larva_phase(sender_node, receiver_node, payload_size_B=message.payload_size_B)

                if self.message_index >= self.i_pupa:
                    # PUPA phase - get weighted prediction
                    d_on_pred, d_w_pred = self.execute_pupa_phase(
                        sender_node, receiver_node,
                        payload_size_B=message.payload_size_B,
                        cluster_prediction=d_cl_pred
                    )

            self.message_observations[message.msg_id] = MessageObservation(sender=message.sender,
                                                                           receiver=message.receiver,
                                                                           payload_size_B=message.payload_size_B,
                                                                           time_send_ms=message.time_send_ms,
                                                                           msg_id=message.msg_id,
                                                                           sender_node_state=sender_node_state,
                                                                           receiver_node_state=receiver_node_state,
                                                                           network_state=network_state,
                                                                           cluster_predicted_delay_ms=d_cl_pred,
                                                                           online_predicted_delay_ms=d_on_pred,
                                                                           weighted_predicted_delay_ms=d_w_pred)

        if self.mode == self.Mode.PRODUCTION and self.message_index >= self.i_pupa:
            if self.substitution_threshold_reached:
                # threshold has already been reached -> return True
                return True
            return self.execute_butterfly_phase()
        else:
            return False

    def get_observations_as_df(self):
        observations_data = []
        for msg_obs in self.message_observations.values():
            obs_dict = {
                'msg_id': msg_obs.msg_id,
                'sender': msg_obs.sender,
                'receiver': msg_obs.receiver,
                'payload_size_B': msg_obs.payload_size_B,
                'time_send_ms': msg_obs.time_send_ms,
                'time_receive_ms': msg_obs.time_receive_ms if msg_obs.time_receive_ms != math.inf else None,
                'actual_delay_ms': msg_obs.actual_delay_ms if msg_obs.actual_delay_ms != math.inf else None,
                'predicted_delay_ms': msg_obs.weighted_predicted_delay_ms if msg_obs.weighted_predicted_delay_ms != math.inf else None
            }
            obs_dict.update(msg_obs.sender_node_state.get_as_sender_node_dict())
            obs_dict.update(msg_obs.receiver_node_state.get_as_receiver_node_dict())
            obs_dict.update(msg_obs.network_state.get_as_dict())
            observations_data.append(obs_dict)

        return pd.DataFrame(observations_data)

    async def save_observations(self):
        observations_data = []
        for msg_obs in self.message_observations.values():
            obs_dict = {
                'msg_id': msg_obs.msg_id,
                'sender': msg_obs.sender,
                'receiver': msg_obs.receiver,
                'payload_size_B': msg_obs.payload_size_B,
                'time_send_ms': msg_obs.time_send_ms,
                'time_receive_ms': msg_obs.time_receive_ms if msg_obs.time_receive_ms != math.inf else None,
                'actual_delay_ms': msg_obs.actual_delay_ms if msg_obs.actual_delay_ms != math.inf else None,
                'predicted_delay_ms': msg_obs.weighted_predicted_delay_ms if msg_obs.weighted_predicted_delay_ms != math.inf else None
            }
            obs_dict.update(msg_obs.sender_node_state.get_as_sender_node_dict())
            obs_dict.update(msg_obs.receiver_node_state.get_as_receiver_node_dict())
            obs_dict.update(msg_obs.network_state.get_as_dict())
            observations_data.append(obs_dict)

        df = pd.DataFrame(observations_data)
        df.to_csv(self.output_file_name, index=False)

        print(f"📊 Observations saved to CSV: {self.output_file_name}")
