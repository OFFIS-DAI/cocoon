"""
Analysis example use case to visualize different phases.
"""
import time

import numpy as np
import plotly.graph_objects as go
from _plotly_utils.colors.qualitative import Pastel
from plotly.subplots import make_subplots
import os
from sklearn.tree import DecisionTreeRegressor
from src.training.egg import cluster_messages, get_scenario_files_and_example_scenario_as_state_dataframes
from src.training.larva import assign_message_to_cluster
from src.training.pupa import train_decision_tree_regressor_online, weighted_prediction

# Define file paths and configurations
TRAINING_SAMPLE_PERCENT = 5
DELAY_THRESHOLD = 3000
THRESHOLD = 5
MINIMUM_NUMBER_OF_MESSAGES_PER_SCENARIO = 70
technology = 'LTE450'
EXAMPLE_SCENARIO_IDX = 41
I_PUPA = 100
ALPHA_WEIGHTED_AVERAGE = 0.9

# Define feature variables
OBJECT_VARIABLES = ['network_num_messages_in_transit', 'network_num_busy_links', 'network_num_network_nodes',
                    'network_messages_sent_at_current_time', 'packet_size_B',
                    'sender_num_messages_sent_simultaneously']
emerging_variables = ['network_average_delay_time', 'sender_average_outgoing_delay_time',
                      'receiver_average_incoming_delay_time']
MODEL_FEATURES = OBJECT_VARIABLES + emerging_variables

best_params = {
    1: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 10, 'min_samples_split': 2},
    2: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 10, 'min_samples_split': 2},
    3: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 5, 'min_samples_split': 15},
    4: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 10, 'min_samples_split': 2},
    5: {'max_depth': 20, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
    6: {'max_depth': 10, 'max_features': None, 'max_leaf_nodes': None, 'min_samples_leaf': 5, 'min_samples_split': 2},
    7: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 50, 'min_samples_leaf': 1, 'min_samples_split': 2},
    8: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 50, 'min_samples_leaf': 1, 'min_samples_split': 2},
    9: {'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 50, 'min_samples_leaf': 1, 'min_samples_split': 2},
    10: {'max_depth': 10, 'max_features': None, 'max_leaf_nodes': None, 'min_samples_leaf': 10, 'min_samples_split': 2}
}

online_models = {key: None for key in best_params.keys()}
# Load state data and apply clustering
train_df, test_df = get_scenario_files_and_example_scenario_as_state_dataframes(
    technology=technology,
    delay_threshold=DELAY_THRESHOLD,
    minimum_number_of_messages_per_scenario=MINIMUM_NUMBER_OF_MESSAGES_PER_SCENARIO,
    object_variables=OBJECT_VARIABLES,
    sample_percent=TRAINING_SAMPLE_PERCENT
)
scenario_names = test_df['scenario_name'].unique()

example_scenario = scenario_names[EXAMPLE_SCENARIO_IDX]
print('example scenario: ', example_scenario)

# Fetch example scenario
example_scenario_df = test_df[test_df['scenario_name'] == example_scenario]
example_scenario_df.sort_values('time_ms')
print('Application: ', example_scenario_df['application'].values[0])
print('len: ', len(example_scenario_df))

clustered_state_df = cluster_messages(state_df=train_df, clustering_distance_threshold=THRESHOLD,
                                      object_variables=OBJECT_VARIABLES)
sorted_cluster_ids = sorted(train_df["cluster_av"].unique())
cluster_to_regressor = {}

for cluster_id in sorted_cluster_ids:
    print(f"Processing Cluster {cluster_id}")
    cluster_data = train_df[train_df["cluster_av"] == cluster_id]

    dt_regressor = DecisionTreeRegressor(random_state=42,
                                         max_depth=10,
                                         min_samples_split=best_params[cluster_id]['min_samples_split'],
                                         min_samples_leaf=best_params[cluster_id]['min_samples_leaf'],
                                         max_features=best_params[cluster_id]['max_features'],
                                         max_leaf_nodes=best_params[cluster_id]['max_leaf_nodes'])

    dt_regressor.fit(cluster_data[MODEL_FEATURES], cluster_data['delay_ms'])
    cluster_to_regressor[cluster_id] = dt_regressor

d_cl_pred_values = []
d_real_values = []
d_on_pred_values = []

weighted_pred_values = []

abs_error_d_w_pred = []
abs_error_d_on_pred = []
abs_error_d_cl_pred = []

closest_clusters = []

msg_num = 0
for i, msg_row in example_scenario_df.iterrows():
    closest_cluster, distance = assign_message_to_cluster(row=msg_row,
                                                          variables=OBJECT_VARIABLES,
                                                          train_df=train_df)
    closest_clusters.append(closest_cluster)
    reg_cl = cluster_to_regressor[closest_cluster]
    d_real = msg_row['delay_ms']
    d_real_values.append(d_real)
    test_data = msg_row[MODEL_FEATURES].to_frame().T
    d_cl_pred = reg_cl.predict(test_data)[0]
    d_cl_pred_values.append(d_cl_pred)
    abs_error_d_cl_pred.append(abs(d_real - d_cl_pred))
    print(msg_num, 'cluster:', closest_cluster, 'pred: ', d_cl_pred, ', real: ', d_real)

    online_prediction = None

    ### Pupa-Phase
    if msg_num >= I_PUPA:
        if msg_num % I_PUPA == 0:
            # train regressor online -> training iteration
            data_subset = example_scenario_df.iloc[:msg_num]
            for key in online_models.keys():
                online_models[key] = train_decision_tree_regressor_online(cluster_model=cluster_to_regressor[key],
                                                                          state_df=data_subset,
                                                                          model_features=MODEL_FEATURES)
        online_model = online_models[closest_cluster]
        online_prediction = online_model.predict(msg_row[MODEL_FEATURES].to_frame().T)[0]
        print('Online prediction: ', online_prediction)
    d_on_pred_values.append(online_prediction)
    if online_prediction:
        abs_error_d_on_pred.append(abs(d_real - online_prediction))
    weighted_pred = weighted_prediction(alpha=ALPHA_WEIGHTED_AVERAGE,
                                        cluster_prediction=d_cl_pred,
                                        online_prediction=online_prediction,
                                        abs_error_cluster_predictions=abs_error_d_cl_pred,
                                        abs_error_online_predictions=abs_error_d_on_pred)
    weighted_pred_values.append(weighted_pred)
    abs_error_d_w_pred.append(abs(d_real - weighted_pred))

    print('Weighted prediction: ', weighted_pred)
    msg_num += 1

print('mean abs error for cluster model: ', np.mean(abs_error_d_cl_pred))
print('mean abs error for online model: ', np.mean(abs_error_d_on_pred))
print('mean abs error for weighted prediction: ', np.mean(abs_error_d_w_pred))
print('max error for weighted prediction: ', np.max(abs_error_d_w_pred))

# Replace the plotting section with this improved version

# Create a single figure with subplots
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, vertical_spacing=0.1,
                    subplot_titles=[r'$d_{\text{cl pred}}$',
                                    r'$d_{\text{on pred}}$',
                                    r'$d_{\text{w pred}}$'])

marker_size = 5  # Slightly larger markers for better visibility
transparency = 0.4

for idx, cluster_id in enumerate(sorted(set(closest_clusters))):
    base_color_idx = 4 * idx
    indices = [i for i, cl in enumerate(closest_clusters) if cl == cluster_id]

    for i in range(3):
        fig.add_trace(go.Scatter(
            x=indices,
            y=[d_real_values[i] for i in indices],
            mode='markers',
            marker=dict(symbol='circle', size=marker_size, color=f'rgba(0, 0, 0, 1)'),
            name=r'$d_{\text{real}} {\text{Cluster }' + str(cluster_id) + '}$',
            showlegend=(i == 0)
        ), row=1, col=i + 1)

    # Cluster predictions - use X symbols with the full color
    fig.add_trace(go.Scatter(
        x=indices,
        y=[d_cl_pred_values[i] for i in indices],
        mode='markers',
        marker=dict(symbol='x', size=marker_size, color=f'rgba(137, 147, 222, {transparency})'),
        name=r'$d_{\text{cl pred}} {\text{Cluster }' + str(cluster_id) + '}$'
    ), row=1, col=1)

    # Online predictions - use squares with transparency
    fig.add_trace(go.Scatter(
        x=indices,
        y=[d_on_pred_values[i] for i in indices],
        mode='markers',
        marker=dict(symbol='square', size=marker_size, color=f'rgba(85, 128, 172, {transparency})'),
        name=r'$d_{\text{on pred}} {\text{Cluster }' + str(cluster_id) + '}$'
    ), row=1, col=2)

    # Weighted predictions - use diamond symbols with a different transparency
    fig.add_trace(go.Scatter(
        x=indices,
        y=[weighted_pred_values[i] for i in indices],
        mode='markers',
        marker=dict(symbol='diamond', size=marker_size, color=f'rgba(85, 172, 172, {transparency})'),
        name=r'$d_{\text{w pred}} {\text{Cluster }' + str(cluster_id) + '}$'
    ), row=1, col=3)

# Add vertical lines at x = I_PUPA, 2*I_PUPA, ...
for subfigure in range(3):
    for x_val in range(I_PUPA, msg_num, I_PUPA):
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=x_val,
                x1=x_val,
                y0=0,
                y1=50,
                line=dict(color="rgba(128, 128, 128, 0.7)", width=1, dash="dash"),  # Transparent grey
                showlegend=False
            ),
            row=1, col=subfigure + 1
        )

font_size = 24

# Add legend entry for the vertical lines
fig.add_trace(go.Scatter(
    x=[None],  # Dummy point for legend
    y=[None],
    mode='lines',
    line=dict(color="rgba(128, 128, 128, 0.7)", width=1, dash="dash"),
    name=r'$i_{\text{pupa}} \text{Iterations}$',
), col=1, row=1)

# Set y-axis range
fig.update_yaxes(range=[0, 50])

# Update layout with larger font sizes and better spacing
fig.update_layout(
    xaxis2_title="Message Index",
    yaxis_title='delay (ms)',
    height=400,
    width=1500,
    font=dict(size=font_size),
    margin=dict(l=50, r=50, t=50, b=50),  # More top margin for titles
    legend=dict(
        itemsizing='constant',
        font=dict(size=font_size - 4),  # Slightly smaller legend text
        orientation="h",  # Horizontal legend
        yanchor="bottom",
        y=-0.5,  # Position legend below the plots
        xanchor="center",
        x=0.5
    ),
    plot_bgcolor='rgba(240,240,240,0.5)',  # Light grey background
    paper_bgcolor='white'
)


# Increase font size for subplot titles
fig.update_annotations(font_size=30)

# Increase axis label font sizes
fig.update_xaxes(title_font=dict(size=font_size))
fig.update_yaxes(title_font=dict(size=font_size))

# Make tick labels larger
fig.update_xaxes(tickfont=dict(size=font_size))
fig.update_yaxes(tickfont=dict(size=font_size))

fig.show()