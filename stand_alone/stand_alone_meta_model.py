"""
Standalone COCOON Meta-Model Usage Example.

This module demonstrates how to use the COCOON meta-model without the full
MANGO agent framework. It's useful for:
- Testing the meta-model with CSV-based message data
- Evaluating different hyperparameter configurations
- Generating training data for the EGG phase

Example:
    python stand_alone_meta_model.py

Author: Malin Radtke (OFFIS)
License: MIT
"""

import asyncio
import logging
import time

import pandas as pd

from integration_environment.network_models.cocoon_meta_model import CocoonMetaModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_fes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a Future Event Set (FES) from message data.

    Converts a DataFrame with message send times and delays into a sorted
    event stream containing both send and receive events.

    Args:
        df: DataFrame with columns: sender, receiver, delay_ms, timestamp

    Returns:
        DataFrame with events sorted by timestamp, containing:
        - msg_id: Unique message identifier
        - event: 'sent' or 'received'
        - unix_ms: Event timestamp in milliseconds
    """
    df["msg_id"] = [f"msg{i}" for i in range(len(df))]

    # Send events (original time)
    sent_df = df.copy()
    sent_df["event"] = "sent"

    # Receive events (time + delay_ms)
    recv_df = df.copy()
    recv_df["timestamp"] = recv_df["timestamp"] + pd.to_timedelta(recv_df["delay_ms"], unit="ms")
    recv_df["event"] = "received"

    # Combine send and receive events
    fes_df = pd.concat([sent_df, recv_df], ignore_index=True)

    # Sort: first by time, then 'sent' before 'received' for same timestamp
    event_order = pd.CategoricalDtype(categories=["sent", "received"], ordered=True)
    fes_df["event"] = fes_df["event"].astype(event_order)
    fes_df = fes_df.sort_values(["timestamp", "event"]).reset_index(drop=True)

    # Convert to Unix time in milliseconds
    fes_df["unix_ms"] = (fes_df["timestamp"].astype("int64") // 1_000_000)

    return fes_df


async def process_fes(fes_df: pd.DataFrame, meta_model: CocoonMetaModel) -> None:
    """
    Process a Future Event Set through the meta-model.

    Args:
        fes_df: Event DataFrame from make_fes()
        meta_model: Initialized CocoonMetaModel instance
    """
    substitution_occurred = False

    for i, row in fes_df.iterrows():
        receiver = row['receiver']
        sender = row['sender']
        unix_ts = row['unix_ms']
        event_type = row['event']
        msg_id = row['msg_id']

        if event_type == 'sent':
            await meta_model.process_sent_message(
                sender=sender,
                receiver=receiver,
                payload_size_B=100,
                current_time_ms=unix_ts,
                msg_id=msg_id
            )

        await meta_model.process_observations()

        if not substitution_occurred and meta_model.mode == CocoonMetaModel.Mode.PRODUCTION:
            substitution_occurred = meta_model.substitution_threshold_reached
            if substitution_occurred:
                logger.info(f'Substitution triggered at message ID: {msg_id}')

        if event_type == 'received':
            meta_model.process_received_message(
                msg_id=msg_id,
                current_time_ms=unix_ts
            )

    await meta_model.process_observations()
    await meta_model.save_observations()


async def main(include_training_data_generation: bool = True) -> None:
    """
    Main function demonstrating standalone meta-model usage.

    Args:
        include_training_data_generation: If True, generate training data first
    """
    # Load message data
    logger.info("Loading message data...")
    message_df = pd.read_csv('messages.csv')
    message_df.drop(columns=['timestamp'], inplace=True)
    message_df = message_df.reset_index().rename(columns={"index": "timestamp"})
    message_df["timestamp"] = pd.to_datetime(message_df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    message_df["day"] = message_df["timestamp"].dt.day
    message_df = make_fes(message_df)

    # Split data: use first day for training, rest for testing
    training_message_df = message_df[message_df['day'] == 6]
    test_message_df = message_df[message_df['day'] != 6]

    # Step 1: Generate training data (EGG phase preparation)
    if include_training_data_generation:
        logger.info("Generating training data...")
        meta_model_training = CocoonMetaModel(
            output_file_name='cocoon_training_data.csv',
            mode=CocoonMetaModel.Mode.TRAINING,
            cluster_distance_threshold=5,
            i_pupa=50,
            alpha=0.5,
            butterfly_threshold_value=0.8,
            substitution_priority='none',
            substitution_enabled=False
        )
        await process_fes(fes_df=training_message_df, meta_model=meta_model_training)

    # Step 2: Test with different hyperparameter configurations
    logger.info("Testing meta-model with different configurations...")
    combined_df = None
    statistic_list = []

    # Hyperparameter grid search
    for c_dt in [1, 3, 5]:  # cluster_distance_threshold
        for c_ip in [50, 100, 150]:  # i_pupa (batch size)
            for c_lr in [0.1, 0.5, 0.9]:  # alpha (learning rate)
                for c_bt in [0.1, 0.5, 0.9]:  # butterfly_threshold_value
                    for c_sp in ['error_level', 'error_trend', 'none']:  # substitution_priority
                        start_time = time.time()

                        meta_model_test = CocoonMetaModel(
                            output_file_name='results/cocoon_test_data.csv',
                            mode=CocoonMetaModel.Mode.PRODUCTION,
                            cluster_distance_threshold=c_dt,
                            i_pupa=c_ip,
                            alpha=c_lr,
                            butterfly_threshold_value=c_bt,
                            substitution_priority=c_sp,
                            substitution_enabled=True
                        )

                        # Execute EGG phase with pre-generated training data
                        meta_model_test.execute_egg_phase(pd.read_csv('cocoon_training_data.csv'))

                        logger.info(f'Config: cdt={c_dt}, ip={c_ip}, lr={c_lr}, bt={c_bt}, sp={c_sp}')
                        logger.info(f'Number of clusters: {len(meta_model_test.model_for_cluster_id)}')

                        # Process test messages
                        await process_fes(
                            fes_df=test_message_df[:1000],
                            meta_model=meta_model_test
                        )

                        duration = time.time() - start_time

                        statistic_list.append({
                            'config': f'cdt{c_dt}_cip{c_ip}_clr{c_lr}_cbt{c_bt}_csp{c_sp}',
                            'execution_time_s': duration,
                            'substitution': meta_model_test.substitution_info
                        })

                        # Collect predictions for analysis
                        msg_ids = []
                        d_real = []
                        d_cl = []
                        d_on = []
                        d_w = []

                        for m_id, m in meta_model_test.message_observations.items():
                            msg_ids.append(m_id)
                            d_real.append(test_message_df[test_message_df['msg_id'] == m_id]['delay_ms'].values[0])
                            d_cl.append(m.cluster_predicted_delay_ms)
                            d_on.append(m.online_predicted_delay_ms)
                            d_w.append(m.weighted_predicted_delay_ms)

                        # Initialize base DataFrame on first iteration
                        if combined_df is None:
                            combined_df = pd.DataFrame({
                                "msg_id": msg_ids,
                                "real_delay_ms": d_real
                            })

                        # Add prediction columns for this configuration
                        config_suffix = f"cdt{c_dt}_cip{c_ip}_clr{c_lr}_cbt{c_bt}_csp{c_sp}"
                        preds_df = pd.DataFrame({
                            "msg_id": msg_ids,
                            f"cluster_predicted_delay_ms_{config_suffix}": d_cl,
                            f"online_predicted_delay_ms_{config_suffix}": d_on,
                            f"weighted_predicted_delay_ms_{config_suffix}": d_w
                        })
                        combined_df = combined_df.merge(preds_df, on="msg_id", how="left")

    # Save results
    combined_df.to_csv("results/delay_comparison.csv", index=False)
    with open('results/statistics.txt', 'w') as f:
        f.write(str(statistic_list))

    logger.info("Results saved to results/delay_comparison.csv and results/statistics.txt")


if __name__ == "__main__":
    asyncio.run(main(include_training_data_generation=False))
