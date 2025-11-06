import asyncio
import time

import pandas as pd
import logging

from integration_environment.network_models.cocoon_meta_model import CocoonMetaModel

logging.getLogger().level = logging.DEBUG


def make_fes(df):
    df["msg_id"] = [f"msg{i}" for i in range(len(df))]

    # Sende-Events (Originalzeit)
    sent_df = df.copy()
    sent_df["event"] = "sent"

    # Empfangs-Events (Zeit + delay_ms)
    recv_df = df.copy()
    recv_df["timestamp"] = recv_df["timestamp"] + pd.to_timedelta(recv_df["delay_ms"], unit="ms")
    recv_df["event"] = "received"

    # Zusammenführen
    fes_df = pd.concat([sent_df, recv_df], ignore_index=True)

    # Sortierung: erst Zeit, dann sent vor received bei gleichem Timestamp
    event_order = pd.CategoricalDtype(categories=["sent", "received"], ordered=True)
    fes_df["event"] = fes_df["event"].astype(event_order)
    fes_df = fes_df.sort_values(["timestamp", "event"]).reset_index(drop=True)

    # Unix-Zeit in Millisekunden (pandas Datetime -> ns -> ms)
    fes_df["unix_ms"] = (fes_df["timestamp"].astype("int64") // 1_000_000)

    return fes_df


async def process_fes(fes_df: pd.DataFrame, meta_model: CocoonMetaModel):
    substitution_occurred = False
    for i, row in fes_df.iterrows():
        receiver = row['receiver']
        sender = row['sender']
        unix_ts = row['unix_ms']  # seconds since epoch (float)
        event_type = row['event']
        msg_id = row['msg_id']
        if event_type == 'sent':
            await meta_model.process_sent_message(sender=sender,
                                                  receiver=receiver,
                                                  payload_size_B=100,
                                                  current_time_ms=unix_ts,
                                                  msg_id=msg_id)
        await meta_model.process_observations()
        if not substitution_occurred and meta_model.mode == CocoonMetaModel.Mode.PRODUCTION:
            substitution_occurred = meta_model.substitution_threshold_reached
            if substitution_occurred:
                print('Substitution at message ID ', msg_id)
        if event_type == 'received':
            meta_model.process_received_message(msg_id=msg_id,
                                                current_time_ms=unix_ts)

    await meta_model.process_observations()
    await meta_model.save_observations()


async def main(include_training_data_generation: bool = True):
    """
    Load message data.
    """
    message_df = pd.read_csv('messages.csv')
    message_df.drop(columns=['timestamp'], inplace=True)
    message_df = message_df.reset_index().rename(columns={"index": "timestamp"})
    message_df["timestamp"] = pd.to_datetime(message_df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    message_df["day"] = message_df["timestamp"].dt.day
    message_df = make_fes(message_df)
    training_message_df = message_df[message_df['day'] == 6]
    test_message_df = message_df[message_df['day'] != 6]

    """
    First, generate training data with messages from the first day.
    """
    if include_training_data_generation:
        meta_model_training = CocoonMetaModel(output_file_name='cocoon_training_data.csv',
                                              mode=CocoonMetaModel.Mode.TRAINING,
                                              cluster_distance_threshold=5,
                                              i_pupa=50,
                                              alpha=0.5,
                                              butterfly_threshold_value=0.8,
                                              substitution_priority='none',
                                              substitution_enabled=False)
        await process_fes(fes_df=training_message_df,
                          meta_model=meta_model_training)

    """
    Second, test with the rest of the data. 
    """
    combined_df = None
    statistic_list = []
    for c_dt in [1, 3, 5]:
        for c_ip in [50, 100, 150]:
            for c_lr in [0.1, 0.5, 0.9]:
                for c_bt in [0.1, 0.5, 0.9]:
                    for c_sp in ['error_level', 'error_trend', 'none']:
                        start_time = time.time()
                        meta_model_test = CocoonMetaModel(output_file_name=f'results/cocoon_test_data.csv',
                                                          mode=CocoonMetaModel.Mode.PRODUCTION,
                                                          cluster_distance_threshold=c_dt,
                                                          i_pupa=c_ip,
                                                          alpha=c_lr,
                                                          butterfly_threshold_value=c_bt,
                                                          substitution_priority=c_sp,
                                                          substitution_enabled=True)
                        meta_model_test.execute_egg_phase(pd.read_csv('cocoon_training_data.csv'))
                        print('Number of clusters: ', len(meta_model_test.model_for_cluster_id))
                        print('Length of DF: ', len(test_message_df))
                        await process_fes(fes_df=test_message_df[:1000],
                                          meta_model=meta_model_test)

                        duration = time.time() - start_time

                        statistic_list.append({
                            'config': f'cdt{c_dt}_cip{c_ip}_clr{c_lr}_cbt{c_bt}_csp{c_sp}',
                            'execution_time_s': duration,
                            'substitution': meta_model_test.substitution_info
                        })

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
                        # Initialize base only once: msg_id + real_delay_ms (same across c_dt)
                        if combined_df is None:
                            combined_df = pd.DataFrame({
                                "msg_id": msg_ids,
                                "real_delay_ms": d_real
                            })
                        # Only prediction columns for this c_dt
                        preds_df = pd.DataFrame({
                            "msg_id": msg_ids,
                            f"cluster_predicted_delay_ms_cdt{c_dt}_cip{c_ip}_clr{c_lr}_cbt{c_bt}_csp{c_sp}": d_cl,
                            f"online_predicted_delay_ms_cdt{c_dt}_cip{c_ip}_clr{c_lr}_cbt{c_bt}_csp{c_sp}": d_on,
                            f"weighted_predicted_delay_ms_cdt{c_dt}_cip{c_ip}_clr{c_lr}_cbt{c_bt}_csp{c_sp}": d_w
                        })
                        # Merge by msg_id; keeps single real_delay_ms
                        combined_df = combined_df.merge(preds_df, on="msg_id", how="left")

    combined_df.to_csv("results/delay_comparison.csv", index=False)
    with open('results/statistics.txt') as f:
        f.write(str(statistic_list))


if __name__ == "__main__":
    asyncio.run(main(include_training_data_generation=False))
