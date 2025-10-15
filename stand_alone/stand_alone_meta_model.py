import asyncio
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


async def main(test: bool = True):
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
    if test:
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
    meta_model_test = CocoonMetaModel(output_file_name='cocoon_test_data.csv',
                                      mode=CocoonMetaModel.Mode.PRODUCTION,
                                      cluster_distance_threshold=3,
                                      i_pupa=150,
                                      alpha=0.5,
                                      butterfly_threshold_value=0.9,
                                      substitution_priority='none',
                                      substitution_enabled=True)
    meta_model_test.execute_egg_phase(pd.read_csv('cocoon_training_data.csv'))
    await process_fes(fes_df=test_message_df,
                      meta_model=meta_model_test)


if __name__ == "__main__":
    asyncio.run(main(test=False))
