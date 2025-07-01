import time
import unittest
import logging

from integration_environment.network_models.detailed_network_model import OmnetConnection, DetailedNetworkModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestOmnetSocketConnection(unittest.TestCase):
    """Test the OMNeT++ socket connection"""

    def test_message_dispatch(self):
        # Create a connection
        connection = OmnetConnection(inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                     config_name='General',
                                     omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')
        # Start OMNeT++ and connect socket
        connection.initialize()

        time.sleep(1)
        message_list = []
        msg_ids = []
        message_list.append({
            'sender': 'node2',
            'receiver': 'node1',
            'size_B': 1024,
            'time_send_ms': 10,
            'msg_id': f'msg_25'
        })
        msg_ids.append(f'msg_25')
        payload = {
            "messages": message_list,
            "max_advance": 15
        }

        try:
            # Send several messages to simulate
            self.assertTrue(
                connection.send_message_to_omnet(payload=payload, msg_ids=msg_ids))
            time.sleep(10)
            print(connection.get_all_messages())
            self.assertTrue(connection.send_termination_signal())

        finally:
            # Clean up
            connection.cleanup()

    def test_max_advance(self):
        """Test that OMNeT++ respects max advance limits"""
        # Create a connection
        connection = OmnetConnection(
            inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
            config_name='General',
            omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/'
        )

        try:
            # Start OMNeT++ and connect socket
            connection.initialize()
            time.sleep(1)

            # Test 1: Send messages with different timing and max_advance limits
            logger.info("Test 1: Testing basic max advance functionality")

            # First batch: Schedule messages at different times with small max_advance
            message_list_1 = [
                {
                    'sender': 'node1',
                    'receiver': 'node2',
                    'size_B': 512,
                    'time_send_ms': 5,
                    'msg_id': 'msg_1'
                },
                {
                    'sender': 'node2',
                    'receiver': 'node1',
                    'size_B': 1024,
                    'time_send_ms': 20,  # This is beyond first max_advance
                    'msg_id': 'msg_2'
                }
            ]

            payload_1 = {
                "messages": message_list_1,
                "max_advance": 10  # Only allow advancement up to 10ms
            }

            # Send first batch
            self.assertTrue(
                connection.send_message_to_omnet(
                    payload=payload_1,
                    msg_ids=['msg_1', 'msg_2']
                )
            )

            # Wait briefly and check responses
            time.sleep(2)
            messages = connection.get_all_messages()
            logger.info(f"Messages after first batch: {messages}")

            # We should get acknowledgment that messages were scheduled
            self.assertTrue(any('SCHEDULED' in msg for msg in messages))

            # Test 2: Advance time window to allow second message
            logger.info("Test 2: Advancing time window")

            # Send empty message list but increase max_advance
            payload_2 = {
                "messages": [],
                "max_advance": 25  # Now allow up to 25ms
            }

            self.assertTrue(
                connection.send_message_to_omnet(payload=payload_2, msg_ids=[])
            )

            # Wait for processing
            time.sleep(3)
            messages = connection.get_all_messages()
            logger.info(f"Messages after time advance: {messages}")

            # Test 3: Multiple small advances
            logger.info("Test 3: Testing incremental advances")

            # Schedule messages spread across time
            message_list_3 = [
                {
                    'sender': 'node1',
                    'receiver': 'node2',
                    'size_B': 256,
                    'time_send_ms': 30,
                    'msg_id': 'msg_3'
                },
                {
                    'sender': 'node2',
                    'receiver': 'node1',
                    'size_B': 256,
                    'time_send_ms': 40,
                    'msg_id': 'msg_4'
                },
                {
                    'sender': 'node1',
                    'receiver': 'node2',
                    'size_B': 256,
                    'time_send_ms': 50,
                    'msg_id': 'msg_5'
                }
            ]

            # Send with limited advance
            payload_3 = {
                "messages": message_list_3,
                "max_advance": 35  # Only process first message
            }

            self.assertTrue(
                connection.send_message_to_omnet(
                    payload=payload_3,
                    msg_ids=['msg_3', 'msg_4', 'msg_5']
                )
            )

            time.sleep(2)

            # Advance to allow second message
            payload_4 = {
                "messages": [],
                "max_advance": 45
            }

            self.assertTrue(
                connection.send_message_to_omnet(payload=payload_4, msg_ids=[])
            )

            time.sleep(2)

            # Advance to allow third message
            payload_5 = {
                "messages": [],
                "max_advance": 55
            }

            self.assertTrue(
                connection.send_message_to_omnet(payload=payload_5, msg_ids=[])
            )

            time.sleep(2)

            # Test 4: Verify message delivery notifications
            logger.info("Test 4: Checking message deliveries")

            all_messages = connection.get_all_messages()
            logger.info(f"All messages received: {all_messages}")

            # Count RECEIVED messages
            received_count = sum(1 for msg in all_messages if 'RECEIVED' in msg)
            logger.info(f"Total RECEIVED messages: {received_count}")

            # We should have received some delivery notifications
            self.assertGreater(received_count, 0, "Should have received some message deliveries")

            # Test 5: Large time jump
            logger.info("Test 5: Testing large time advance")

            message_list_6 = [
                {
                    'sender': 'node1',
                    'receiver': 'node2',
                    'size_B': 2048,
                    'time_send_ms': 1000,  # 1 second in the future
                    'msg_id': 'msg_6'
                }
            ]

            payload_6 = {
                "messages": message_list_6,
                "max_advance": 500  # Don't allow it yet
            }

            self.assertTrue(
                connection.send_message_to_omnet(
                    payload=payload_6,
                    msg_ids=['msg_6']
                )
            )

            time.sleep(1)

            # Now advance to allow it
            payload_7 = {
                "messages": [],
                "max_advance": 1500  # Allow up to 1.5 seconds
            }

            self.assertTrue(
                connection.send_message_to_omnet(payload=payload_7, msg_ids=[])
            )

            time.sleep(3)

            # Final check of all messages
            final_messages = connection.get_all_messages()
            logger.info(f"Final message count: {len(final_messages)}")

            # Verify we got acknowledgments for all our messages
            scheduled_messages = [msg for msg in final_messages if 'SCHEDULED' in msg]
            self.assertGreater(len(scheduled_messages), 0, "Should have scheduled messages")

            # Send termination signal
            self.assertTrue(connection.send_termination_signal())

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            raise
        finally:
            # Clean up
            connection.cleanup()



if __name__ == '__main__':
    unittest.main()
