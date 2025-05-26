import math
from dataclasses import dataclass
from typing import Dict, Optional


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
        # TODO: update node state

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


class CocoonNetworkGraph:
    def __init__(self):
        self.nodes = {}  # {node_name: CocoonNetworkNode object}
        self.network_state = NetworkState()

        # Global message tracking - this is the single source of truth
        self.all_messages_by_id: Dict[str, Message] = {}

    def get_or_initialize_node(self, node_name) -> CocoonNetworkNode:
        if node_name not in self.nodes:
            self.nodes[node_name] = CocoonNetworkNode(name=node_name)
        return self.nodes[node_name]

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
            print('registered message: ', msg_id)

        # TODO: update network state

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

        # TODO: update network state
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
    def __init__(self):
        self.network_graph = CocoonNetworkGraph()

    def process_sent_message(self, sender: str, receiver: str,
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
            print(f"Failed to process received message {msg_id}")

    def get_message_status(self, msg_id: str) -> Optional[Message]:
        """Get the current status of a message by its ID."""
        return self.network_graph.get_message_by_id(msg_id)

    def get_network_statistics(self) -> dict:
        """Get comprehensive network statistics."""
        completed_messages = self.network_graph.get_completed_messages()
        messages_in_transit = self.network_graph.get_messages_in_transit()

        # Calculate delay statistics for completed messages
        delays = [msg.delay_ms for msg in completed_messages if msg.delay_ms != math.inf]

        stats = {
            'total_nodes': len(self.network_graph.nodes),
            'total_messages_sent': len(self.network_graph.all_messages_by_id),
            'messages_completed': len(completed_messages),
            'messages_in_transit': len(messages_in_transit),
            'message_completion_rate': len(completed_messages) / max(len(self.network_graph.all_messages_by_id), 1),
        }

        if delays:
            stats.update({
                'average_delay_ms': sum(delays) / len(delays),
                'min_delay_ms': min(delays),
                'max_delay_ms': max(delays),
                'total_delays_measured': len(delays)
            })

        return stats

    def print_network_summary(self):
        """Print a formatted summary of network statistics."""
        stats = self.get_network_statistics()

        print("\n" + "=" * 50)
        print("COCOON META-MODEL NETWORK SUMMARY")
        print("=" * 50)
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Messages Sent: {stats['total_messages_sent']}")
        print(f"Messages Completed: {stats['messages_completed']}")
        print(f"Messages In Transit: {stats['messages_in_transit']}")
        print(f"Message Completion Rate: {stats['message_completion_rate']:.2%}")

        if 'average_delay_ms' in stats:
            print(f"\nDelay Statistics:")
            print(f"  Average Delay: {stats['average_delay_ms']:.2f} ms")
            print(f"  Min Delay: {stats['min_delay_ms']:.2f} ms")
            print(f"  Max Delay: {stats['max_delay_ms']:.2f} ms")
            print(f"  Total Delays Measured: {stats['total_delays_measured']}")

        print("=" * 50 + "\n")