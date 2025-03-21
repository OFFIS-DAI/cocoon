from abc import ABC, abstractmethod


class Event(ABC):
    """
    Abstract event class.
    """
    def __init__(self,
                 event_time_ms: int):
        self.event_time_ms = event_time_ms

    @abstractmethod
    def print_event_info(self):
        pass


class MessageArrival(Event):
    """
    Event of message arrival.
    """
    def __init__(self,
                 sender: str,
                 receiver: str,
                 size_B: int,
                 time_send_ms: int,
                 msg_id: int,
                 departure_time_ms=0):
        super().__init__(event_time_ms=time_send_ms)
        self.sender = sender
        self.receiver = receiver
        self.size_B = size_B
        self.msg_id = msg_id
        self.departure_time_ms = departure_time_ms

    def print_event_info(self):
        print(f'{self.msg_id} -- {self.sender} --> {self.size_B} B --> {self.receiver}')


class MessageDeparture(Event):
    """
    Event of message departure.
    """
    def __init__(self,
                 sender: str,
                 receiver: str,
                 size_B: int,
                 arrival_time_ms: int,
                 departure_time_ms: int,
                 msg_id: int):
        super().__init__(departure_time_ms)
        self.sender = sender
        self.receiver = receiver
        self.size_B = size_B
        self.msg_id = msg_id
        self.arrival_time_ms = arrival_time_ms

    def print_event_info(self):
        print(f'{self.event_time_ms}: {self.msg_id} -- completed')
