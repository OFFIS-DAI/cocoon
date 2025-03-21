from src.communication_network_graph import CommunicationNetworkGraph
from src.events import Event, MessageArrival, MessageDeparture
from src.scenario_configuration import ScenarioConfiguration
from src.utils.util import LoggingOption


class FutureEventSet:
    """
    Handles the future event set of the simulation.
    """

    def __init__(self):
        self.event_set = {}

    def insert_event(self,
                     event: Event):
        """
        Method to insert an event into the FES.
        Events are saves in a python dictionary with arrival times as keys.
        :param event: event to be scheduled.
        """
        arr_time = event.event_time_ms
        if arr_time not in self.event_set.keys():
            self.event_set[arr_time] = []
            keys = list(self.event_set.keys())
            keys.sort()
            self.event_set = {i: self.event_set[i] for i in keys}
        self.event_set[arr_time].append(event)

    def is_empty(self):
        """
        Returns if FES is empty.
        :return: True, if empty.
        """
        return len(self.event_set) == 0

    def get_next_events(self) -> tuple[int, list]:
        """
        Returns a tuple of the event time and event list for the next events.
        :return: tuple[event time, event list]
        """
        if self.is_empty():
            return 0, []
        next_event_time = list(self.event_set.keys())[0]
        next_events = self.event_set[next_event_time]
        del self.event_set[next_event_time]
        return next_event_time, next_events

    def print_FES(self):
        """
        Prints future event set.
        """
        print('Future Event Set')
        print('-' * 50)
        for arr_time, events in self.event_set.items():
            print('time : ', arr_time, ' ms')
            for event in events:
                event.print_event_info()


class Cocoon:
    """
    Main class of the meta-model.
    Holds the configuration of the scenario, a FES and the network graph.
    """

    def __init__(self,
                 scenario_configuration: ScenarioConfiguration,
                 training_mode_on=True,
                 logging_option=LoggingOption.INFO):
        self.scenario_configuration = scenario_configuration
        self.training_mode_on = training_mode_on
        self.simulation_time = 0
        self.future_event_set = FutureEventSet()
        self.logging_option = logging_option

        self.communication_network_graph = CommunicationNetworkGraph(logging_option=self.logging_option)

    def schedule_event(self,
                       event: Event):
        """
        Schedules an event at the FES.
        """
        self.future_event_set.insert_event(event=event)

    def run(self):
        """
        Runs simulation.
        While the future event set is not empty:
            - get next events
            - update graph on events
            - if not in training: predict departure
        - if in training: save data for predictions
        """
        while True:
            if self.future_event_set.is_empty():
                break
            arrival_time, events = self.future_event_set.get_next_events()
            if self.simulation_time > arrival_time:
                raise ValueError('Error in simulation time.')
            self.simulation_time = arrival_time

            # iterate over currently scheduled events
            for event in events:
                # if event is of type message -> add to communication graph and get link predictor
                if isinstance(event, MessageArrival):
                    sender_node, receiver_node, communication_link = \
                        (self.communication_network_graph.
                         check_or_initialize_model_for_message_transit(sender=event.sender,
                                                                       receiver=event.receiver))
                    communication_link.update_states(event)
                    if self.training_mode_on:
                        communication_link.save_state_data(message_arrival=event)
                    communication_link.process_message_arrival_event(message_arrival=event)
                    if not self.training_mode_on:
                        communication_link.predict_message_departure(event)
                if isinstance(event, MessageDeparture):
                    communication_link = (self.communication_network_graph.
                                          get_link_by_sender_and_receiver_names(sender=event.sender,
                                                                                receiver=event.receiver))
                    if communication_link is not None:
                        communication_link.process_message_departure_event(event)
                        if self.logging_option == LoggingOption.DEBUG:
                            event.print_event_info()
        if self.training_mode_on:
            state_data_df = self.communication_network_graph.get_state_data_from_links()
            return state_data_df
