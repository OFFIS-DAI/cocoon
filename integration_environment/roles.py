import asyncio
import logging
import random
from dataclasses import dataclass

from random import choice, expovariate
from string import ascii_uppercase
from typing import List, Optional

from mango import Role, AgentAddress

from integration_environment.messages import *
from integration_environment.results_recorder import ResultsRecorder, ScenarioConfiguration
from integration_environment.scenario_configuration import TrafficConfig

logger = logging.getLogger(__name__)


@dataclass
class SendMessage:
    sender: str
    receiver: AgentAddress
    payload_size_B: int
    time_send_ms: int
    msg_id: str


@dataclass
class ReceiveMessage:
    time_receive_ms: int
    msg_id: str


class ResultsRecorderRole(Role):
    def __init__(self, results_recorder: ResultsRecorder):
        super().__init__()
        self.results_recorder = results_recorder

    def setup(self):
        self.context.subscribe_event(self, SendMessage, self.handle_event_send_message)
        self.context.subscribe_event(self, ReceiveMessage, self.handle_event_receive_message)

    def handle_event_send_message(self, event, source):
        self.results_recorder.record_message_send_event(event)

    def handle_event_receive_message(self, event, source):
        self.results_recorder.record_message_receive_event(event)


def generate_payload_with_byte_size(byte_size: int):
    return ''.join(choice(ascii_uppercase) for _ in range(byte_size))


class ConstantBitrateSenderRole(Role):
    def __init__(self, receiver_addresses: list, scenario_config: ScenarioConfiguration):
        super().__init__()
        if scenario_config.traffic_configuration == TrafficConfig.cbr_broadcast_1_mps:
            self.frequency_s = 1  # every second
        elif scenario_config.traffic_configuration == TrafficConfig.cbr_broadcast_1_mpm:
            self.frequency_s = 60  # every 60 seconds
        elif scenario_config.traffic_configuration == TrafficConfig.cbr_broadcast_4_mph:
            self.frequency_s = 60 * 15  # every 15 minutes
        else:
            self.frequency_s = 1  # default
        self.receiver_addresses = receiver_addresses
        self.scenario_configuration = scenario_config

        self._message_counter = 0
        self._periodic_task = None

    def setup(self):
        pass

    def on_start(self):
        pass

    def on_ready(self):
        self._periodic_task = self.context.schedule_periodic_task(self.send_message, self.frequency_s)

    async def send_message(self):
        if self.context.current_timestamp == 0:
            return  # skip the first iteration
        logger.debug(f'Send message at time {self.context.current_timestamp}')
        time_send = round(self.context.current_timestamp * 1000)
        for receiver in self.receiver_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_{self._message_counter}'

            # send message
            payload = generate_payload_with_byte_size(self.scenario_configuration.payload_size.value)
            await self.context.send_message(
                TrafficMessage(msg_id=msg_id,
                               payload=payload),
                receiver_addr=receiver,
            )
            # initialize event for results recording
            event = SendMessage(sender=self.context.addr,
                                receiver=receiver,
                                msg_id=msg_id,
                                payload_size_B=self.scenario_configuration.payload_size.value,
                                time_send_ms=time_send)
            self.context.emit_event(event=event, event_source=self)

            self._message_counter += 1

    async def on_stop(self):
        """Clean shutdown - cancel the periodic task."""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass


class ReceiverRole(Role):
    def __init__(self):
        super().__init__()
        self.received_messages = []

    def setup(self):
        self.context.subscribe_message(self, self.handle_traffic_message,
                                       lambda content, meta: isinstance(content, TrafficMessage))

    def handle_traffic_message(self, content: TrafficMessage, meta):
        logger.debug(f'Traffic Message received at time {self.context.current_timestamp}.')
        # initialize event for results recording
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)
        self.received_messages.append(content)

    def on_start(self):
        pass

    def on_ready(self):
        pass

    async def on_stop(self):
        pass


class FlexAgentRole(Role):
    def __init__(self, aggregator_address: AgentAddress, scenario_config: ScenarioConfiguration,
                 can_provide_power: bool = True, flexibility_value: int = 0):
        super().__init__()

        self.aggregator_address = aggregator_address
        self.scenario_configuration = scenario_config

        # flag if this is an agent that cannot provide the requested power value
        self.can_provide_power = can_provide_power
        self.sent_notification_infeasible_power = False
        self.calculation_time = random.random()
        self.flexibility_value = flexibility_value

        self.infeasible_requests = []

        self._message_counter = 0
        self._periodic_task = None

    def setup(self):
        self.context.subscribe_message(self, self.handle_fixed_power,
                                       lambda content, meta: isinstance(content, FixedPowerMessage))
        self.context.subscribe_message(self, self.handle_power_deviation_request,
                                       lambda content, meta: isinstance(content, PowerDeviationRequest))

    def on_start(self):
        pass

    def on_ready(self):
        # start after 5 minutes
        self.context.schedule_timestamp_task(self.schedule_periodic_task(), timestamp=5 * 60)

    async def schedule_periodic_task(self):
        # send planning data every 15 minutes
        self._periodic_task = self.context.schedule_periodic_task(self.send_planning_data, 15 * 60)

    def handle_fixed_power(self, content: FixedPowerMessage, meta):
        logger.debug(f'Received fixed power of {content.power_value} W '
                     f'at time {self.context.current_timestamp / 60} minutes.')
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)

        self.context.schedule_instant_task(self.mock_calculation_time())
        if (self.context.current_timestamp / 60) >= content.t_start:
            # cannot control asset anymore
            print('Infeasible request!')
            self.infeasible_requests.append({'power_value': content.power_value, 't_start': content.t_start,
                                             'cur_time': self.context.current_timestamp/60})
            return

        if self.can_provide_power or self.sent_notification_infeasible_power:
            return
        self.context.schedule_instant_task(self.send_infeasible_power_notification(requested_power=content.power_value,
                                                                                   t_start=content.t_start))

    def handle_power_deviation_request(self, content: PowerDeviationRequest, meta):
        logger.debug(f'Received power deviation request '
                     f'at time {self.context.current_timestamp / 60} minutes.')
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)

        self.context.schedule_instant_task(
            self.send_power_deviation_response(power_deviation_requested=content.power_deviation_value_requested,
                                               t_start=content.t_start))

    async def send_power_deviation_response(self, power_deviation_requested: int, t_start: int):
        time_send = round(self.context.current_timestamp * 1000)
        msg_id = f'{self.context.addr.protocol_addr}_power_deviation_response_{self._message_counter}'

        await self.context.send_message(
            PowerDeviationResponse(msg_id=msg_id,
                                   power_deviation_value_requested=power_deviation_requested,
                                   power_deviation_value_available=self.flexibility_value,
                                   t_start=t_start),
            receiver_addr=self.aggregator_address,
        )
        # initialize event for results recording
        event = SendMessage(sender=self.context.addr,
                            receiver=self.aggregator_address,
                            msg_id=msg_id,
                            payload_size_B=self.scenario_configuration.payload_size.value,
                            time_send_ms=time_send)
        self.context.emit_event(event=event, event_source=self)

        self._message_counter += 1

    async def send_infeasible_power_notification(self, requested_power: int, t_start: int):
        logger.debug(f'Send infeasible power notification '
                     f'at time {self.context.current_timestamp / 60} minutes.')
        time_send = round(self.context.current_timestamp * 1000)
        msg_id = f'{self.context.addr.protocol_addr}_infeasible_power_{self._message_counter}'

        await self.context.send_message(
            InfeasiblePowerNotification(msg_id=msg_id,
                                        power_value_requested=requested_power,
                                        power_value_available=int(requested_power * 0.9),
                                        t_start=t_start),
            receiver_addr=self.aggregator_address,
        )
        # initialize event for results recording
        event = SendMessage(sender=self.context.addr,
                            receiver=self.aggregator_address,
                            msg_id=msg_id,
                            payload_size_B=self.scenario_configuration.payload_size.value,
                            time_send_ms=time_send)
        self.context.emit_event(event=event, event_source=self)

        self._message_counter += 1
        self.sent_notification_infeasible_power = True

    async def mock_calculation_time(self):
        await asyncio.sleep(self.calculation_time)

    async def send_planning_data(self):
        if self.context.current_timestamp == 0:
            return  # skip the first iteration
        logger.debug(f'Send planning data at time {self.context.current_timestamp / 60} minutes.')
        time_send = round(self.context.current_timestamp * 1000)
        msg_id = f'{self.context.addr.protocol_addr}_planning_data_{self._message_counter}'

        await self.context.send_message(
            PlanningDataMessage(msg_id=msg_id,
                                baseline=0,
                                min_p=0,
                                max_p=1),
            receiver_addr=self.aggregator_address,
        )
        # initialize event for results recording
        event = SendMessage(sender=self.context.addr,
                            receiver=self.aggregator_address,
                            msg_id=msg_id,
                            payload_size_B=self.scenario_configuration.payload_size.value,
                            time_send_ms=time_send)
        self.context.emit_event(event=event, event_source=self)

        self._message_counter += 1

    async def on_stop(self):
        """Clean shutdown - cancel the periodic task."""
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass


class AggregatorAgentRole(Role):
    def __init__(self, flex_agent_addresses: Optional[List[AgentAddress]]):
        super().__init__()
        self.flex_agent_addresses = flex_agent_addresses

        self.individual_baselines = {}

        self._fix_power_tasks = []
        self.x_minute_time_window = 5

        self._message_counter = 0

        self.power_deviation_responses = {}

    def setup(self):
        self.context.subscribe_message(self, self.handle_planning_data,
                                       lambda content, meta: isinstance(content, PlanningDataMessage))
        self.context.subscribe_message(self, self.handle_infeasible_power_notification,
                                       lambda content, meta: isinstance(content, InfeasiblePowerNotification))
        self.context.subscribe_message(self, self.handle_power_deviation_response,
                                       lambda content, meta: isinstance(content, PowerDeviationResponse))

    def on_ready(self):
        # send planning data every 15 minutes
        time_stamps = [(15 * i - self.x_minute_time_window) * 60 for i in range(4)]
        for t in time_stamps:
            self._fix_power_tasks.append(self.context.schedule_timestamp_task(timestamp=t,
                                                                              coroutine=self.send_fixed_power()))

    def handle_power_deviation_response(self, content: PowerDeviationResponse, meta):
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)

        self.power_deviation_responses[meta['sender_addr']] = content.power_deviation_value_available

        if len(self.power_deviation_responses) == len(self.flex_agent_addresses):
            print('Received all responses')
            total_received_deviation = sum(self.power_deviation_responses.values())
            print(f'Requested deviation of {content.power_deviation_value_requested},'
                  f' received {total_received_deviation}')
            assigned_deviation = 0
            requested_deviation = content.power_deviation_value_requested
            for dev_agent, dev_value in self.power_deviation_responses.items():
                if assigned_deviation >= requested_deviation:
                    break
                if (assigned_deviation + dev_value) < requested_deviation:
                    # take everything
                    assigned_deviation += dev_value
                    self.individual_baselines[dev_agent] += dev_value
                else:
                    # only take part of deviation
                    needed_dev = requested_deviation - assigned_deviation
                    self.individual_baselines[dev_agent] += needed_dev
                    break
            self.context.schedule_instant_task(self.send_fixed_power(t_start=content.t_start))

    def handle_infeasible_power_notification(self, content: InfeasiblePowerNotification, meta):
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)

        self.context.schedule_instant_task(
            self.request_power_deviation(
                requested_power_deviation=content.power_value_requested - content.power_value_available,
                t_start=content.t_start))

    async def request_power_deviation(self, requested_power_deviation: int, t_start: int):
        time_send = round(self.context.current_timestamp * 1000)

        for flex_agent in self.flex_agent_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_power_deviation_request_{self._message_counter}'

            await self.context.send_message(
                PowerDeviationRequest(msg_id=msg_id,
                                      power_deviation_value_requested=requested_power_deviation,
                                      t_start=t_start),
                receiver_addr=flex_agent,
            )
            # initialize event for results recording
            event = SendMessage(sender=self.context.addr,
                                receiver=flex_agent,
                                msg_id=msg_id,
                                payload_size_B=8,
                                time_send_ms=time_send)
            self.context.emit_event(event=event, event_source=self)

            self._message_counter += 1

    async def send_fixed_power(self, t_start: int = None):
        if self.context.current_timestamp == 0:
            return
        logger.debug(f'Aggregator sends fixed power at time {self.context.current_timestamp / 60} minutes.')
        time_send = round(self.context.current_timestamp * 1000)
        if not t_start:
            next_t_start = int((self.context.current_timestamp / 60) + (15 - ((self.context.current_timestamp / 60) % 15)))
        else:
            next_t_start = t_start

        for flex_agent in self.flex_agent_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_fixed_power_{self._message_counter}'
            if flex_agent.protocol_addr not in self.individual_baselines:
                logger.warning(f'No baseline available for {flex_agent.protocol_addr}.')
                continue

            await self.context.send_message(
                FixedPowerMessage(msg_id=msg_id,
                                  power_value=self.individual_baselines[flex_agent.protocol_addr],
                                  t_start=next_t_start),
                receiver_addr=flex_agent,
            )
            # initialize event for results recording
            event = SendMessage(sender=self.context.addr,
                                receiver=flex_agent,
                                msg_id=msg_id,
                                payload_size_B=8,
                                time_send_ms=time_send)
            self.context.emit_event(event=event, event_source=self)

            self._message_counter += 1

    def handle_planning_data(self, content: PlanningDataMessage, meta):
        logger.debug(f'Planning Data received at time {self.context.current_timestamp / 60} minutes.')
        # initialize event for results recording
        event = ReceiveMessage(msg_id=content.msg_id,
                               time_receive_ms=round(self.context.current_timestamp * 1000))
        self.context.emit_event(event=event, event_source=self)

        self.individual_baselines[meta['sender_addr']] = content.baseline

    def on_start(self):
        pass

    async def on_stop(self):
        for t in self._fix_power_tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass


class PoissonSenderRole(Role):
    def __init__(self, receiver_addresses: list, scenario_config: ScenarioConfiguration):
        super().__init__()
        self.receiver_addresses = receiver_addresses
        self.scenario_configuration = scenario_config
        self._message_counter = 0
        self._running = False
        self._scheduled_tasks = []

        # Configure lambda rate based on traffic configuration
        self.lambda_rate = self._get_lambda_rate_from_config()

    def _get_lambda_rate_from_config(self) -> float:
        """Map traffic configuration to Poisson rate parameter."""
        # You can extend this mapping based on your TrafficConfig enum
        if self.scenario_configuration.traffic_configuration == TrafficConfig.poisson_broadcast_1_mps:
            return 1.0  # 1 message per second on average
        elif self.scenario_configuration.traffic_configuration == TrafficConfig.poisson_broadcast_1_mpm:
            return 1.0 / 60.0  # 1 message per minute on average
        elif self.scenario_configuration.traffic_configuration == TrafficConfig.poisson_broadcast_4_mph:
            return 4.0 / 3600.0  # 4 messages per hour on average
        else:
            return 1.0  # Default: 1 message per second

    def setup(self):
        pass

    def on_start(self):
        pass

    def on_ready(self):
        self._running = True
        self._schedule_all_poisson_events()

    def _schedule_all_poisson_events(self):
        """Schedule all Poisson-distributed message sending events."""
        current_time = self.context.current_timestamp

        # Schedule events for the entire simulation duration
        # You may want to set a reasonable upper bound based on your simulation length
        max_simulation_time = current_time + 3600  # Example: 1 hour from start

        while current_time < max_simulation_time and self._running:
            # Generate exponentially distributed inter-arrival time
            inter_arrival_time = expovariate(self.lambda_rate)
            current_time += inter_arrival_time

            # Schedule the message sending task
            task = self.context.schedule_timestamp_task(
                timestamp=current_time,
                coroutine=self._send_message_to_all_receivers()
            )
            self._scheduled_tasks.append(task)

    async def _send_message_to_all_receivers(self):
        """Send message to all receivers with event recording."""
        if not self._running:
            return

        time_send = round(self.context.current_timestamp * 1000)

        for receiver in self.receiver_addresses:
            msg_id = f'{self.context.addr.protocol_addr}_{self._message_counter}'

            await self.context.send_message(
                TrafficMessage(msg_id=msg_id, payload=self.scenario_configuration.payload_size.value),
                receiver_addr=receiver,
            )

            event = SendMessage(
                sender=self.context.addr,
                receiver=receiver,
                msg_id=msg_id,
                payload_size_B=self.scenario_configuration.payload_size.value,
                time_send_ms=time_send
            )
            self.context.emit_event(event=event, event_source=self)
            self._message_counter += 1

        logger.debug(f'Sent Poisson message at time {self.context.current_timestamp}, '
                     f'lambda rate: {self.lambda_rate}, message count: {self._message_counter}')

    async def on_stop(self):
        """Clean shutdown."""
        self._running = False
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
