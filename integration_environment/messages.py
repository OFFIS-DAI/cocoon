from dataclasses import dataclass

from mango import json_serializable


@json_serializable
@dataclass
class TrafficMessage:
    msg_id: str
    payload: str


@json_serializable
@dataclass
class PlanningDataMessage:
    msg_id: str
    baseline: int
    min_p: int
    max_p: int


@json_serializable
@dataclass
class FixedPowerMessage:
    msg_id: str
    power_value: int
    t_start: int


@json_serializable
@dataclass
class InfeasiblePowerNotification:
    msg_id: str
    power_value_requested: int
    power_value_available: int
    t_start: int


@json_serializable
@dataclass
class PowerDeviationRequest:
    msg_id: str
    power_deviation_value_requested: int
    t_start: int


@json_serializable
@dataclass
class PowerDeviationResponse:
    msg_id: str
    power_deviation_value_requested: int
    power_deviation_value_available: int
    t_start: int


deer_message_classes = [PlanningDataMessage, FixedPowerMessage, InfeasiblePowerNotification,
                        PowerDeviationRequest, PowerDeviationResponse]
