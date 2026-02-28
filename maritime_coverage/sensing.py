from __future__ import annotations

from math import atan2, exp, pi, sqrt
from typing import List, Sequence, Tuple

from .models import AgentState, SensorParams


def _wrap_to_pi(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def uav_detection(
    points_xy: Sequence[Tuple[float, float]],
    state: AgentState,
    sensor: SensorParams,
) -> List[float]:
    detections: List[float] = []
    for x, y in points_xy:
        dx = x - state.x
        dy = y - state.y
        dist = sqrt(dx * dx + dy * dy)
        if dist > sensor.radius:
            detections.append(0.0)
            continue
        q = sensor.kappa * exp(-(dist * dist) / (2.0 * sensor.sigma_r * sensor.sigma_r))
        detections.append(_clamp01(q))
    return detections


def usv_detection(
    points_xy: Sequence[Tuple[float, float]],
    state: AgentState,
    sensor: SensorParams,
) -> List[float]:
    detections: List[float] = []
    for x, y in points_xy:
        dx = x - state.x
        dy = y - state.y
        dist = sqrt(dx * dx + dy * dy)
        if dist > sensor.radius:
            detections.append(0.0)
            continue
        bearing = atan2(dy, dx)
        dphi = _wrap_to_pi(bearing - state.psi)
        if abs(dphi) > sensor.fov / 2.0:
            detections.append(0.0)
            continue

        qr = exp(-(dist * dist) / (2.0 * sensor.sigma_r * sensor.sigma_r))
        qphi = exp(-(dphi * dphi) / (2.0 * sensor.sigma_phi * sensor.sigma_phi))
        q = sensor.kappa * qr * qphi
        detections.append(_clamp01(q))
    return detections


def fused_detection(detection_stack: Sequence[Sequence[float]]) -> List[float]:
    # 1 - product(1 - q_i) models independent detection fusion.
    if not detection_stack:
        return []

    num_cells = len(detection_stack[0])
    fused = [0.0] * num_cells
    for idx in range(num_cells):
        miss_prob = 1.0
        for detections in detection_stack:
            q = _clamp01(float(detections[idx]))
            miss_prob *= 1.0 - q
        fused[idx] = 1.0 - miss_prob
    return fused
