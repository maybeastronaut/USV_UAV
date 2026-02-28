from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, pi, sin, sqrt
from typing import Dict, List, Sequence, Tuple

from .models import AgentParams, AgentState, AgentType, Environment


def _wrap_to_pi(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


@dataclass
class WeightedVoronoiConfig:
    lambda_energy: float = 0.65
    lambda_current: float = 1.15
    lambda_priority: float = 1.05
    uav_speed_factor: float = 1.0
    usv_speed_factor: float = 0.78
    uav_priority_gain: float = 1.00
    usv_priority_gain: float = 0.88
    min_speed: float = 0.8
    uav_type_bias: float = 0.0
    usv_type_bias: float = 0.0


@dataclass
class VoronoiPartitionResult:
    owner_by_cell: List[str]
    cost_by_cell: List[float]
    cells_by_agent: Dict[str, List[Tuple[float, float]]]
    mean_cost_by_agent: Dict[str, float]
    count_by_agent: Dict[str, int]


def make_priority_map(
    grid_xy: Sequence[Tuple[float, float]],
    hotspots: Sequence[Tuple[float, float, float, float]],
    base: float = 0.20,
) -> List[float]:
    """
    Build [0,1] priority scores with Gaussian hotspots.

    hotspots item format: (cx, cy, sigma, amplitude)
    """
    out: List[float] = []
    for x, y in grid_xy:
        score = base
        for cx, cy, sigma, amp in hotspots:
            if sigma <= 1e-6:
                continue
            dx = x - cx
            dy = y - cy
            score += amp * exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        out.append(score)
    return out


def weighted_voronoi_partition(
    grid_xy: Sequence[Tuple[float, float]],
    states: Dict[str, AgentState],
    params_map: Dict[str, AgentParams],
    env: Environment,
    priority_map: Sequence[float] | None = None,
    cfg: WeightedVoronoiConfig | None = None,
    current_time: float = 0.0,
) -> VoronoiPartitionResult:
    if cfg is None:
        cfg = WeightedVoronoiConfig()

    agent_names = list(states.keys())
    if not agent_names:
        raise ValueError("No agents provided")

    if priority_map is None:
        priority_map = [0.5] * len(grid_xy)
    if len(priority_map) != len(grid_xy):
        raise ValueError("priority_map length must match grid size")

    max_energy = max(max(s.energy, 1.0) for s in states.values())
    owner_by_cell: List[str] = []
    cost_by_cell: List[float] = []

    count_by_agent = {name: 0 for name in agent_names}
    sum_cost_by_agent = {name: 0.0 for name in agent_names}
    cells_by_agent: Dict[str, List[Tuple[float, float]]] = {name: [] for name in agent_names}

    for idx, point in enumerate(grid_xy):
        px, py = point
        pscore = float(priority_map[idx])
        best_name = agent_names[0]
        best_cost = 1e18

        for name in agent_names:
            st = states[name]
            params = params_map[name]

            dx = px - st.x
            dy = py - st.y
            dist = sqrt(dx * dx + dy * dy)

            if params.agent_type == AgentType.UAV:
                speed_eff = max(cfg.min_speed, params.v_max * cfg.uav_speed_factor)
                priority_gain = cfg.uav_priority_gain
                current_term = 0.0
                type_bias = cfg.uav_type_bias
            else:
                speed_eff = max(cfg.min_speed, params.v_max * cfg.usv_speed_factor)
                priority_gain = cfg.usv_priority_gain
                type_bias = cfg.usv_type_bias
                cx, cy = env.current.velocity(st.x, st.y, current_time)
                if dist > 1e-9:
                    ux = dx / dist
                    uy = dy / dist
                else:
                    ux, uy = 0.0, 0.0
                opposing = max(0.0, -(cx * ux + cy * uy))
                cross = abs(cx * uy - cy * ux)
                current_term = (0.7 * opposing + 0.22 * cross) / max(params.v_max, cfg.min_speed)

            travel_term = dist / speed_eff
            energy_term = max_energy / max(st.energy, 1.0)
            priority_term = 1.0 - min(1.0, max(0.0, pscore * priority_gain))

            cost = travel_term + cfg.lambda_energy * energy_term + cfg.lambda_current * current_term
            cost += cfg.lambda_priority * priority_term
            cost += type_bias

            if cost < best_cost:
                best_cost = cost
                best_name = name

        owner_by_cell.append(best_name)
        cost_by_cell.append(best_cost)
        count_by_agent[best_name] += 1
        sum_cost_by_agent[best_name] += best_cost
        cells_by_agent[best_name].append(point)

    mean_cost_by_agent: Dict[str, float] = {}
    for name in agent_names:
        cnt = count_by_agent[name]
        mean_cost_by_agent[name] = (sum_cost_by_agent[name] / cnt) if cnt > 0 else 0.0

    return VoronoiPartitionResult(
        owner_by_cell=owner_by_cell,
        cost_by_cell=cost_by_cell,
        cells_by_agent=cells_by_agent,
        mean_cost_by_agent=mean_cost_by_agent,
        count_by_agent=count_by_agent,
    )


def _ordered_lawnmower_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    rows: Dict[float, List[Tuple[float, float]]] = {}
    for x, y in points:
        yk = round(y, 6)
        rows.setdefault(yk, []).append((x, y))

    ys = sorted(rows.keys())
    out: List[Tuple[float, float]] = []
    for i, yk in enumerate(ys):
        row = sorted(rows[yk], key=lambda p: p[0], reverse=(i % 2 == 1))
        out.extend(row)
    return out


def _rotate_to_nearest(
    route: Sequence[Tuple[float, float]],
    start: Tuple[float, float],
) -> List[Tuple[float, float]]:
    if not route:
        return []
    best_idx = 0
    best_dist = _euclid(route[0], start)
    for i in range(1, len(route)):
        d = _euclid(route[i], start)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return list(route[best_idx:]) + list(route[:best_idx])


def plan_uav_lawnmower(
    points: Sequence[Tuple[float, float]],
    start: Tuple[float, float],
) -> List[Tuple[float, float]]:
    ordered = _ordered_lawnmower_points(points)
    return _rotate_to_nearest(ordered, start)


def _downsample_points(points: Sequence[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    pts = list(points)
    n = len(pts)
    if n <= max_points:
        return pts
    stride = max(1, n // max_points)
    sampled = pts[::stride]
    if sampled and sampled[-1] != pts[-1]:
        sampled.append(pts[-1])
    return sampled


def plan_usv_turn_constrained(
    points: Sequence[Tuple[float, float]],
    start: Tuple[float, float],
    initial_heading: float,
    max_points: int = 80,
    turn_limit: float = 0.42,
    turn_weight: float = 8.8,
    bridge_step: float = 8.0,
) -> List[Tuple[float, float]]:
    if not points:
        return []

    ordered_seed = _ordered_lawnmower_points(points)
    candidates = _downsample_points(ordered_seed, max_points=max_points)

    remaining = candidates[:]
    route: List[Tuple[float, float]] = []
    current = start
    heading = initial_heading

    while remaining:
        best_idx = 0
        best_score = 1e18
        best_heading = heading

        for i, target in enumerate(remaining):
            dx = target[0] - current[0]
            dy = target[1] - current[1]
            dist = sqrt(dx * dx + dy * dy)
            if dist < 1e-9:
                score = 0.0
                target_heading = heading
            else:
                target_heading = atan2(dy, dx)
                dtheta = abs(_wrap_to_pi(target_heading - heading))
                score = dist + turn_weight * dtheta

            if score < best_score:
                best_score = score
                best_idx = i
                best_heading = target_heading

        target = remaining.pop(best_idx)
        dtheta = _wrap_to_pi(best_heading - heading)
        if abs(dtheta) > turn_limit:
            n_bridge = int(abs(dtheta) // turn_limit)
            sgn = 1.0 if dtheta > 0.0 else -1.0
            for _ in range(n_bridge):
                heading = _wrap_to_pi(heading + sgn * turn_limit)
                # Insert short bridge points to avoid unrealistically sharp turns.
                bx = current[0] + bridge_step * cos(heading)
                by = current[1] + bridge_step * sin(heading)
                bridge = (bx, by)
                route.append(bridge)
                current = bridge

        route.append(target)
        current = target
        heading = best_heading

    return route


def plan_heterogeneous_paths(
    cells_by_agent: Dict[str, List[Tuple[float, float]]],
    states: Dict[str, AgentState],
    params_map: Dict[str, AgentParams],
) -> Dict[str, List[Tuple[float, float]]]:
    paths: Dict[str, List[Tuple[float, float]]] = {}
    for name, points in cells_by_agent.items():
        st = states[name]
        start = (st.x, st.y)
        params = params_map[name]
        if params.agent_type == AgentType.UAV:
            path = plan_uav_lawnmower(points, start)
        else:
            path = plan_usv_turn_constrained(points, start, initial_heading=st.psi)
        paths[name] = path
    return paths
