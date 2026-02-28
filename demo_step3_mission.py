from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, exp, pi, sin, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from maritime_coverage import (
    AgentParams,
    AgentState,
    AgentType,
    Control,
    CoverageProblem,
    Environment,
    OceanCurrent,
    ObstacleRect,
    SensorParams,
    SimulationConfig,
    WeightedVoronoiConfig,
    WindField,
    fused_detection,
    plan_heterogeneous_paths,
    save_mission_animation_html,
    save_partition_svg,
    save_snapshot_svg,
    step_agent,
    uav_detection,
    usv_detection,
    weighted_voronoi_partition,
)

DOMAIN_X = (0.0, 560.0)
DOMAIN_Y = (0.0, 420.0)
NUM_OBSTACLES = 4
NUM_REGIONS = 12
CURRENT_ALGORITHM = "hierarchical_v1"
SCENARIO_PROFILES = ("random", "corridor", "clustered", "split-corners")
TASK_BASE_SECONDS = {"low": 2.0, "mid": 4.0, "high": 8.0}
TASK_LEVEL_CN = {"low": "下", "mid": "中", "high": "上"}
TASK_LEVELS = ("low", "mid", "high")
UNKNOWN_DIFFICULTY = "unknown"
OBSTACLE_SAFE_MARGIN = 8.0
USV_OBSTACLE_LOOKAHEAD_SCALE = 1.35
USV_OBSTACLE_LOOKAHEAD_MIN = 20.0
USV_OBSTACLE_PROBE_MARGIN = OBSTACLE_SAFE_MARGIN + 4.0
USV_BOUNDARY_LOOKAHEAD_SCALE = 1.20
USV_BOUNDARY_LOOKAHEAD_MIN = 24.0
USV_BOUNDARY_PROBE_MARGIN = 8.0
USV_BOUNDARY_HARD_MARGIN = 2.0
USV_ROUTE_BOUNDARY_MARGIN = 10.0
LOAD_WORK_SECONDS_TO_EQ_M = 2.2
LOAD_TRAVEL_COMMIT_RATIO = 0.55
LOAD_BALANCE_OVERLOAD_WEIGHT = 1.35
LOAD_BALANCE_UNDERLOAD_BONUS = 0.22
LOAD_QUEUE_PENALTY = 42.0
ASSIGNMENT_OBSTACLE_PENALTY = 160.0
ASSIGNMENT_TRAVEL_SPEED_FACTOR = 0.82
ASSIGNMENT_OBS_TO_SECONDS = 0.10
ASSIGNMENT_QUEUE_SECONDS = 3.4
ASSIGNMENT_LOAD_SECONDS = 0.065
ASSIGNMENT_OWNER_BONUS_SECONDS = 5.5
ASSIGNMENT_STICKY_BONUS_SECONDS = 3.5
ASSIGNMENT_WAIT_AGE_GAIN = 0.28
ASSIGNMENT_WAIT_AGE_CAP_SECONDS = 22.0
ASSIGNMENT_NON_EDGE_TASK_MARGIN = 56.0
ASSIGNMENT_EDGE_PENALTY_TRIGGER = 48.0
ASSIGNMENT_EDGE_PENALTY_SECONDS_PER_M = 0.24
TASK_CAPTURE_RADIUS_RATIO = 1.35
TASK_LOCK_ANCHOR_RATIO = 0.62
TASK_QUEUE_RADIUS_RATIO = 1.28
TASK_QUEUE_LAYER_STEP = 0.55
LOCK_ANCHOR_HYSTERESIS_M = 1.6
IDLE_EDGE_MARGIN = 24.0
IDLE_WAYPOINT_MIN_GAP = 14.0
IDLE_NAV_WAYPOINT_TOL = 12.0
USV_DEADLOCK_RELAX_TRIGGER = 0.85
USV_DEADLOCK_RELAX_STEP = 1.2
USV_DEADLOCK_MIN_SAFE = 1.6
USV_PROX_TRIGGER_RATIO = 1.75
USV_PROX_MIN_SEP_RATIO = 1.10
USV_DETOUR_LATERAL_RATIO = 0.78
USV_DETOUR_FORWARD_RATIO = 0.95
USV_DETOUR_COOLDOWN_STEPS = 10
USV_DETOUR_MIN_GAP = 6.0
USV_IDLE_EXTRA_WAYPOINT_TOL = 14.0
USV_IDLE_STUCK_WINDOW_STEPS = 20
USV_IDLE_STUCK_PROGRESS_MIN = 5.0
USV_IDLE_STUCK_TARGET_RADIUS = 26.0
USV_OMEGA_RATE_LIMIT = 0.22
USV_OMEGA_DEADBAND = 0.03
USV_AVOID_SIGN_DPSI_DEADBAND = 0.10
USV_AVOID_SIGN_KEEP_EDGE_MARGIN = 2.6
USV_TARGET_SIDE_HEADING_MAX = 1.05
USV_TARGET_SIDE_EDGE_BUFFER = 5.0
USV_TARGET_SIDE_OVERRIDE_DPSI_MAX = 0.70
USV_TARGET_SIDE_NEAR_DIST_RATIO = 3.2
USV_TARGET_SIDE_NEAR_OVERRIDE_DPSI_MAX = 1.40
USV_OMEGA_CHANGE_PENALTY = 0.26
USV_TURN_FLIP_PENALTY = 0.46
USV_TURN_INTENT_OMEGA_MIN = 0.15
USV_TURN_INTENT_OMEGA_RESET = 0.05
SUPPORT_UNCOVERED_RATIO_TOL = 0.01
SUPPORT_UNCOVERED_MIN_CELLS_TOL = 2
OWN_REGION_EXPLORE_THRESHOLD = 0.90
OBSTACLE_CENTER_BIAS_PROB = 0.88
OBSTACLE_CENTER_SIGMA_RATIO = 0.17
OBSTACLE_CENTER_MIN_NORM = 0.14
OBSTACLE_CENTER_MAX_NORM = 0.70
OBSTACLE_CENTER_INNER_KEEP_PROB = 0.08
OBSTACLE_CENTER_OUTLIER_KEEP_PROB = 0.08
OBSTACLE_CENTER_SEP_MIN = 104.0
UAV_DISPLAY_COLOR = "#f97316"
USV_DISPLAY_PALETTE = (
    "#2563eb",
    "#0f766e",
    "#dc2626",
    "#ca8a04",
    "#9333ea",
    "#be123c",
    "#0369a1",
    "#ea580c",
)


@dataclass
class DetectionRegion:
    x: float
    y: float
    radius: float
    severity: float
    label: str
    difficulty: str


@dataclass
class AgentRunner:
    name: str
    params: AgentParams
    state: AgentState
    waypoints: List[Tuple[float, float]]
    wp_index: int = 0

    def current_target(self) -> Tuple[float, float]:
        return self.waypoints[self.wp_index]

    def maybe_advance_waypoint(self, tol: float = 8.0) -> None:
        if not self.waypoints:
            return
        tx, ty = self.current_target()
        if sqrt((self.state.x - tx) ** 2 + (self.state.y - ty) ** 2) < tol:
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)


@dataclass
class MissionTargets:
    coverage_goal: float
    handling_goal: float


@dataclass
class AlgorithmProfile:
    name: str
    replan_interval_steps: int
    max_execute_steps: int
    detection_threshold: float
    usv_heading_gain: float
    usv_speed_gain: float
    usv_turn_cap_ratio: float
    usv_waypoint_tol: float
    route_spacing: float
    urgent_insert_interval: int
    voronoi_cfg: WeightedVoronoiConfig


@dataclass
class MissionSummary:
    mission_seed: int
    profile_name: str
    scenario_profile: str
    sea_width: float
    sea_height: float
    num_obstacles: int
    num_regions: int
    grid_cells: int
    stage1_mean_scan_quality: float
    rolling_map_quality: float
    latest_detected_regions: int
    replan_count: int
    executed_steps: int
    latest_partition_counts: Dict[str, int]
    assign_sum: Dict[str, int]
    latest_priority_counts: Dict[str, int]
    latest_path_lengths: Dict[str, int]
    coverage_rate: float
    usv_only_coverage_rate: float
    mean_usv_coverage: float
    all_problem_rate: float
    handled_all: int
    detected_problem_rate: float
    handled_detected: int
    turn_index_global: float
    turn_sum: float
    turn_events: int
    dist_sum: float
    usv_travel_m: Dict[str, float]
    usv_work_s: Dict[str, float]
    usv_load_eq_m: Dict[str, float]
    load_balance_cv: float
    mission_completion_time: float
    avg_response_time: float
    avg_processing_completion_time: float
    known_difficulty_count: int
    log_path: str
    meets_targets: bool


def wrap_to_pi(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def uav_holonomic_control(state: AgentState, target: Tuple[float, float], v_max: float) -> Control:
    dx = target[0] - state.x
    dy = target[1] - state.y
    gain = 0.52
    vx = gain * dx
    vy = gain * dy
    speed = sqrt(vx * vx + vy * vy)
    if speed > v_max and speed > 1e-9:
        s = v_max / speed
        vx *= s
        vy *= s
    return Control(a=vx, omega=vy)


def usv_control(
    state: AgentState,
    target: Tuple[float, float],
    params: AgentParams,
    heading_gain: float,
    speed_gain: float,
    turn_cap_ratio: float,
    allow_stop_near_target: bool = False,
) -> Control:
    tx, ty = target
    dx = tx - state.x
    dy = ty - state.y
    dist = sqrt(dx * dx + dy * dy)
    heading_target = atan2(dy, dx)
    heading_error = wrap_to_pi(heading_target - state.psi)

    omega_cap = max(0.1, params.omega_max * turn_cap_ratio)
    omega = max(-omega_cap, min(heading_gain * heading_error, omega_cap))
    v_floor = params.v_min
    if allow_stop_near_target:
        if dist < 18.0:
            v_floor = 0.0
        elif dist < 28.0:
            v_floor = min(v_floor, 1.0)
    desired_speed = max(v_floor, min(speed_gain * dist, params.v_max))
    a = max(-params.a_max, min(desired_speed - state.v, params.a_max))
    return Control(a=a, omega=omega)


def _rate_limit(value: float, prev: float, max_delta: float) -> float:
    return max(prev - max_delta, min(value, prev + max_delta))


def _update_turn_intent_sign(prev_sign: float, omega: float) -> float:
    if omega >= USV_TURN_INTENT_OMEGA_MIN:
        return 1.0
    if omega <= -USV_TURN_INTENT_OMEGA_MIN:
        return -1.0
    if abs(omega) <= USV_TURN_INTENT_OMEGA_RESET:
        return 0.0
    return prev_sign


def make_sweep_waypoints(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    lane_step: float,
) -> List[Tuple[float, float]]:
    y = y_min + lane_step * 0.5
    lanes: List[float] = []
    while y <= y_max - lane_step * 0.3:
        lanes.append(y)
        y += lane_step

    points: List[Tuple[float, float]] = []
    for i, yy in enumerate(lanes):
        if i % 2 == 0:
            points.append((x_min, yy))
            points.append((x_max, yy))
        else:
            points.append((x_max, yy))
            points.append((x_min, yy))
    return points


def _rects_overlap(a: ObstacleRect, b: ObstacleRect, pad: float = 0.0) -> bool:
    return not (
        a.xmax + pad < b.xmin
        or b.xmax + pad < a.xmin
        or a.ymax + pad < b.ymin
        or b.ymax + pad < a.ymin
    )


def _point_near_rect(x: float, y: float, rect: ObstacleRect, margin: float) -> bool:
    return (
        rect.xmin - margin <= x <= rect.xmax + margin
        and rect.ymin - margin <= y <= rect.ymax + margin
    )


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _point_to_segment_dist2(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    wx = px - ax
    wy = py - ay
    t = (wx * vx + wy * vy) / vv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * vx
    cy = ay + t * vy
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def _edge_clearance(x: float, y: float, env: Environment) -> float:
    return min(x - env.xlim[0], env.xlim[1] - x, y - env.ylim[0], env.ylim[1] - y)


def _point_too_close_to_boundary(x: float, y: float, env: Environment, margin: float) -> bool:
    return _edge_clearance(x, y, env) < margin


def _sanitize_waypoints_for_env(
    points: Sequence[Tuple[float, float]],
    env: Environment,
    boundary_margin: float,
    min_gap: float = IDLE_WAYPOINT_MIN_GAP,
) -> List[Tuple[float, float]]:
    if not points:
        return []
    xmin, xmax = env.xlim
    ymin, ymax = env.ylim
    out: List[Tuple[float, float]] = []
    min_gap2 = max(1e-6, min_gap * min_gap)
    for x, y in points:
        px = min(max(x, xmin + boundary_margin), xmax - boundary_margin)
        py = min(max(y, ymin + boundary_margin), ymax - boundary_margin)
        if env.in_obstacle(px, py):
            continue
        if _point_too_close_to_obstacle(px, py, env, margin=OBSTACLE_SAFE_MARGIN + 2.0):
            continue
        p = (px, py)
        if all(_dist2(p, q) >= min_gap2 for q in out):
            out.append(p)
    return out


def _coarsen_route(route: Sequence[Tuple[float, float]], min_spacing: float) -> List[Tuple[float, float]]:
    if not route:
        return []
    if min_spacing <= 1e-6:
        return list(route)

    out: List[Tuple[float, float]] = [route[0]]
    min_spacing2 = min_spacing * min_spacing
    for p in route[1:]:
        if _dist2(out[-1], p) >= min_spacing2:
            out.append(p)

    if out[-1] != route[-1]:
        out.append(route[-1])
    return out


def _fallback_obstacles() -> List[ObstacleRect]:
    return [
        ObstacleRect(188.0, 242.0, 122.0, 176.0),
        ObstacleRect(316.0, 372.0, 126.0, 182.0),
        ObstacleRect(220.0, 276.0, 244.0, 300.0),
        ObstacleRect(298.0, 354.0, 248.0, 306.0),
    ]


def _severity_for_difficulty(rng: random.Random, difficulty: str) -> float:
    if difficulty == "low":
        return rng.uniform(0.76, 0.86)
    if difficulty == "mid":
        return rng.uniform(0.84, 0.94)
    return rng.uniform(0.92, 1.00)


def _difficulty_priority(difficulty: str | None) -> float:
    if difficulty == "low":
        return 1.0
    if difficulty == "mid":
        return 1.35
    if difficulty == "high":
        return 1.75
    return 1.2


def _required_agents_for_task(difficulty: str | None) -> int:
    if difficulty == "high":
        return 3
    if difficulty == "mid":
        return 2
    if difficulty == "low":
        return 1
    return 2


def _expected_task_seconds(difficulty: str | None) -> float:
    if difficulty in TASK_BASE_SECONDS:
        return float(TASK_BASE_SECONDS[difficulty])
    # Unknown tasks are estimated as the average of low/mid/high.
    return sum(TASK_BASE_SECONDS.values()) / float(len(TASK_BASE_SECONDS))


def _expected_agent_work_seconds(difficulty: str | None, n_agents: int) -> float:
    n = max(1, int(n_agents))
    total = _expected_task_seconds(difficulty)
    coop_factor = 1.0 + 0.80 * (n - 1)
    return total / max(1e-6, n * coop_factor)


def _usv_load_equivalent(travel_m: float, work_s: float) -> float:
    return float(travel_m) + LOAD_WORK_SECONDS_TO_EQ_M * float(work_s)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean <= 1e-9:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return sqrt(var) / mean


def _segment_obstacle_penalty(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    env: Environment,
    margin: float = OBSTACLE_SAFE_MARGIN + 2.0,
) -> float:
    if not env.obstacles:
        return 0.0
    x0, y0 = p0
    x1, y1 = p1
    seg_len = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    samples = max(8, int(seg_len / 16.0))
    hit = False
    near_hits = 0
    for k in range(1, samples + 1):
        a = k / float(samples)
        x = x0 + (x1 - x0) * a
        y = y0 + (y1 - y0) * a
        if env.in_obstacle(x, y):
            hit = True
            break
        if _point_too_close_to_obstacle(x, y, env, margin=margin):
            near_hits += 1
    if hit:
        return ASSIGNMENT_OBSTACLE_PENALTY
    if near_hits == 0:
        return 0.0
    return ASSIGNMENT_OBSTACLE_PENALTY * min(0.75, near_hits / max(1.0, 0.4 * samples))


def _required_agents_for_dispatch(
    difficulty: str | None,
    remaining_work_s: float,
) -> int:
    if difficulty == "high":
        return 2 if remaining_work_s > 1.6 else 1
    if difficulty == "mid":
        return 1
    if difficulty == "low":
        return 1
    # Unknown: first response should be broad exploration, avoid over-allocation.
    return 1


def _max_safe_collab_for_region(radius: float, safe_distance: float) -> int:
    sd = max(1e-6, float(safe_distance))
    rr = float(radius) / sd
    if rr < 1.35:
        return 1
    if rr < 2.10:
        return 2
    return 3


def _fallback_regions(rng: random.Random) -> List[DetectionRegion]:
    base_points = [
        (68.0, 76.0),
        (156.0, 332.0),
        (248.0, 256.0),
        (346.0, 84.0),
        (432.0, 314.0),
        (520.0, 138.0),
        (118.0, 216.0),
        (212.0, 116.0),
        (304.0, 360.0),
        (392.0, 212.0),
        (474.0, 78.0),
        (540.0, 274.0),
    ]
    out: List[DetectionRegion] = []
    for i, (x, y) in enumerate(base_points[:NUM_REGIONS], start=1):
        difficulty = rng.choice(TASK_LEVELS)
        out.append(
            DetectionRegion(
                x=x,
                y=y,
                radius=rng.uniform(10.0, 16.0),
                severity=_severity_for_difficulty(rng, difficulty),
                label=f"Target-{i}",
                difficulty=difficulty,
            )
        )
    return out


def _sample_random_obstacles(
    rng: random.Random,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    clear_points: Sequence[Tuple[float, float]],
) -> List[ObstacleRect]:
    cx_mid = 0.5 * (xlim[0] + xlim[1])
    cy_mid = 0.5 * (ylim[0] + ylim[1])
    sx = OBSTACLE_CENTER_SIGMA_RATIO * (xlim[1] - xlim[0])
    sy = OBSTACLE_CENTER_SIGMA_RATIO * (ylim[1] - ylim[0])

    obstacles: List[ObstacleRect] = []
    attempts = 0
    while len(obstacles) < NUM_OBSTACLES and attempts < 6000:
        attempts += 1
        w = rng.uniform(48.0, 86.0)
        h = rng.uniform(34.0, 68.0)

        x_lo = xlim[0] + w * 0.5 + 20.0
        x_hi = xlim[1] - w * 0.5 - 20.0
        y_lo = ylim[0] + h * 0.5 + 20.0
        y_hi = ylim[1] - h * 0.5 - 20.0

        if rng.random() < OBSTACLE_CENTER_BIAS_PROB:
            cx = min(max(rng.gauss(cx_mid, sx), x_lo), x_hi)
            cy = min(max(rng.gauss(cy_mid, sy), y_lo), y_hi)
        else:
            cx = rng.uniform(x_lo, x_hi)
            cy = rng.uniform(y_lo, y_hi)

        x_span = max(1.0, 0.5 * (x_hi - x_lo))
        y_span = max(1.0, 0.5 * (y_hi - y_lo))
        center_norm = ((cx - cx_mid) / x_span) ** 2 + ((cy - cy_mid) / y_span) ** 2
        if center_norm < OBSTACLE_CENTER_MIN_NORM and rng.random() > OBSTACLE_CENTER_INNER_KEEP_PROB:
            continue
        if center_norm > OBSTACLE_CENTER_MAX_NORM and rng.random() > OBSTACLE_CENTER_OUTLIER_KEEP_PROB:
            continue

        if obstacles:
            sep_min2 = OBSTACLE_CENTER_SEP_MIN * OBSTACLE_CENTER_SEP_MIN
            if any(
                ((0.5 * (obs.xmin + obs.xmax) - cx) ** 2 + (0.5 * (obs.ymin + obs.ymax) - cy) ** 2) < sep_min2
                for obs in obstacles
            ):
                continue
        cand = ObstacleRect(cx - w * 0.5, cx + w * 0.5, cy - h * 0.5, cy + h * 0.5)

        if any(_rects_overlap(cand, obs, pad=18.0) for obs in obstacles):
            continue
        if any(_point_near_rect(px, py, cand, margin=18.0) for px, py in clear_points):
            continue

        obstacles.append(cand)

    if len(obstacles) < NUM_OBSTACLES:
        return _fallback_obstacles()
    return obstacles


def _sample_random_regions(
    rng: random.Random,
    env: Environment,
    obstacles: Sequence[ObstacleRect],
) -> List[DetectionRegion]:
    labels = [f"Target-{i+1}" for i in range(NUM_REGIONS)]
    regions: List[DetectionRegion] = []

    attempts = 0
    while len(regions) < NUM_REGIONS and attempts < 12000:
        attempts += 1
        radius = rng.uniform(9.0, 16.0)
        x = rng.uniform(env.xlim[0] + radius + 16.0, env.xlim[1] - radius - 16.0)
        y = rng.uniform(env.ylim[0] + radius + 16.0, env.ylim[1] - radius - 16.0)

        if any(_point_near_rect(x, y, obs, margin=radius + 6.0) for obs in obstacles):
            continue

        ok = True
        for rg in regions:
            min_gap = radius + rg.radius + 14.0
            if (x - rg.x) ** 2 + (y - rg.y) ** 2 < min_gap * min_gap:
                ok = False
                break
        if not ok:
            continue

        difficulty = rng.choice(TASK_LEVELS)
        severity = _severity_for_difficulty(rng, difficulty)
        regions.append(
            DetectionRegion(
                x=x,
                y=y,
                radius=radius,
                severity=severity,
                label=labels[len(regions)],
                difficulty=difficulty,
            )
        )

    if len(regions) < NUM_REGIONS:
        return _fallback_regions(rng)
    return regions


def _sample_anchor_regions(
    rng: random.Random,
    env: Environment,
    obstacles: Sequence[ObstacleRect],
    anchors: Sequence[Tuple[float, float, float]],
    radius_min: float = 7.0,
    radius_max: float = 12.8,
    min_gap_extra: float = 7.0,
) -> List[DetectionRegion]:
    labels = [f"Target-{i+1}" for i in range(NUM_REGIONS)]
    regions: List[DetectionRegion] = []

    def can_place(x: float, y: float, radius: float) -> bool:
        if any(_point_near_rect(x, y, obs, margin=radius + 6.0) for obs in obstacles):
            return False
        for rg in regions:
            min_gap = radius + rg.radius + min_gap_extra
            if (x - rg.x) ** 2 + (y - rg.y) ** 2 < min_gap * min_gap:
                return False
        return True

    for cx, cy, jitter in anchors[:NUM_REGIONS]:
        placed = False
        for _ in range(140):
            radius = rng.uniform(radius_min, radius_max)
            x = rng.gauss(cx, jitter)
            y = rng.gauss(cy, jitter)
            x = min(max(x, env.xlim[0] + radius + 12.0), env.xlim[1] - radius - 12.0)
            y = min(max(y, env.ylim[0] + radius + 12.0), env.ylim[1] - radius - 12.0)
            if not can_place(x, y, radius):
                continue
            difficulty = rng.choice(TASK_LEVELS)
            regions.append(
                DetectionRegion(
                    x=x,
                    y=y,
                    radius=radius,
                    severity=_severity_for_difficulty(rng, difficulty),
                    label=labels[len(regions)],
                    difficulty=difficulty,
                )
            )
            placed = True
            break
        if not placed and len(regions) >= NUM_REGIONS:
            break

    attempts = 0
    while len(regions) < NUM_REGIONS and attempts < 15000:
        attempts += 1
        radius = rng.uniform(radius_min, radius_max)
        x = rng.uniform(env.xlim[0] + radius + 12.0, env.xlim[1] - radius - 12.0)
        y = rng.uniform(env.ylim[0] + radius + 12.0, env.ylim[1] - radius - 12.0)
        if not can_place(x, y, radius):
            continue
        difficulty = rng.choice(TASK_LEVELS)
        regions.append(
            DetectionRegion(
                x=x,
                y=y,
                radius=radius,
                severity=_severity_for_difficulty(rng, difficulty),
                label=labels[len(regions)],
                difficulty=difficulty,
            )
        )

    if len(regions) < NUM_REGIONS:
        return _fallback_regions(rng)
    return regions


def _sample_corridor_obstacles(
    rng: random.Random,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> List[ObstacleRect]:
    xmin, xmax = xlim
    ymin, ymax = ylim
    pad = 16.0
    x1 = rng.uniform(xmin + 170.0, xmin + 196.0)
    x2 = rng.uniform(xmin + 350.0, xmin + 382.0)
    w1 = rng.uniform(24.0, 30.0)
    w2 = rng.uniform(24.0, 32.0)
    g1 = rng.uniform(ymin + 112.0, ymin + 148.0)
    g2 = rng.uniform(ymin + 272.0, ymin + 306.0)
    gh1 = rng.uniform(30.0, 38.0)
    gh2 = rng.uniform(30.0, 38.0)
    return [
        ObstacleRect(x1 - w1 * 0.5, x1 + w1 * 0.5, ymin + pad, g1 - gh1),
        ObstacleRect(x1 - w1 * 0.5, x1 + w1 * 0.5, g1 + gh1, ymax - pad),
        ObstacleRect(x2 - w2 * 0.5, x2 + w2 * 0.5, ymin + pad, g2 - gh2),
        ObstacleRect(x2 - w2 * 0.5, x2 + w2 * 0.5, g2 + gh2, ymax - pad),
    ]


def _sample_clustered_obstacles(
    rng: random.Random,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> List[ObstacleRect]:
    del xlim, ylim
    j = lambda b: b + rng.uniform(-10.0, 10.0)
    return [
        ObstacleRect(j(198.0), j(262.0), j(144.0), j(274.0)),
        ObstacleRect(j(302.0), j(366.0), j(150.0), j(280.0)),
        ObstacleRect(j(238.0), j(322.0), j(76.0), j(122.0)),
        ObstacleRect(j(244.0), j(328.0), j(300.0), j(346.0)),
    ]


def _sample_split_corner_obstacles(
    rng: random.Random,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> List[ObstacleRect]:
    del xlim, ylim
    j = lambda b: b + rng.uniform(-8.0, 8.0)
    return [
        ObstacleRect(j(258.0), j(302.0), j(94.0), j(326.0)),
        ObstacleRect(j(124.0), j(436.0), j(188.0), j(232.0)),
        ObstacleRect(j(152.0), j(214.0), j(82.0), j(144.0)),
        ObstacleRect(j(348.0), j(412.0), j(276.0), j(338.0)),
    ]


def _sample_regions_for_profile(
    rng: random.Random,
    env: Environment,
    obstacles: Sequence[ObstacleRect],
    scenario_profile: str,
) -> List[DetectionRegion]:
    if scenario_profile == "corridor":
        anchors = (
            [(170.0, 130.0, 16.0)] * 4
            + [(374.0, 286.0, 16.0)] * 4
            + [
                (86.0, 336.0, 14.0),
                (502.0, 86.0, 14.0),
                (110.0, 84.0, 14.0),
                (500.0, 338.0, 14.0),
            ]
        )
        return _sample_anchor_regions(rng, env, obstacles, anchors=anchors)

    if scenario_profile == "clustered":
        c1 = (rng.uniform(126.0, 206.0), rng.uniform(104.0, 188.0))
        c2 = (rng.uniform(352.0, 440.0), rng.uniform(234.0, 318.0))
        far_pool = [
            (74.0, 72.0),
            (74.0, 350.0),
            (502.0, 70.0),
            (502.0, 350.0),
            (282.0, 58.0),
            (282.0, 362.0),
            (526.0, 206.0),
        ]
        far_points = rng.sample(far_pool, 5)
        anchors = (
            [(c1[0], c1[1], 22.0)] * 4
            + [(c2[0], c2[1], 22.0)] * 3
            + [(x, y, 14.0) for x, y in far_points]
        )
        return _sample_anchor_regions(rng, env, obstacles, anchors=anchors)

    if scenario_profile == "split-corners":
        anchors = (
            [(78.0, 74.0, 22.0)] * 3
            + [(78.0, 346.0, 22.0)] * 3
            + [(486.0, 74.0, 22.0)] * 3
            + [(486.0, 346.0, 22.0)] * 3
        )
        return _sample_anchor_regions(rng, env, obstacles, anchors=anchors)

    return _sample_random_regions(rng, env=env, obstacles=obstacles)


def build_large_mission(seed: int, scenario_profile: str = "random"):
    rng = random.Random(seed)

    start_points = [
        (20.0, 20.0),
        (540.0, 20.0),
        (28.0, 200.0),
        (34.0, 34.0),
        (34.0, 360.0),
    ]
    if scenario_profile == "corridor":
        obstacles = _sample_corridor_obstacles(rng, DOMAIN_X, DOMAIN_Y)
    elif scenario_profile == "clustered":
        obstacles = _sample_clustered_obstacles(rng, DOMAIN_X, DOMAIN_Y)
    elif scenario_profile == "split-corners":
        obstacles = _sample_split_corner_obstacles(rng, DOMAIN_X, DOMAIN_Y)
    else:
        obstacles = _sample_random_obstacles(rng, DOMAIN_X, DOMAIN_Y, clear_points=start_points)

    env = Environment(
        xlim=DOMAIN_X,
        ylim=DOMAIN_Y,
        obstacles=obstacles,
        current=OceanCurrent(
            c0x=rng.uniform(0.10, 0.26),
            c0y=rng.uniform(-0.08, 0.10),
            gx=0.0,
            gy=0.0,
            gyre1_cx=rng.uniform(120.0, 220.0),
            gyre1_cy=rng.uniform(95.0, 320.0),
            gyre1_strength=rng.uniform(0.08, 0.20),
            gyre1_sigma=rng.uniform(70.0, 120.0),
            gyre1_clockwise=bool(rng.randint(0, 1)),
            gyre2_cx=rng.uniform(330.0, 470.0),
            gyre2_cy=rng.uniform(90.0, 330.0),
            gyre2_strength=rng.uniform(0.08, 0.20),
            gyre2_sigma=rng.uniform(75.0, 125.0),
            gyre2_clockwise=bool(rng.randint(0, 1)),
            tide_amp_u=rng.uniform(0.03, 0.10),
            tide_amp_v=rng.uniform(0.02, 0.08),
            tide_period=rng.uniform(130.0, 260.0),
            tide_phase=rng.uniform(0.0, 2.0 * pi),
            boundary_shear=rng.uniform(0.02, 0.08),
        ),
        wind=WindField(
            wx=rng.uniform(0.38, 0.68),
            wy=rng.uniform(0.08, 0.22),
        ),
    )

    sim = SimulationConfig(
        dt=0.6,
        horizon=560.0,
        nx=112,
        ny=84,
        safe_distance=10.0,
        coverage_threshold=0.72,
        revisit_max_gap=110.0,
        seen_prob_threshold=0.28,
        control_penalty=0.018,
    )

    uav_sensor = SensorParams(radius=58.0, kappa=0.96, sigma_r=24.0, fov=2.0 * pi)
    usv_sensor = SensorParams(radius=24.0, kappa=0.88, sigma_r=12.0, fov=pi / 2.0, sigma_phi=0.52)

    uav_params = AgentParams(
        agent_type=AgentType.UAV,
        v_min=0.0,
        v_max=23.0,
        omega_max=99.0,
        a_max=23.0,
        energy_min=4000.0,
        base_power=8.8,
        kv=0.075,
        kw=0.0,
        sensor=uav_sensor,
    )
    usv_params = AgentParams(
        agent_type=AgentType.USV,
        v_min=1.8,
        v_max=7.8,
        omega_max=0.52,
        a_max=1.1,
        energy_min=1800.0,
        base_power=4.6,
        kv=0.23,
        kw=2.1,
        sensor=usv_sensor,
    )

    uav_1_waypoints = make_sweep_waypoints(18.0, 278.0, 12.0, 408.0, lane_step=26.0)
    uav_2_waypoints = make_sweep_waypoints(282.0, 542.0, 12.0, 408.0, lane_step=26.0)

    uavs = [
        AgentRunner(
            name="uav_1",
            params=uav_params,
            state=AgentState(x=20.0, y=20.0, psi=0.0, v=12.0, energy=22000.0),
            waypoints=uav_1_waypoints,
        ),
        AgentRunner(
            name="uav_2",
            params=uav_params,
            state=AgentState(x=540.0, y=20.0, psi=pi, v=12.0, energy=22000.0),
            waypoints=uav_2_waypoints,
        ),
    ]

    usvs = [
        AgentRunner(
            name="usv_1",
            params=usv_params,
            state=AgentState(x=28.0, y=200.0, psi=-0.6, v=3.0, energy=9000.0),
            waypoints=[],
        ),
        AgentRunner(
            name="usv_2",
            params=usv_params,
            state=AgentState(x=34.0, y=34.0, psi=0.1, v=3.1, energy=9000.0),
            waypoints=[],
        ),
        AgentRunner(
            name="usv_3",
            params=usv_params,
            state=AgentState(x=34.0, y=360.0, psi=0.2, v=3.2, energy=9000.0),
            waypoints=[],
        ),
    ]

    detection_regions = _sample_regions_for_profile(
        rng=rng,
        env=env,
        obstacles=obstacles,
        scenario_profile=scenario_profile,
    )
    return env, sim, uavs, usvs, detection_regions


def risk_map_from_regions(
    grid_xy: Sequence[Tuple[float, float]],
    regions: Sequence[DetectionRegion],
) -> List[float]:
    values: List[float] = []
    for x, y in grid_xy:
        score = 0.0
        for rg in regions:
            sigma = max(6.0, 0.62 * rg.radius)
            dx = x - rg.x
            dy = y - rg.y
            score += rg.severity * exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        if score > 1.0:
            score = 1.0
        values.append(score)
    return values


def nearest_cell_index(grid_xy: Sequence[Tuple[float, float]], p: Tuple[float, float]) -> int:
    px, py = p
    best_i = 0
    best_d = 1e30
    for i, (x, y) in enumerate(grid_xy):
        dx = x - px
        dy = y - py
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def prepend_priority_stops(
    start: Tuple[float, float],
    route: Sequence[Tuple[float, float]],
    stops: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not stops:
        merged = list(route)
        if not merged:
            return [start]
        out: List[Tuple[float, float]] = []
        for p in merged:
            if not out or _dist2(p, out[-1]) >= IDLE_WAYPOINT_MIN_GAP * IDLE_WAYPOINT_MIN_GAP:
                out.append(p)
        return out or [start]

    remaining = list(stops)
    ordered: List[Tuple[float, float]] = []
    current = start
    while remaining:
        best_i = 0
        best_d = 1e30
        for i, p in enumerate(remaining):
            d = _dist2(p, current)
            if d < best_d:
                best_d = d
                best_i = i
        nxt = remaining.pop(best_i)
        ordered.append(nxt)
        current = nxt
    merged = ordered + list(route)
    out: List[Tuple[float, float]] = []
    for p in merged:
        if not out or _dist2(p, out[-1]) >= IDLE_WAYPOINT_MIN_GAP * IDLE_WAYPOINT_MIN_GAP:
            out.append(p)
    return out or [start]


def compute_turning_metrics_from_states(states: Sequence[AgentState]) -> Tuple[float, float, float, int]:
    total_dist = 0.0
    total_turn = 0.0
    turn_events = 0
    for i in range(1, len(states)):
        prev = states[i - 1]
        cur = states[i]
        dx = cur.x - prev.x
        dy = cur.y - prev.y
        dist = sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            continue
        total_dist += dist
        dtheta = abs(wrap_to_pi(cur.psi - prev.psi))
        total_turn += dtheta
        if dtheta > 0.04:  # around 2.3 deg in one control step
            turn_events += 1
    turning_index = total_turn / max(1e-6, total_dist)
    return total_turn, total_dist, turning_index, turn_events


def make_priority_and_detection(
    grid_xy: Sequence[Tuple[float, float]],
    global_coverage: Sequence[float],
    latent_risk: Sequence[float],
    regions: Sequence[DetectionRegion],
    detection_threshold: float,
) -> Tuple[List[float], List[float], List[DetectionRegion]]:
    detected_risk = [
        min(1.0, latent_risk[i] * (0.28 + 0.72 * global_coverage[i])) for i in range(len(global_coverage))
    ]
    priority_map = [
        min(1.0, 0.20 + 0.34 * global_coverage[i] + 0.80 * detected_risk[i]) for i in range(len(global_coverage))
    ]
    detected_regions: List[DetectionRegion] = []
    for rg in regions:
        idx = nearest_cell_index(grid_xy, (rg.x, rg.y))
        if detected_risk[idx] >= detection_threshold:
            detected_regions.append(rg)
    return detected_risk, priority_map, detected_regions


def make_algorithm_profile(name: str = CURRENT_ALGORITHM) -> AlgorithmProfile:
    if name != CURRENT_ALGORITHM:
        raise ValueError(f"Unsupported algorithm '{name}'. Available: {CURRENT_ALGORITHM}")

    return AlgorithmProfile(
        name=CURRENT_ALGORITHM,
        replan_interval_steps=60,
        max_execute_steps=2200,
        detection_threshold=0.26,
        usv_heading_gain=1.15,
        usv_speed_gain=0.38,
        usv_turn_cap_ratio=1.0,
        usv_waypoint_tol=8.0,
        route_spacing=10.0,
        urgent_insert_interval=18,
        voronoi_cfg=WeightedVoronoiConfig(
            lambda_energy=0.52,
            lambda_current=0.95,
            lambda_priority=1.28,
            usv_speed_factor=0.86,
            usv_priority_gain=1.14,
        ),
    )


def resolve_seed(args: argparse.Namespace) -> int:
    if args.seed_mode == "fixed":
        return int(args.seed)
    return random.SystemRandom().randint(1, 2_000_000_000)


def _inject_missing_region_waypoints(
    usv_by_name: Dict[str, AgentRunner],
    regions: Sequence[DetectionRegion],
    done_time_by_region: Dict[str, float | None],
    known_difficulty_by_task: Dict[str, str | None],
    eligible_labels: set[str] | None = None,
    assignment_by_task: Dict[str, List[str]] | None = None,
) -> None:
    pending = [
        rg
        for rg in regions
        if done_time_by_region.get(rg.label) is None
        and (eligible_labels is None or rg.label in eligible_labels)
    ]
    if not pending:
        return

    for rg in pending:
        selected: List[AgentRunner] = []
        if assignment_by_task is not None:
            assignees = assignment_by_task.get(rg.label, [])
            selected = [usv_by_name[name] for name in assignees if name in usv_by_name]
        if not selected:
            need_n = min(len(usv_by_name), _required_agents_for_task(known_difficulty_by_task.get(rg.label)))
            ranked = sorted(
                usv_by_name.values(),
                key=lambda ag: (ag.state.x - rg.x) ** 2 + (ag.state.y - rg.y) ** 2,
            )
            selected = ranked[:need_n]

        for best in selected:
            if best.waypoints:
                current = best.current_target()
                if _dist2(current, (rg.x, rg.y)) < 14.0 * 14.0:
                    continue
                remaining = best.waypoints[best.wp_index :] + best.waypoints[: best.wp_index]
                remaining = [p for p in remaining if _dist2(p, (rg.x, rg.y)) > 10.0 * 10.0]
            else:
                remaining = []

            # Keep current task continuity: when USV is already heading to/processing a task,
            # queue new discovered tasks behind the current target instead of hard preemption.
            insert_front = True
            if best.waypoints:
                cur_rg = _pending_region_for_waypoint(best.current_target(), regions, done_time_by_region)
                if cur_rg is not None and cur_rg.label != rg.label:
                    insert_front = False

            if insert_front:
                best.waypoints = [(rg.x, rg.y)] + remaining
            else:
                best.waypoints = remaining + [(rg.x, rg.y)]
            best.wp_index = 0


def _dedup_points(points: Sequence[Tuple[float, float]], min_gap: float) -> List[Tuple[float, float]]:
    if not points:
        return []
    out: List[Tuple[float, float]] = []
    min_gap2 = min_gap * min_gap
    for p in points:
        if all(_dist2(p, q) >= min_gap2 for q in out):
            out.append(p)
    return out


def _nearby_priority_stops(
    grid_xy: Sequence[Tuple[float, float]],
    priority_map: Sequence[float],
    center: Tuple[float, float],
    env: Environment,
    radius: float = 90.0,
    max_points: int = 3,
    min_spacing: float = 18.0,
) -> List[Tuple[float, float]]:
    if not grid_xy or not priority_map:
        return []

    cx, cy = center
    r2 = radius * radius
    scored: List[Tuple[float, float, float]] = []
    for i, (x, y) in enumerate(grid_xy):
        if env.in_obstacle(x, y):
            continue
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 > r2:
            continue
        closeness = 1.0 - min(1.0, sqrt(d2) / max(1e-6, radius))
        score = float(priority_map[i]) * (0.45 + 0.55 * closeness)
        scored.append((score, x, y))

    scored.sort(key=lambda it: it[0], reverse=True)
    out: List[Tuple[float, float]] = []
    min_spacing2 = min_spacing * min_spacing
    for _, x, y in scored:
        p = (x, y)
        if all(_dist2(p, q) >= min_spacing2 for q in out):
            out.append(p)
            if len(out) >= max_points:
                break
    return out


def _pick_waypoint_index_for_state(
    waypoints: Sequence[Tuple[float, float]],
    state: AgentState,
) -> int:
    if not waypoints:
        return 0
    best_i = 0
    best_score = 1e30
    for i, (x, y) in enumerate(waypoints):
        d = sqrt((x - state.x) ** 2 + (y - state.y) ** 2)
        bearing = atan2(y - state.y, x - state.x)
        heading_err = abs(wrap_to_pi(bearing - state.psi))
        score = d + 7.0 * heading_err
        if score < best_score:
            best_score = score
            best_i = i
    return best_i


def _fallback_patrol_loop(state: AgentState, env: Environment, span: float = 34.0) -> List[Tuple[float, float]]:
    xmin, xmax = env.xlim
    ymin, ymax = env.ylim

    cx = min(max(state.x, xmin + 12.0), xmax - 12.0)
    cy = min(max(state.y, ymin + 12.0), ymax - 12.0)

    pts = [
        (cx + span, cy),
        (cx, cy + span),
        (cx - span, cy),
        (cx, cy - span),
    ]
    out: List[Tuple[float, float]] = []
    for x, y in pts:
        px = min(max(x, xmin + 6.0), xmax - 6.0)
        py = min(max(y, ymin + 6.0), ymax - 6.0)
        if not env.in_obstacle(px, py):
            out.append((px, py))
    if not out:
        out.append((cx, cy))
    return out


def _transition_hits_obstacle(
    start: AgentState,
    end: AgentState,
    env: Environment,
    samples: int = 6,
    margin: float = OBSTACLE_SAFE_MARGIN,
) -> bool:
    if env.in_obstacle(end.x, end.y):
        return True
    if _point_too_close_to_obstacle(end.x, end.y, env, margin=margin):
        return True
    for k in range(1, samples + 1):
        a = k / float(samples)
        x = start.x + (end.x - start.x) * a
        y = start.y + (end.y - start.y) * a
        if env.in_obstacle(x, y):
            return True
        if _point_too_close_to_obstacle(x, y, env, margin=margin):
            return True
    return False


def _transition_hits_boundary(
    start: AgentState,
    end: AgentState,
    env: Environment,
    samples: int = 6,
    margin: float = USV_BOUNDARY_HARD_MARGIN,
) -> bool:
    if _point_too_close_to_boundary(end.x, end.y, env, margin=margin):
        return True
    for k in range(1, samples + 1):
        a = k / float(samples)
        x = start.x + (end.x - start.x) * a
        y = start.y + (end.y - start.y) * a
        if _point_too_close_to_boundary(x, y, env, margin=margin):
            return True
    return False


def _rect_clearance(x: float, y: float, rect: ObstacleRect) -> float:
    if rect.contains(x, y):
        return 0.0
    dx = max(rect.xmin - x, 0.0, x - rect.xmax)
    dy = max(rect.ymin - y, 0.0, y - rect.ymax)
    return sqrt(dx * dx + dy * dy)


def _min_obstacle_clearance(x: float, y: float, env: Environment) -> float:
    if not env.obstacles:
        return 1e6
    return min(_rect_clearance(x, y, obs) for obs in env.obstacles)


def _point_too_close_to_obstacle(x: float, y: float, env: Environment, margin: float) -> bool:
    return _min_obstacle_clearance(x, y, env) < margin


def _usv_conflict(
    cand: AgentState,
    other_positions: Sequence[Tuple[float, float]],
    safe_distance: float,
) -> bool:
    lim2 = safe_distance * safe_distance
    for ox, oy in other_positions:
        if (cand.x - ox) ** 2 + (cand.y - oy) ** 2 < lim2:
            return True
    return False


def _nearest_ship_distance(state: AgentState, other_positions: Sequence[Tuple[float, float]]) -> float:
    if not other_positions:
        return 1e9
    return min(sqrt((state.x - ox) ** 2 + (state.y - oy) ** 2) for ox, oy in other_positions)


def _safe_usv_step(
    state: AgentState,
    control: Control,
    params: AgentParams,
    env: Environment,
    dt: float,
    t: float,
    other_positions: Sequence[Tuple[float, float]],
    safe_distance: float,
    preferred_turn_sign: float = 0.0,
    prev_omega: float = 0.0,
    target_heading_error: float | None = None,
    target_distance: float | None = None,
) -> AgentState:
    # Layered hard safety: do not relax minimum inter-ship distance.
    conflict_safe_distance = safe_distance
    turn_sign = 1.0 if preferred_turn_sign > 0.0 else (-1.0 if preferred_turn_sign < 0.0 else 0.0)

    # Long-range obstacle probe: steer away before entering obstacle boundaries.
    avoid_sign = 0.0
    if env.obstacles:
        hx = cos(state.psi)
        hy = sin(state.psi)
        lookahead = max(
            USV_OBSTACLE_LOOKAHEAD_MIN,
            params.sensor.radius * USV_OBSTACLE_LOOKAHEAD_SCALE + max(0.0, state.v) * 2.0,
        )
        samples = max(8, int(lookahead / 3.0))
        hit_at: Tuple[float, float] | None = None
        for k in range(1, samples + 1):
            d = lookahead * (k / float(samples))
            px = state.x + hx * d
            py = state.y + hy * d
            if _point_too_close_to_obstacle(px, py, env, margin=USV_OBSTACLE_PROBE_MARGIN):
                hit_at = (px, py)
                break
        if hit_at is not None:
            px, py = hit_at
            nearest = min(env.obstacles, key=lambda obs: _rect_clearance(px, py, obs))
            cx = 0.5 * (nearest.xmin + nearest.xmax)
            cy = 0.5 * (nearest.ymin + nearest.ymax)
            rel_x = cx - state.x
            rel_y = cy - state.y
            side = hx * rel_y - hy * rel_x  # >0 means obstacle is on the left side
            avoid_sign = -1.0 if side > 0.0 else 1.0

    # Long-range boundary probe: treat map edges as virtual walls and turn before clamping at borders.
    hx = cos(state.psi)
    hy = sin(state.psi)
    lookahead_bnd = max(
        USV_BOUNDARY_LOOKAHEAD_MIN,
        params.sensor.radius * USV_BOUNDARY_LOOKAHEAD_SCALE + max(0.0, state.v) * 2.0,
    )
    samples_bnd = max(8, int(lookahead_bnd / 3.0))
    boundary_hit = False
    for k in range(1, samples_bnd + 1):
        d = lookahead_bnd * (k / float(samples_bnd))
        px = state.x + hx * d
        py = state.y + hy * d
        if _point_too_close_to_boundary(px, py, env, margin=USV_BOUNDARY_PROBE_MARGIN):
            boundary_hit = True
            break
    if boundary_hit and abs(avoid_sign) < 1e-9:
        cx = 0.5 * (env.xlim[0] + env.xlim[1])
        cy = 0.5 * (env.ylim[0] + env.ylim[1])
        desired = atan2(cy - state.y, cx - state.x)
        dpsi = wrap_to_pi(desired - state.psi)
        edge_clear_now = _edge_clearance(state.x, state.y, env)
        target_sign = 0.0
        prefer_target_side = False
        if target_heading_error is not None and abs(target_heading_error) > USV_AVOID_SIGN_DPSI_DEADBAND:
            target_sign = 1.0 if target_heading_error > 0.0 else -1.0
            near_target = (
                target_distance is not None
                and target_distance <= USV_TARGET_SIDE_NEAR_DIST_RATIO * safe_distance
            )
            prefer_target_side = (
                (abs(target_heading_error) <= USV_TARGET_SIDE_HEADING_MAX or near_target)
                and edge_clear_now >= USV_BOUNDARY_PROBE_MARGIN + USV_TARGET_SIDE_EDGE_BUFFER
            )
        if abs(dpsi) > USV_AVOID_SIGN_DPSI_DEADBAND:
            cand_sign = 1.0 if dpsi > 0.0 else -1.0
            if (
                prefer_target_side
                and target_sign != 0.0
                and cand_sign * target_sign < 0.0
                and abs(dpsi)
                <= (
                    USV_TARGET_SIDE_NEAR_OVERRIDE_DPSI_MAX
                    if (
                        target_distance is not None
                        and target_distance <= USV_TARGET_SIDE_NEAR_DIST_RATIO * safe_distance
                    )
                    else USV_TARGET_SIDE_OVERRIDE_DPSI_MAX
                )
            ):
                cand_sign = target_sign
            if (
                turn_sign != 0.0
                and cand_sign * turn_sign < 0.0
                and abs(dpsi) < 0.45
                and not prefer_target_side
            ):
                avoid_sign = turn_sign
            else:
                avoid_sign = cand_sign
        elif (
            turn_sign != 0.0
            and _edge_clearance(state.x, state.y, env) < USV_BOUNDARY_PROBE_MARGIN + USV_AVOID_SIGN_KEEP_EDGE_MARGIN
        ):
            avoid_sign = turn_sign

    if abs(avoid_sign) > 0.0:
        proactive_candidates = (
            Control(a=min(control.a, -0.45 * params.a_max), omega=avoid_sign * params.omega_max),
            Control(a=-0.70 * params.a_max, omega=avoid_sign * 0.82 * params.omega_max),
            Control(a=-params.a_max, omega=avoid_sign * 0.58 * params.omega_max),
        )
        best_state: AgentState | None = None
        best_score = -1e9
        for cand_ctrl in proactive_candidates:
            cand_state = step_agent(state, cand_ctrl, params, env, dt, t)
            if _transition_hits_obstacle(state, cand_state, env, margin=OBSTACLE_SAFE_MARGIN + 2.0):
                continue
            if _transition_hits_boundary(state, cand_state, env, margin=USV_BOUNDARY_HARD_MARGIN):
                continue
            if _usv_conflict(cand_state, other_positions, safe_distance=conflict_safe_distance):
                continue
            clearance = _min_obstacle_clearance(cand_state.x, cand_state.y, env)
            edge_clear = _edge_clearance(cand_state.x, cand_state.y, env)
            omega_change_pen = USV_OMEGA_CHANGE_PENALTY * abs(cand_ctrl.omega - prev_omega)
            turn_flip_pen = USV_TURN_FLIP_PENALTY if (turn_sign != 0.0 and cand_ctrl.omega * turn_sign < 0.0) else 0.0
            score = clearance + 0.35 * edge_clear - omega_change_pen - turn_flip_pen
            if score > best_score:
                best_score = score
                best_state = cand_state
        if best_state is not None:
            return best_state

    primary = step_agent(state, control, params, env, dt, t)
    if (
        not _transition_hits_obstacle(state, primary, env)
        and not _transition_hits_boundary(state, primary, env, margin=USV_BOUNDARY_HARD_MARGIN)
        and not _usv_conflict(
        primary, other_positions, safe_distance=conflict_safe_distance
        )
    ):
        return primary

    candidates: List[Control] = []
    # Proactive obstacle/collision bypass candidates.
    for a_scale in (-1.0, -0.6, -0.2, 0.0):
        for omega_scale in (1.0, 0.72, 0.48, 0.25):
            candidates.append(Control(a=a_scale * params.a_max, omega=params.omega_max * omega_scale))
            candidates.append(Control(a=a_scale * params.a_max, omega=-params.omega_max * omega_scale))
    candidates.append(Control(a=-params.a_max, omega=0.0))

    best_state: AgentState | None = None
    best_score = -1e9
    for cand_ctrl in candidates:
        cand_state = step_agent(state, cand_ctrl, params, env, dt, t)
        if _transition_hits_obstacle(state, cand_state, env):
            continue
        if _transition_hits_boundary(state, cand_state, env, margin=USV_BOUNDARY_HARD_MARGIN):
            continue
        if _usv_conflict(cand_state, other_positions, safe_distance=conflict_safe_distance):
            continue
        clearance = _min_obstacle_clearance(cand_state.x, cand_state.y, env)
        edge_clear = _edge_clearance(cand_state.x, cand_state.y, env)
        nearest_ship = min(
            [sqrt((cand_state.x - ox) ** 2 + (cand_state.y - oy) ** 2) for ox, oy in other_positions] or [1e6]
        )
        omega_change_pen = USV_OMEGA_CHANGE_PENALTY * abs(cand_ctrl.omega - prev_omega)
        turn_flip_pen = USV_TURN_FLIP_PENALTY if (turn_sign != 0.0 and cand_ctrl.omega * turn_sign < 0.0) else 0.0
        score = (
            clearance
            + 0.32 * edge_clear
            + 0.2 * nearest_ship
            - 0.22 * abs(cand_ctrl.omega)
            - omega_change_pen
            - turn_flip_pen
        )
        if score > best_score:
            best_score = score
            best_state = cand_state

    if best_state is not None:
        return best_state

    # Emergency rotate-in-place to break deadlock near obstacle or map boundaries.
    new_psi = state.psi
    near_boundary = _point_too_close_to_boundary(state.x, state.y, env, margin=USV_BOUNDARY_PROBE_MARGIN + 0.8)
    if near_boundary:
        cx = 0.5 * (env.xlim[0] + env.xlim[1])
        cy = 0.5 * (env.ylim[0] + env.ylim[1])
        desired = atan2(cy - state.y, cx - state.x)
        dpsi = wrap_to_pi(desired - state.psi)
        max_step = 0.55
        new_psi = wrap_to_pi(state.psi + max(-max_step, min(max_step, dpsi)))
    elif env.obstacles:
        nearest = min(
            env.obstacles,
            key=lambda obs: _rect_clearance(state.x, state.y, obs),
        )
        cx = 0.5 * (nearest.xmin + nearest.xmax)
        cy = 0.5 * (nearest.ymin + nearest.ymax)
        desired = atan2(state.y - cy, state.x - cx)
        dpsi = wrap_to_pi(desired - state.psi)
        max_step = 0.55
        new_psi = wrap_to_pi(state.psi + max(-max_step, min(max_step, dpsi)))

    return AgentState(
        x=state.x,
        y=state.y,
        psi=new_psi,
        v=0.0,
        energy=max(0.0, state.energy - 0.15 * params.base_power * dt),
    )


def _in_active_region(
    state: AgentState,
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
) -> bool:
    for rg in regions:
        if processing_done_time.get(rg.label) is not None:
            continue
        dx = state.x - rg.x
        dy = state.y - rg.y
        if dx * dx + dy * dy <= (0.98 * rg.radius) ** 2:
            return True
    return False


def _active_region_for_state(
    state: AgentState,
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
) -> DetectionRegion | None:
    for rg in regions:
        if processing_done_time.get(rg.label) is not None:
            continue
        dx = state.x - rg.x
        dy = state.y - rg.y
        if dx * dx + dy * dy <= (1.02 * rg.radius) ** 2:
            return rg
    return None


def _capture_region_for_state(
    state: AgentState,
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
    prefer_label: str | None = None,
) -> DetectionRegion | None:
    if prefer_label is not None:
        for rg in regions:
            if rg.label != prefer_label or processing_done_time.get(rg.label) is not None:
                continue
            dx = state.x - rg.x
            dy = state.y - rg.y
            if dx * dx + dy * dy <= (TASK_CAPTURE_RADIUS_RATIO * rg.radius) ** 2:
                return rg
    return None


def _inside_anchor_for_region(state: AgentState, rg: DetectionRegion) -> Tuple[float, float]:
    dx = state.x - rg.x
    dy = state.y - rg.y
    d = sqrt(dx * dx + dy * dy)
    anchor_r = TASK_LOCK_ANCHOR_RATIO * rg.radius
    if d <= max(1e-6, anchor_r):
        return (state.x, state.y)
    ux = dx / max(1e-6, d)
    uy = dy / max(1e-6, d)
    return (rg.x + ux * anchor_r, rg.y + uy * anchor_r)


def _position_conflict(
    p: Tuple[float, float],
    other_positions: Sequence[Tuple[float, float]],
    safe_distance: float,
) -> bool:
    lim2 = safe_distance * safe_distance
    for ox, oy in other_positions:
        if (p[0] - ox) ** 2 + (p[1] - oy) ** 2 < lim2:
            return True
    return False


def _pick_task_lock_anchor(
    state: AgentState,
    rg: DetectionRegion,
    env: Environment,
    other_positions: Sequence[Tuple[float, float]],
    safe_distance: float,
    is_primary: bool,
    queue_slot_idx: int,
    queue_slot_count: int,
    allow_inner_hold: bool = False,
) -> Tuple[float, float]:
    boundary_margin = max(USV_ROUTE_BOUNDARY_MARGIN, 0.85 * safe_distance)
    obstacle_margin = OBSTACLE_SAFE_MARGIN + 1.0
    best_any = (state.x, state.y)
    best_min_d = -1.0

    # If a queued USV has already entered the target disk and it is still safe,
    # keep it inside to avoid unnecessary leave-and-return loops.
    if allow_inner_hold:
        inside = _inside_anchor_for_region(state, rg)
        if (
            not env.in_obstacle(inside[0], inside[1])
            and not _point_too_close_to_obstacle(inside[0], inside[1], env, margin=obstacle_margin)
            and not _position_conflict(inside, other_positions, safe_distance=safe_distance)
        ):
            return inside

    if is_primary:
        base = _inside_anchor_for_region(state, rg)
        base_angle = atan2(base[1] - rg.y, base[0] - rg.x)
        radii = [
            max(1.2, 0.58 * rg.radius),
            max(1.6, TASK_LOCK_ANCHOR_RATIO * rg.radius),
            max(1.8, 1.02 * rg.radius),
            max(2.0, rg.radius + 0.45 * safe_distance),
            max(2.2, rg.radius + 0.90 * safe_distance),
        ]
        angles = [base_angle, base_angle + 0.35, base_angle - 0.35, base_angle + 0.75, base_angle - 0.75, base_angle + pi]
    else:
        n = max(1, queue_slot_count)
        slot = max(0, queue_slot_idx) % n
        base_angle = 2.0 * pi * (slot / float(n))
        wait_r0 = max(
            TASK_QUEUE_RADIUS_RATIO * rg.radius,
            rg.radius + 0.95 * safe_distance,
            TASK_CAPTURE_RADIUS_RATIO * rg.radius,
        )
        radii = [
            wait_r0,
            wait_r0 + TASK_QUEUE_LAYER_STEP * safe_distance,
            wait_r0 + 1.10 * safe_distance,
            wait_r0 + 1.55 * safe_distance,
        ]
        angles = [base_angle, base_angle + 0.45, base_angle - 0.45, base_angle + 0.90, base_angle - 0.90, base_angle + pi]

    for rr in radii:
        for ang in angles:
            x = rg.x + rr * cos(ang)
            y = rg.y + rr * sin(ang)
            x = min(max(x, env.xlim[0] + boundary_margin), env.xlim[1] - boundary_margin)
            y = min(max(y, env.ylim[0] + boundary_margin), env.ylim[1] - boundary_margin)
            p = (x, y)
            if env.in_obstacle(x, y):
                continue
            if _point_too_close_to_obstacle(x, y, env, margin=obstacle_margin):
                continue
            min_d = min([sqrt((x - ox) ** 2 + (y - oy) ** 2) for ox, oy in other_positions] or [1e6])
            if min_d > best_min_d:
                best_min_d = min_d
                best_any = p
            if not _position_conflict(p, other_positions, safe_distance=safe_distance):
                return p
    return best_any


def _pending_region_for_waypoint(
    waypoint: Tuple[float, float],
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
    snap_tol: float = 2.5,
) -> DetectionRegion | None:
    wx, wy = waypoint
    lim2 = snap_tol * snap_tol
    for rg in regions:
        if processing_done_time.get(rg.label) is not None:
            continue
        if (wx - rg.x) ** 2 + (wy - rg.y) ** 2 <= lim2:
            return rg
    return None


def _completed_region_for_waypoint(
    waypoint: Tuple[float, float],
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
    snap_tol: float = 3.0,
) -> DetectionRegion | None:
    wx, wy = waypoint
    lim2 = snap_tol * snap_tol
    for rg in regions:
        if processing_done_time.get(rg.label) is None:
            continue
        if (wx - rg.x) ** 2 + (wy - rg.y) ** 2 <= lim2:
            return rg
    return None


def _build_proximity_detour(
    usv: AgentRunner,
    current_target: Tuple[float, float],
    other_positions: Sequence[Tuple[float, float]],
    env: Environment,
    safe_distance: float,
) -> Tuple[float, float] | None:
    if not other_positions:
        return None

    sx, sy = usv.state.x, usv.state.y
    tx, ty = current_target
    dx = tx - sx
    dy = ty - sy
    dist_to_target = sqrt(dx * dx + dy * dy)
    if dist_to_target < 1e-6:
        return None

    ux = dx / dist_to_target
    uy = dy / dist_to_target
    lookahead = min(max(12.0, dist_to_target), 32.0)
    trigger = USV_PROX_TRIGGER_RATIO * safe_distance
    min_sep = USV_PROX_MIN_SEP_RATIO * safe_distance

    best_other: Tuple[float, float] | None = None
    best_score = -1e9
    for ox, oy in other_positions:
        rx = ox - sx
        ry = oy - sy
        d_now = sqrt(rx * rx + ry * ry)
        if d_now > trigger:
            continue
        along = rx * ux + ry * uy
        if along < -4.0:
            continue
        proj = min(max(along, 0.0), lookahead)
        cx = sx + ux * proj
        cy = sy + uy * proj
        d_close = sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
        if d_close > min_sep:
            continue
        score = (trigger - d_now) + 1.7 * (min_sep - d_close)
        if score > best_score:
            best_score = score
            best_other = (ox, oy)

    if best_other is None:
        return None

    ox, oy = best_other
    side = ux * (oy - sy) - uy * (ox - sx)  # >0 means other ship on left side
    preferred_sign = -1.0 if side > 0.0 else 1.0
    lateral = max(4.0, USV_DETOUR_LATERAL_RATIO * safe_distance)
    forward = max(6.0, min(lookahead, USV_DETOUR_FORWARD_RATIO * safe_distance))
    px = -uy
    py = ux

    candidates: List[Tuple[float, float]] = []
    for sign in (preferred_sign, -preferred_sign):
        x = sx + ux * forward + px * sign * lateral
        y = sy + uy * forward + py * sign * lateral
        x = min(max(x, env.xlim[0] + IDLE_EDGE_MARGIN), env.xlim[1] - IDLE_EDGE_MARGIN)
        y = min(max(y, env.ylim[0] + IDLE_EDGE_MARGIN), env.ylim[1] - IDLE_EDGE_MARGIN)
        if env.in_obstacle(x, y):
            continue
        if _point_too_close_to_obstacle(x, y, env, margin=OBSTACLE_SAFE_MARGIN):
            continue
        if any((x - qx) ** 2 + (y - qy) ** 2 < min_sep * min_sep for qx, qy in other_positions):
            continue
        candidates.append((x, y))

    if not candidates:
        return None

    # Keep the change minimal: choose detour with smallest extra path cost.
    best = candidates[0]
    base = sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
    best_extra = 1e30
    for x, y in candidates:
        extra = sqrt((x - sx) ** 2 + (y - sy) ** 2) + sqrt((tx - x) ** 2 + (ty - y) ** 2) - base
        if extra < best_extra:
            best_extra = extra
            best = (x, y)
    return best


def _maybe_insert_minimal_detour(
    usv: AgentRunner,
    other_positions: Sequence[Tuple[float, float]],
    env: Environment,
    safe_distance: float,
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
    cooldown_steps: int,
) -> bool:
    if cooldown_steps > 0 or not usv.waypoints:
        return False
    current_target = usv.current_target()
    if _pending_region_for_waypoint(current_target, regions, processing_done_time) is not None:
        return False
    detour = _build_proximity_detour(
        usv=usv,
        current_target=current_target,
        other_positions=other_positions,
        env=env,
        safe_distance=safe_distance,
    )
    if detour is None:
        return False

    remaining = usv.waypoints[usv.wp_index :] + usv.waypoints[: usv.wp_index]
    if remaining and _dist2(detour, remaining[0]) < USV_DETOUR_MIN_GAP * USV_DETOUR_MIN_GAP:
        return False
    usv.waypoints = [detour] + remaining
    usv.wp_index = 0
    return True


def _nearest_pending_region_ahead(
    usv: AgentRunner,
    regions: Sequence[DetectionRegion],
    processing_done_time: Dict[str, float | None],
) -> DetectionRegion | None:
    best: DetectionRegion | None = None
    best_d2 = 1e30
    for rg in regions:
        if processing_done_time.get(rg.label) is not None:
            continue
        dx = rg.x - usv.state.x
        dy = rg.y - usv.state.y
        d2 = dx * dx + dy * dy
        dist_gate = max(20.0, 1.8 * rg.radius)
        if d2 > dist_gate * dist_gate:
            continue
        bearing = atan2(dy, dx)
        heading_err = abs(wrap_to_pi(bearing - usv.state.psi))
        if heading_err > 0.95:
            continue
        if d2 < best_d2:
            best_d2 = d2
            best = rg
    return best


def _outward_exploration_waypoints(name: str, state: AgentState, env: Environment) -> List[Tuple[float, float]]:
    xmin, xmax = env.xlim
    ymin, ymax = env.ylim
    center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)

    vx = state.x - center[0]
    vy = state.y - center[1]
    norm = sqrt(vx * vx + vy * vy)
    if norm < 1e-6:
        idx = sum(ord(ch) for ch in name) % 3
        dirs = [(1.0, 0.25), (0.2, 1.0), (-0.9, 0.5)]
        vx, vy = dirs[idx]
        norm = sqrt(vx * vx + vy * vy)

    ux = vx / max(norm, 1e-6)
    uy = vy / max(norm, 1e-6)
    tx = -uy
    ty = ux

    # Near map boundaries, pure outward points collapse onto clipped edge coordinates,
    # which can create small loops around nearly identical waypoints.
    toward_left_edge = state.x - xmin < 44.0 and ux < -0.25
    toward_right_edge = xmax - state.x < 44.0 and ux > 0.25
    toward_bottom_edge = state.y - ymin < 44.0 and uy < -0.25
    toward_top_edge = ymax - state.y < 44.0 and uy > 0.25
    if toward_left_edge or toward_right_edge or toward_bottom_edge or toward_top_edge:
        # Pick the tangent side that keeps more edge clearance and smaller heading correction.
        best_score = -1e9
        best_dir = (tx, ty)
        for tangent_sign in (1.0, -1.0):
            dux = tangent_sign * tx
            duy = tangent_sign * ty
            test_x = state.x + dux * 56.0
            test_y = state.y + duy * 56.0
            px = min(max(test_x, xmin + IDLE_EDGE_MARGIN), xmax - IDLE_EDGE_MARGIN)
            py = min(max(test_y, ymin + IDLE_EDGE_MARGIN), ymax - IDLE_EDGE_MARGIN)
            edge_clear = min(px - xmin, xmax - px, py - ymin, ymax - py)
            heading_err = abs(wrap_to_pi(atan2(py - state.y, px - state.x) - state.psi))
            score = edge_clear - 9.0 * heading_err
            if score > best_score:
                best_score = score
                best_dir = (dux, duy)
        ux, uy = best_dir
        tx, ty = -uy, ux

    offsets = [52.0, 104.0, 156.0]
    pts: List[Tuple[float, float]] = []
    for i, d in enumerate(offsets):
        lat = 20.0 if i % 2 == 0 else -20.0
        x = state.x + ux * d + tx * lat
        y = state.y + uy * d + ty * lat
        px = min(max(x, xmin + IDLE_EDGE_MARGIN), xmax - IDLE_EDGE_MARGIN)
        py = min(max(y, ymin + IDLE_EDGE_MARGIN), ymax - IDLE_EDGE_MARGIN)
        if env.in_obstacle(px, py):
            continue
        pts.append((px, py))

    dedup: List[Tuple[float, float]] = []
    min_gap2 = IDLE_WAYPOINT_MIN_GAP * IDLE_WAYPOINT_MIN_GAP
    for p in pts:
        if all(_dist2(p, q) >= min_gap2 for q in dedup):
            dedup.append(p)
    return dedup


def _assign_idle_waypoints(
    usv: AgentRunner,
    env: Environment,
    grid_xy: Sequence[Tuple[float, float]],
    priority_map: Sequence[float] | None,
    force_exploration: bool = False,
) -> None:
    local_priority: List[Tuple[float, float]] = []
    if priority_map and not force_exploration:
        local_priority = _nearby_priority_stops(
            grid_xy=grid_xy,
            priority_map=priority_map,
            center=(usv.state.x, usv.state.y),
            env=env,
            radius=110.0,
            max_points=4,
            min_spacing=16.0,
        )

    if local_priority:
        usv.waypoints = local_priority
        usv.wp_index = 0
        return

    outward = _outward_exploration_waypoints(usv.name, usv.state, env)
    if outward:
        sx, sy, spsi = usv.state.x, usv.state.y, usv.state.psi
        outward.sort(
            key=lambda p: 12.0 * abs(wrap_to_pi(atan2(p[1] - sy, p[0] - sx) - spsi))
            + sqrt(_dist2((sx, sy), p))
        )
        usv.waypoints = outward
        usv.wp_index = 0
        return

    usv.waypoints = _fallback_patrol_loop(usv.state, env=env, span=(56.0 if force_exploration else 34.0))
    usv.wp_index = 0


def _update_region_processing(
    usvs: Sequence[AgentRunner],
    regions: Sequence[DetectionRegion],
    processing_progress: Dict[str, float],
    processing_required: Dict[str, float],
    processing_done_time: Dict[str, float | None],
    first_hit_time: Dict[str, float | None],
    known_difficulty_by_task: Dict[str, str | None],
    difficulty_discovered_by: Dict[str, str | None],
    difficulty_discovered_time: Dict[str, float | None],
    usv_work_s: Dict[str, float] | None,
    sim_time: float,
    dt: float,
) -> bool:
    discovered_changed = False
    for rg in regions:
        if processing_done_time.get(rg.label) is not None:
            continue

        radius2 = (1.02 * rg.radius) ** 2
        contributors = 0
        contributor_names: List[str] = []
        for usv in usvs:
            dx = usv.state.x - rg.x
            dy = usv.state.y - rg.y
            if dx * dx + dy * dy <= radius2:
                contributors += 1
                contributor_names.append(usv.name)
                if first_hit_time.get(rg.label) is None:
                    first_hit_time[rg.label] = sim_time
                if known_difficulty_by_task.get(rg.label) is None:
                    known_difficulty_by_task[rg.label] = rg.difficulty
                    difficulty_discovered_by[rg.label] = usv.name
                    difficulty_discovered_time[rg.label] = sim_time
                    discovered_changed = True

        if contributors == 0:
            continue

        if usv_work_s is not None:
            for nm in contributor_names:
                usv_work_s[nm] = usv_work_s.get(nm, 0.0) + dt

        # More boats working on the same region yields super-linear speedup.
        coop_factor = 1.0 + 0.80 * (contributors - 1)
        work_gain = dt * float(contributors) * coop_factor
        processing_progress[rg.label] = processing_progress.get(rg.label, 0.0) + work_gain
        if processing_progress[rg.label] >= processing_required[rg.label]:
            processing_done_time[rg.label] = sim_time
    return discovered_changed


def _task_label_key(label: str) -> int:
    try:
        return int(label.split("-")[-1])
    except ValueError:
        return 10**9


def _write_run_log(
    log_dir: Path,
    mission_seed: int,
    profile_name: str,
    scenario_profile: str,
    env: Environment,
    regions: Sequence[DetectionRegion],
    stage1_snapshot_step: int,
    replan_count: int,
    executed_steps: int,
    coverage_rate: float,
    all_problem_rate: float,
    turn_events: int,
    turn_index_global: float,
    mission_completion_time: float,
    avg_response_time: float,
    avg_processing_completion_time: float,
    processing_required: Dict[str, float],
    processing_progress: Dict[str, float],
    processing_done_time: Dict[str, float | None],
    first_hit_time: Dict[str, float | None],
    known_difficulty_by_task: Dict[str, str | None],
    difficulty_discovered_by: Dict[str, str | None],
    difficulty_discovered_time: Dict[str, float | None],
    task_assignment_latest: Dict[str, List[str]],
    task_assignment_history: Sequence[Tuple[int, Dict[str, List[str]]]],
    max_contributors_by_region: Dict[str, int],
    usv_travel_m: Dict[str, float],
    usv_work_s: Dict[str, float],
    usv_load_eq_m: Dict[str, float],
    load_balance_cv: float,
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = log_dir / f"mission_seed{mission_seed}_{ts}.log"

    lines: List[str] = []
    lines.append("=== UAV-USV Mission Log ===")
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"seed: {mission_seed}")
    lines.append(f"algorithm: {profile_name}")
    lines.append(f"scenario_profile: {scenario_profile}")
    lines.append(f"sea_area: {env.xlim[1]-env.xlim[0]:.0f}m x {env.ylim[1]-env.ylim[0]:.0f}m")
    lines.append(f"obstacles: {len(env.obstacles)}")
    lines.append(f"tasks_total: {len(regions)}")
    lines.append(f"stage1_snapshot_step: {stage1_snapshot_step}")
    lines.append(f"replans: {replan_count}")
    lines.append(f"executed_steps: {executed_steps}")
    lines.append(f"coverage_rate: {coverage_rate:.4f}")
    lines.append(f"task_handling_rate: {all_problem_rate:.4f}")
    lines.append(f"turn_events: {turn_events}")
    lines.append(f"turn_index: {turn_index_global:.6f}")
    lines.append(f"load_balance_cv: {load_balance_cv:.4f}")
    lines.append(f"mission_completion_time_s: {mission_completion_time:.2f}")
    lines.append(f"avg_response_time_s: {avg_response_time:.2f}")
    lines.append(f"avg_processing_completion_time_s: {avg_processing_completion_time:.2f}")
    lines.append("")

    lines.append("=== USV Load Balance ===")
    lines.append("format: usv | travel_m | work_s | load_eq_m")
    for nm in sorted(usv_travel_m.keys()):
        lines.append(
            f"{nm} | {usv_travel_m.get(nm, 0.0):.1f} | "
            f"{usv_work_s.get(nm, 0.0):.1f} | {usv_load_eq_m.get(nm, 0.0):.1f}"
        )
    lines.append("")

    lines.append("=== Task Definitions (12 tasks) ===")
    lines.append("format: label | true_difficulty | known_difficulty | base_sec | discovered_by@time | pos(x,y)")
    for rg in sorted(regions, key=lambda r: _task_label_key(r.label)):
        base_sec = TASK_BASE_SECONDS[rg.difficulty]
        diff_cn = TASK_LEVEL_CN.get(rg.difficulty, rg.difficulty)
        known = known_difficulty_by_task.get(rg.label) or UNKNOWN_DIFFICULTY
        known_cn = TASK_LEVEL_CN.get(known, known)
        discoverer = difficulty_discovered_by.get(rg.label)
        discover_time = difficulty_discovered_time.get(rg.label)
        discover_txt = "-"
        if discoverer is not None and discover_time is not None:
            discover_txt = f"{discoverer}@{discover_time:.1f}s"
        lines.append(
            f"{rg.label} | {diff_cn}({rg.difficulty}) | {known_cn}({known}) | {base_sec:.1f}s | "
            f"{discover_txt} | ({rg.x:.1f}, {rg.y:.1f})"
        )
    lines.append("")

    lines.append("=== Task Assignment (latest) ===")
    lines.append("format: label | assignees | first_response | complete_time | progress/required | max_collab | status")
    for rg in sorted(regions, key=lambda r: _task_label_key(r.label)):
        assignees = task_assignment_latest.get(rg.label, [])
        assignees_txt = ",".join(assignees) if assignees else "-"
        first_resp = first_hit_time.get(rg.label)
        done_at = processing_done_time.get(rg.label)
        progress = processing_progress.get(rg.label, 0.0)
        required = processing_required.get(rg.label, 0.0)
        max_collab = max_contributors_by_region.get(rg.label, 0)
        status = "DONE" if done_at is not None else "PENDING"
        lines.append(
            f"{rg.label} | {assignees_txt} | "
            f"{('-' if first_resp is None else f'{first_resp:.1f}s')} | "
            f"{('-' if done_at is None else f'{done_at:.1f}s')} | "
            f"{progress:.2f}/{required:.2f} | {max_collab} | {status}"
        )
    lines.append("")

    lines.append("=== Assignment History By Replan ===")
    for ridx, assignments in task_assignment_history:
        parts: List[str] = []
        for rg in sorted(regions, key=lambda r: _task_label_key(r.label)):
            owners = assignments.get(rg.label, [])
            owners_txt = ",".join(owners) if owners else "-"
            parts.append(f"{rg.label}->{owners_txt}")
        lines.append(f"[replan {ridx:02d}] " + " ; ".join(parts))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)


def run_mission_once(
    args: argparse.Namespace,
    mission_seed: int,
    targets: MissionTargets,
    profile: AlgorithmProfile,
    generate_outputs: bool,
) -> MissionSummary:
    env, sim, uavs, usvs, regions = build_large_mission(
        seed=mission_seed,
        scenario_profile=args.scenario_profile,
    )
    stage1 = CoverageProblem(env, sim)

    uav_trajectories: Dict[str, List[AgentState]] = {a.name: [a.state] for a in uavs}
    uav_params = {a.name: a.params for a in uavs}
    region_lookup = {rg.label: rg for rg in regions}
    all_runners = uavs + usvs
    agent_types = {a.name: a.params.agent_type.value for a in all_runners}
    agent_colors: Dict[str, str] = {}
    usv_color_idx = 0
    for ag in all_runners:
        if ag.params.agent_type == AgentType.UAV:
            agent_colors[ag.name] = UAV_DISPLAY_COLOR
        else:
            agent_colors[ag.name] = USV_DISPLAY_PALETTE[usv_color_idx % len(USV_DISPLAY_PALETTE)]
            usv_color_idx += 1
    task_labels_ordered = [rg.label for rg in sorted(regions, key=lambda r: _task_label_key(r.label))]
    frame_times: List[float] = []
    frame_states: Dict[str, List[Tuple[float, float, float]]] = {a.name: [] for a in all_runners}
    frame_known_difficulty: List[Dict[str, str]] = []
    frame_task_assignment: List[Dict[str, List[str]]] = []
    frame_coverage_rate: List[float] = []
    frame_task_completion_rate: List[float] = []
    frame_turn_rate: List[float] = []
    frame_load_table: List[Dict[str, Dict[str, float]]] = []
    frame_partition_snapshot_idx: List[int] = []
    partition_owner_snapshots: List[List[str]] = []
    active_partition_snapshot_idx = -1

    def record_frame(time_s: float) -> None:
        frame_times.append(time_s)
        for ag in all_runners:
            frame_states[ag.name].append((ag.state.x, ag.state.y, ag.state.psi))
        frame_known_difficulty.append(
            {
                label: diff
                for label in task_labels_ordered
                for diff in [known_difficulty_by_task.get(label)]
                if diff is not None
            }
        )
        frame_task_assignment.append(
            {label: list(task_assignment_latest.get(label, [])) for label in task_labels_ordered}
        )
        uav_cov_now = stage1.coverage_quality
        usv_cov_now = [1.0 - m for m in usv_miss]
        mission_cov_now = [
            1.0 - (1.0 - uav_cov_now[i]) * (1.0 - usv_cov_now[i]) for i in range(grid_size)
        ]
        cov_rate = sum(1 for c in mission_cov_now if c >= sim.coverage_threshold) / max(1, grid_size)
        handled_now = sum(1 for tm in processing_done_time.values() if tm is not None and tm <= time_s)
        task_rate = handled_now / max(1, len(regions))
        turn_rate = usv_turn_sum_cum / max(1e-6, usv_dist_sum_cum)
        frame_coverage_rate.append(cov_rate)
        frame_task_completion_rate.append(task_rate)
        frame_turn_rate.append(turn_rate)
        frame_load_table.append(
            {
                name: {
                    "travel_m": float(usv_travel_m.get(name, 0.0)),
                    "work_s": float(usv_work_s.get(name, 0.0)),
                    "load_eq_m": float(_usv_load_equivalent(usv_travel_m.get(name, 0.0), usv_work_s.get(name, 0.0))),
                }
                for name in sorted(usv_by_name.keys())
            }
        )
        frame_partition_snapshot_idx.append(max(0, active_partition_snapshot_idx))

    t = 0.0
    n_steps = max(1, int(args.stage1_steps))
    latent_risk = risk_map_from_regions(stage1.grid_xy, regions)

    usv_params = {a.name: a.params for a in usvs}
    usv_by_name = {a.name: a for a in usvs}
    grid_size = len(stage1.grid_xy)
    usv_miss = [1.0] * grid_size
    usv_trajectories: Dict[str, List[AgentState]] = {a.name: [a.state] for a in usvs}
    usv_travel_m: Dict[str, float] = {a.name: 0.0 for a in usvs}
    usv_work_s: Dict[str, float] = {a.name: 0.0 for a in usvs}
    usv_turn_sum_cum = 0.0
    usv_dist_sum_cum = 0.0

    replan_count = 0
    assign_sum = {a.name: 0 for a in usvs}

    latest_partition = None
    latest_priority_map: List[float] = []
    latest_detected_regions: List[DetectionRegion] = []
    latest_final_paths: Dict[str, List[Tuple[float, float]]] = {}
    latest_priority_counts: Dict[str, int] = {a.name: 0 for a in usvs}
    stage1_coverage: List[float] | None = None
    stage1_uav_trajectories: Dict[str, List[AgentState]] | None = None
    stage1_mean_scan_quality = 0.0
    # Sticky assignment view for UI/log: keep the latest non-empty assignees for each task.
    task_assignment_latest: Dict[str, List[str]] = {rg.label: [] for rg in regions}
    # Active assignment map used by online dispatch/injection logic.
    task_assignment_active: Dict[str, List[str]] = {rg.label: [] for rg in regions}
    task_assignment_history: List[Tuple[int, Dict[str, List[str]]]] = []
    processing_required: Dict[str, float] = {}
    processing_progress: Dict[str, float] = {}
    processing_done_time: Dict[str, float | None] = {}
    first_hit_time: Dict[str, float | None] = {}
    known_difficulty_by_task: Dict[str, str | None] = {}
    difficulty_discovered_by: Dict[str, str | None] = {}
    difficulty_discovered_time: Dict[str, float | None] = {}
    max_contributors_by_region: Dict[str, int] = {rg.label: 0 for rg in regions}
    for rg in regions:
        processing_required[rg.label] = float(TASK_BASE_SECONDS[rg.difficulty])
        processing_progress[rg.label] = 0.0
        processing_done_time[rg.label] = None
        first_hit_time[rg.label] = None
        known_difficulty_by_task[rg.label] = None
        difficulty_discovered_by[rg.label] = None
        difficulty_discovered_time[rg.label] = None

    uav_hold_steps_required = max(1, int(round(1.0 / max(1e-6, sim.dt))))
    # Larger default radius for proactive UAV task identification.
    uav_identify_radius = 24.0
    uav_identify_target: Dict[str, str | None] = {uav.name: None for uav in uavs}
    uav_identify_countdown: Dict[str, int] = {uav.name: 0 for uav in uavs}
    global_scan_discovery_done = False
    usv_task_lock: Dict[str, str | None] = {usv.name: None for usv in usvs}
    usv_hold_anchor: Dict[str, Tuple[float, float] | None] = {usv.name: None for usv in usvs}
    task_lock_order: Dict[str, List[str]] = {rg.label: [] for rg in regions}
    usv_detour_cooldown: Dict[str, int] = {usv.name: 0 for usv in usvs}
    usv_idle_stuck_steps: Dict[str, int] = {usv.name: 0 for usv in usvs}
    usv_idle_stuck_anchor: Dict[str, Tuple[float, float] | None] = {usv.name: None for usv in usvs}
    usv_idle_stuck_target: Dict[str, Tuple[float, float] | None] = {usv.name: None for usv in usvs}
    usv_prev_assigned: Dict[str, bool] = {usv.name: False for usv in usvs}
    usv_prev_omega_cmd: Dict[str, float] = {usv.name: 0.0 for usv in usvs}
    usv_turn_intent_sign: Dict[str, float] = {usv.name: 0.0 for usv in usvs}

    def do_replan() -> None:
        nonlocal replan_count, latest_partition, latest_priority_map, latest_detected_regions
        nonlocal latest_final_paths, latest_priority_counts
        nonlocal active_partition_snapshot_idx

        current_global_map = stage1.coverage_quality
        _, priority_map, detected_regions = make_priority_and_detection(
            grid_xy=stage1.grid_xy,
            global_coverage=current_global_map,
            latent_risk=latent_risk,
            regions=regions,
            detection_threshold=profile.detection_threshold,
        )
        priority_map = list(priority_map)
        for rg in regions:
            idx = nearest_cell_index(stage1.grid_xy, (rg.x, rg.y))
            known = known_difficulty_by_task.get(rg.label)
            p = _difficulty_priority(known)
            priority_map[idx] = min(1.0, priority_map[idx] + 0.16 * p)

        usv_states = {a.name: a.state for a in usvs}
        partition = weighted_voronoi_partition(
            grid_xy=stage1.grid_xy,
            states=usv_states,
            params_map=usv_params,
            env=env,
            priority_map=priority_map,
            cfg=profile.voronoi_cfg,
            current_time=t,
        )
        cell_indices_by_agent: Dict[str, List[int]] = {name: [] for name in usv_states.keys()}
        for idx, owner in enumerate(partition.owner_by_cell):
            if owner in cell_indices_by_agent:
                cell_indices_by_agent[owner].append(idx)

        owned_cell_count: Dict[str, int] = {}
        owned_uncovered_count: Dict[str, int] = {}
        explore_cells_by_agent: Dict[str, List[Tuple[float, float]]] = {}
        for name, idxs in cell_indices_by_agent.items():
            owned_cell_count[name] = len(idxs)
            uncovered_pts = [stage1.grid_xy[i] for i in idxs if current_global_map[i] < OWN_REGION_EXPLORE_THRESHOLD]
            owned_uncovered_count[name] = len(uncovered_pts)
            explore_cells_by_agent[name] = uncovered_pts if uncovered_pts else partition.cells_by_agent.get(name, [])

        base_paths = plan_heterogeneous_paths(explore_cells_by_agent, usv_states, usv_params)

        priority_by_usv: Dict[str, List[Tuple[float, float]]] = {name: [] for name in usv_states.keys()}
        assignment_by_task: Dict[str, set[str]] = {rg.label: set() for rg in regions}
        discovered_labels = {label for label, diff in known_difficulty_by_task.items() if diff is not None}
        detected_labels = {rg.label for rg in detected_regions}

        virtual_load_eq = {
            name: _usv_load_equivalent(usv_travel_m.get(name, 0.0), usv_work_s.get(name, 0.0))
            for name in usv_states.keys()
        }
        virtual_queue_count = {name: 0 for name in usv_states.keys()}
        virtual_available_s = {name: 0.0 for name in usv_states.keys()}
        queued_labels_by_usv: Dict[str, set[str]] = {name: set() for name in usv_states.keys()}

        task_owner: Dict[str, str] = {}
        owned_tasks: Dict[str, List[str]] = {name: [] for name in usv_states.keys()}
        for rg in regions:
            idx = nearest_cell_index(stage1.grid_xy, (rg.x, rg.y))
            owner = partition.owner_by_cell[idx]
            task_owner[rg.label] = owner
            owned_tasks.setdefault(owner, []).append(rg.label)

        support_ready_by_usv: Dict[str, bool] = {}
        for name in usv_states.keys():
            n_cells = owned_cell_count.get(name, 0)
            uncovered = owned_uncovered_count.get(name, 0)
            remain_tol = max(SUPPORT_UNCOVERED_MIN_CELLS_TOL, int(round(SUPPORT_UNCOVERED_RATIO_TOL * max(1, n_cells))))
            own_coverage_done = uncovered <= remain_tol
            own_tasks_done = all(processing_done_time.get(lbl) is not None for lbl in owned_tasks.get(name, []))
            support_ready_by_usv[name] = own_coverage_done and own_tasks_done

        locked_by_task: Dict[str, List[str]] = {}
        for usv_name, lock_label in usv_task_lock.items():
            if lock_label is None or usv_name not in usv_states:
                continue
            if processing_done_time.get(lock_label) is not None:
                continue
            locked_by_task.setdefault(lock_label, []).append(usv_name)

        for label, names in locked_by_task.items():
            rg = region_lookup.get(label)
            if rg is None:
                continue
            known = known_difficulty_by_task.get(label)
            remaining_work_s = max(
                0.0,
                processing_required.get(label, _expected_task_seconds(known))
                - processing_progress.get(label, 0.0),
            )
            total_expected = max(1e-6, _expected_task_seconds(known))
            remain_ratio = min(1.0, remaining_work_s / total_expected)
            n_locked = max(1, len(names))
            per_agent_work_s = _expected_agent_work_seconds(known, n_locked) * remain_ratio
            per_agent_work_eq = LOAD_WORK_SECONDS_TO_EQ_M * per_agent_work_s
            for name in names:
                assignment_by_task.setdefault(label, set()).add(name)
                if label not in queued_labels_by_usv[name]:
                    priority_by_usv.setdefault(name, []).append((rg.x, rg.y))
                    queued_labels_by_usv[name].add(label)
                    virtual_queue_count[name] += 1
                virtual_available_s[name] += per_agent_work_s
                virtual_load_eq[name] += per_agent_work_eq

        unresolved_regions = [
            rg
            for rg in regions
            if processing_done_time.get(rg.label) is None and rg.label in discovered_labels
            and rg.label not in locked_by_task
        ]
        unresolved_regions.sort(
            key=lambda rg: (
                -max(
                    0.0,
                    t - float(difficulty_discovered_time.get(rg.label))
                    if difficulty_discovered_time.get(rg.label) is not None
                    else 0.0,
                ),
                -max(
                    0.0,
                    processing_required.get(rg.label, _expected_task_seconds(known_difficulty_by_task.get(rg.label)))
                    - processing_progress.get(rg.label, 0.0),
                ),
                -_expected_task_seconds(known_difficulty_by_task.get(rg.label)),
                -_difficulty_priority(known_difficulty_by_task.get(rg.label)),
                _task_label_key(rg.label),
            )
        )

        travel_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

        def travel_eta_seconds(usv_name: str, rg: DetectionRegion) -> Tuple[float, float]:
            key = (usv_name, rg.label)
            cached = travel_cache.get(key)
            if cached is not None:
                return cached
            st = usv_states[usv_name]
            dist = sqrt(_dist2((st.x, st.y), (rg.x, rg.y)))
            v_nom = max(1.0, ASSIGNMENT_TRAVEL_SPEED_FACTOR * usv_params[usv_name].v_max)
            obstacle_eq = _segment_obstacle_penalty((st.x, st.y), (rg.x, rg.y), env=env)
            travel_s = dist / v_nom + ASSIGNMENT_OBS_TO_SECONDS * obstacle_eq / max(1.0, v_nom)
            out = (travel_s, dist)
            travel_cache[key] = out
            return out

        for rg in unresolved_regions:
            known = known_difficulty_by_task.get(rg.label)
            remaining_work_s = max(
                0.0,
                processing_required.get(rg.label, _expected_task_seconds(known))
                - processing_progress.get(rg.label, 0.0),
            )
            if remaining_work_s <= 1e-6:
                continue
            task_edge_clear = _edge_clearance(rg.x, rg.y, env)
            task_is_non_edge = task_edge_clear >= ASSIGNMENT_NON_EDGE_TASK_MARGIN
            need_n = min(
                len(usv_states),
                _required_agents_for_dispatch(known, remaining_work_s),
                _max_safe_collab_for_region(rg.radius, sim.safe_distance),
            )
            owner = task_owner.get(rg.label, list(usv_states.keys())[0])
            prev_owners = set(task_assignment_active.get(rg.label, []))
            available = list(usv_states.keys())
            selected: List[str] = []
            task_wait_age_s = max(
                0.0,
                t - float(difficulty_discovered_time.get(rg.label))
                if difficulty_discovered_time.get(rg.label) is not None
                else 0.0,
            )
            wait_bonus_s = min(ASSIGNMENT_WAIT_AGE_CAP_SECONDS, ASSIGNMENT_WAIT_AGE_GAIN * task_wait_age_s)

            for _ in range(max(1, need_n)):
                if not available:
                    break
                eligible = [nm for nm in available if support_ready_by_usv.get(nm, False) or nm == owner]
                if not eligible:
                    eligible = [owner] if owner in available else list(available)

                mean_load = sum(virtual_load_eq.values()) / max(1, len(virtual_load_eq))
                best_name = eligible[0]
                best_cost = 1e30
                for name in eligible:
                    candidate = selected + [name]
                    n_agents = max(1, len(candidate))
                    total_expected = max(1e-6, _expected_task_seconds(known))
                    remain_ratio = min(1.0, remaining_work_s / total_expected)
                    per_agent_work_s = _expected_agent_work_seconds(known, n_agents) * remain_ratio

                    finish_at = 0.0
                    for nm in candidate:
                        travel_s, _ = travel_eta_seconds(nm, rg)
                        arrival = virtual_available_s[nm] + travel_s
                        finish_at = max(finish_at, arrival)
                    finish_at += per_agent_work_s

                    overload = max(0.0, virtual_load_eq[name] - mean_load)
                    queue_penalty = ASSIGNMENT_QUEUE_SECONDS * float(virtual_queue_count[name])
                    load_penalty = ASSIGNMENT_LOAD_SECONDS * overload
                    boundary_penalty = 0.0
                    if task_is_non_edge:
                        st = usv_states[name]
                        edge_clear = _edge_clearance(st.x, st.y, env)
                        if edge_clear < ASSIGNMENT_EDGE_PENALTY_TRIGGER:
                            boundary_penalty = (
                                ASSIGNMENT_EDGE_PENALTY_TRIGGER - edge_clear
                            ) * ASSIGNMENT_EDGE_PENALTY_SECONDS_PER_M
                    owner_bonus = 0.0
                    if rg.label in detected_labels and name == owner:
                        owner_bonus = ASSIGNMENT_OWNER_BONUS_SECONDS
                    sticky_bonus = ASSIGNMENT_STICKY_BONUS_SECONDS if name in prev_owners else 0.0
                    if (not support_ready_by_usv.get(name, False)) and name != owner:
                        continue
                    cost = (
                        finish_at
                        + queue_penalty
                        + load_penalty
                        + boundary_penalty
                        - owner_bonus
                        - sticky_bonus
                        - wait_bonus_s
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_name = name
                selected.append(best_name)
                available.remove(best_name)

            if not selected:
                continue

            n_selected = max(1, len(selected))
            total_expected = max(1e-6, _expected_task_seconds(known))
            remain_ratio = min(1.0, remaining_work_s / total_expected)
            per_agent_work_s = _expected_agent_work_seconds(known, n_selected) * remain_ratio
            per_agent_work_eq = LOAD_WORK_SECONDS_TO_EQ_M * per_agent_work_s
            for name in selected:
                travel_s, travel_dist = travel_eta_seconds(name, rg)
                arrival = virtual_available_s[name] + travel_s
                virtual_available_s[name] = max(virtual_available_s[name], arrival) + per_agent_work_s
                virtual_load_eq[name] += per_agent_work_eq + LOAD_TRAVEL_COMMIT_RATIO * travel_dist
                if rg.label not in queued_labels_by_usv[name]:
                    priority_by_usv.setdefault(name, []).append((rg.x, rg.y))
                    queued_labels_by_usv[name].add(rg.label)
                    virtual_queue_count[name] += 1
                assignment_by_task.setdefault(rg.label, set()).add(name)

        assigned_usv_names = {nm for owners in assignment_by_task.values() for nm in owners}
        self_explore_points_by_usv: Dict[str, List[Tuple[float, float]]] = {}
        for name in usv_states.keys():
            if name in assigned_usv_names:
                continue
            idxs = cell_indices_by_agent.get(name, [])
            n_cells = len(idxs)
            remain_tol = max(SUPPORT_UNCOVERED_MIN_CELLS_TOL, int(round(SUPPORT_UNCOVERED_RATIO_TOL * max(1, n_cells))))
            uncovered_idx = [i for i in idxs if current_global_map[i] < OWN_REGION_EXPLORE_THRESHOLD]
            if len(uncovered_idx) <= remain_tol:
                continue
            self_explore_points_by_usv[name] = [stage1.grid_xy[i] for i in uncovered_idx]

        self_explore_paths: Dict[str, List[Tuple[float, float]]] = {}
        if self_explore_points_by_usv:
            self_explore_paths = plan_heterogeneous_paths(self_explore_points_by_usv, usv_states, usv_params)

        final_paths: Dict[str, List[Tuple[float, float]]] = {}
        for name, route in base_paths.items():
            st = usv_states[name]
            route_active = self_explore_paths.get(name, route)
            route_sparse_raw = [
                p
                for p in _coarsen_route(route_active, profile.route_spacing)
                if not env.in_obstacle(p[0], p[1])
                and not _point_too_close_to_obstacle(p[0], p[1], env, margin=OBSTACLE_SAFE_MARGIN + 2.0)
            ]
            route_sparse = _sanitize_waypoints_for_env(
                route_sparse_raw,
                env=env,
                boundary_margin=USV_ROUTE_BOUNDARY_MARGIN,
                min_gap=max(8.0, profile.route_spacing * 0.7),
            )
            assigned_stops_raw = _dedup_points(priority_by_usv.get(name, []), min_gap=10.0)
            target_stops = _sanitize_waypoints_for_env(
                assigned_stops_raw,
                env=env,
                boundary_margin=USV_ROUTE_BOUNDARY_MARGIN * 0.8,
                min_gap=10.0,
            )
            merged_path = prepend_priority_stops((st.x, st.y), route_sparse, target_stops)
            final_paths[name] = _sanitize_waypoints_for_env(
                merged_path,
                env=env,
                boundary_margin=USV_ROUTE_BOUNDARY_MARGIN,
                min_gap=IDLE_WAYPOINT_MIN_GAP,
            ) or [(st.x, st.y)]
            usv_by_name[name].waypoints = final_paths[name]
            usv_by_name[name].wp_index = _pick_waypoint_index_for_state(final_paths[name], st)

        for name in assign_sum:
            assign_sum[name] += partition.count_by_agent.get(name, 0)
        replan_count += 1

        latest_partition = partition
        owner_snapshot = list(partition.owner_by_cell)
        if partition_owner_snapshots and partition_owner_snapshots[-1] == owner_snapshot:
            active_partition_snapshot_idx = len(partition_owner_snapshots) - 1
        else:
            partition_owner_snapshots.append(owner_snapshot)
            active_partition_snapshot_idx = len(partition_owner_snapshots) - 1
        latest_priority_map = priority_map
        latest_detected_regions = detected_regions
        latest_final_paths = final_paths
        latest_priority_counts = {name: len(priority_by_usv.get(name, [])) for name in usv_states.keys()}
        task_assignment_active.clear()
        for rg in regions:
            owners = sorted(assignment_by_task.get(rg.label, set()))
            task_assignment_active[rg.label] = owners
            if owners:
                task_assignment_latest[rg.label] = owners
        task_assignment_history.append(
            (
                replan_count,
                {label: sorted(names) for label, names in task_assignment_latest.items()},
            )
        )

    def finalize_global_scan_discovery(sim_time: float) -> bool:
        changed = False
        for rg in regions:
            if known_difficulty_by_task.get(rg.label) is not None:
                continue
            known_difficulty_by_task[rg.label] = rg.difficulty
            difficulty_discovered_by[rg.label] = "uav_global_scan"
            difficulty_discovered_time[rg.label] = sim_time
            changed = True
        return changed

    do_replan()
    record_frame(t)
    mission_completion_time: float | None = None
    executed_steps = 0
    stage1_snapshot_step = min(n_steps, profile.max_execute_steps)
    track_interval = max(2, int(args.track_interval))

    for step_idx in range(profile.max_execute_steps):
        # UAV and USV run simultaneously from the first step.
        discovery_changed = False
        detections = []
        for uav in uavs:
            prev_x = uav.state.x
            prev_y = uav.state.y
            hold = uav_identify_countdown[uav.name]
            if hold > 0:
                ctrl = Control(a=0.0, omega=0.0)
                uav_identify_countdown[uav.name] = hold - 1
                if uav_identify_countdown[uav.name] == 0:
                    label = uav_identify_target[uav.name]
                    if label is not None and known_difficulty_by_task.get(label) is None:
                        rg = region_lookup[label]
                        known_difficulty_by_task[label] = rg.difficulty
                        difficulty_discovered_by[label] = uav.name
                        difficulty_discovered_time[label] = (step_idx + 1) * sim.dt
                        discovery_changed = True
                    uav_identify_target[uav.name] = None
            else:
                candidate: DetectionRegion | None = None
                best_d2 = 1e30
                for rg in regions:
                    if known_difficulty_by_task.get(rg.label) is not None:
                        continue
                    d2 = (uav.state.x - rg.x) ** 2 + (uav.state.y - rg.y) ** 2
                    near_r = max(uav_identify_radius, 0.52 * float(uav.params.sensor.radius), rg.radius + 2.0)
                    if d2 <= near_r * near_r and d2 < best_d2:
                        best_d2 = d2
                        candidate = rg
                if candidate is not None:
                    uav_identify_target[uav.name] = candidate.label
                    uav_identify_countdown[uav.name] = uav_hold_steps_required
                    ctrl = Control(a=0.0, omega=0.0)
                else:
                    uav.maybe_advance_waypoint(tol=10.0)
                    ctrl = uav_holonomic_control(uav.state, uav.current_target(), uav.params.v_max)
            uav.state = step_agent(uav.state, ctrl, uav.params, env, sim.dt, t)
            uav_trajectories[uav.name].append(uav.state)
            detections.append(uav_detection(stage1.grid_xy, uav.state, uav.params.sensor))
            # Deterministic sweep detection: if UAV track segment passes near a task region,
            # mark it discovered immediately to avoid frame-step misses.
            sweep_base_r = max(uav_identify_radius, 0.70 * float(uav.params.sensor.radius))
            for rg in regions:
                if known_difficulty_by_task.get(rg.label) is not None:
                    continue
                detect_r = max(sweep_base_r, rg.radius + 2.0)
                d2_seg = _point_to_segment_dist2(
                    (rg.x, rg.y),
                    (prev_x, prev_y),
                    (uav.state.x, uav.state.y),
                )
                if d2_seg > detect_r * detect_r:
                    continue
                known_difficulty_by_task[rg.label] = rg.difficulty
                difficulty_discovered_by[rg.label] = f"{uav.name}_sweep"
                difficulty_discovered_time[rg.label] = (step_idx + 1) * sim.dt
                discovery_changed = True
                if uav_identify_target.get(uav.name) == rg.label:
                    uav_identify_target[uav.name] = None
                    uav_identify_countdown[uav.name] = 0
        stage1.update(fused_detection(detections), sim.dt)

        # Once the global scan phase ends, promote any remaining unknown tasks
        # to known so every task point is guaranteed discovered.
        if (not global_scan_discovery_done) and (step_idx + 1 >= stage1_snapshot_step):
            sim_time_now = (step_idx + 1) * sim.dt
            if finalize_global_scan_discovery(sim_time_now):
                discovery_changed = True
            global_scan_discovery_done = True

        # During initial mapping, replan more frequently for tighter UAV->USV coupling.
        warmup_replan = max(8, profile.replan_interval_steps // 2)
        if step_idx + 1 <= stage1_snapshot_step:
            need_replan = (step_idx + 1) % warmup_replan == 0
        else:
            need_replan = (step_idx + 1) % profile.replan_interval_steps == 0
        if need_replan or discovery_changed:
            do_replan()

        if discovery_changed:
            discovered_labels_now = {label for label, diff in known_difficulty_by_task.items() if diff is not None}
            _inject_missing_region_waypoints(
                usv_by_name,
                regions,
                processing_done_time,
                known_difficulty_by_task,
                eligible_labels=discovered_labels_now,
                assignment_by_task=task_assignment_active,
            )

        # Share UAV detections with USVs and force online tracking of newly visible regions.
        if (step_idx + 1) % track_interval == 0:
            discovered_labels_now = {label for label, diff in known_difficulty_by_task.items() if diff is not None}
            _inject_missing_region_waypoints(
                usv_by_name,
                regions,
                processing_done_time,
                known_difficulty_by_task,
                eligible_labels=discovered_labels_now,
                assignment_by_task=task_assignment_active,
            )

        for nm in list(usv_task_lock.keys()):
            lbl = usv_task_lock.get(nm)
            if lbl is not None and processing_done_time.get(lbl) is not None:
                usv_task_lock[nm] = None
                usv_hold_anchor[nm] = None
                q = task_lock_order.get(lbl, [])
                if nm in q:
                    q.remove(nm)

        has_discovered_tasks = any(diff is not None for diff in known_difficulty_by_task.values())
        active_assigned_usv = {
            nm
            for owners in task_assignment_active.values()
            for nm in owners
        }
        pre_positions = {u.name: (u.state.x, u.state.y) for u in usvs}
        resolved_positions: Dict[str, Tuple[float, float]] = {}
        for usv in sorted(usvs, key=lambda a: a.name):
            if usv_detour_cooldown.get(usv.name, 0) > 0:
                usv_detour_cooldown[usv.name] = max(0, usv_detour_cooldown[usv.name] - 1)
            other_positions = list(resolved_positions.values()) + [
                pre_positions[name] for name in pre_positions.keys() if name != usv.name and name not in resolved_positions
            ]
            if not usv.waypoints:
                _assign_idle_waypoints(
                    usv,
                    env,
                    stage1.grid_xy,
                    latest_priority_map,
                    force_exploration=(not has_discovered_tasks),
                )

            currently_assigned = usv.name in active_assigned_usv
            was_assigned = usv_prev_assigned.get(usv.name, False)
            if was_assigned and (not currently_assigned):
                # Task was released: refresh to an exploration path immediately so the USV
                # does not keep orbiting around stale local task waypoints.
                _assign_idle_waypoints(
                    usv,
                    env,
                    stage1.grid_xy,
                    latest_priority_map,
                    force_exploration=True,
                )
                usv_idle_stuck_steps[usv.name] = 0
                usv_idle_stuck_anchor[usv.name] = None
                usv_idle_stuck_target[usv.name] = None

            lock_label = usv_task_lock.get(usv.name)
            cur_region = _pending_region_for_waypoint(
                usv.current_target(),
                regions,
                processing_done_time,
            )
            completed_target = _completed_region_for_waypoint(
                usv.current_target(),
                regions,
                processing_done_time,
            )
            if (not currently_assigned) and lock_label is None and completed_target is not None:
                # Completed task centers should not remain as idle targets.
                if usv.waypoints and len(usv.waypoints) > 1:
                    usv.wp_index = (usv.wp_index + 1) % len(usv.waypoints)
                else:
                    _assign_idle_waypoints(
                        usv,
                        env,
                        stage1.grid_xy,
                        latest_priority_map,
                        force_exploration=True,
                    )
                usv_idle_stuck_steps[usv.name] = 0
                usv_idle_stuck_anchor[usv.name] = None
                usv_idle_stuck_target[usv.name] = None
                cur_region = _pending_region_for_waypoint(
                    usv.current_target(),
                    regions,
                    processing_done_time,
                )
            if lock_label is None:
                active_rg = _active_region_for_state(usv.state, regions, processing_done_time)
                if active_rg is None and cur_region is not None:
                    active_rg = _capture_region_for_state(
                        usv.state,
                        regions,
                        processing_done_time,
                        prefer_label=cur_region.label,
                    )
                if active_rg is not None:
                    usv_task_lock[usv.name] = active_rg.label
                    q = task_lock_order.setdefault(active_rg.label, [])
                    if usv.name not in q:
                        q.append(usv.name)
                    usv_hold_anchor[usv.name] = None
                    lock_label = active_rg.label

            # Once entering a task region, enforce layered hard constraints:
            # primary worker can hold inside; others queue outside with strict spacing.
            if lock_label is not None and processing_done_time.get(lock_label) is None:
                anchor = usv_hold_anchor.get(usv.name)
                lock_rg = region_lookup.get(lock_label)
                q = task_lock_order.setdefault(lock_label, [])
                if usv.name not in q:
                    q.append(usv.name)
                slot_idx = q.index(usv.name) if usv.name in q else 0
                is_primary = slot_idx == 0
                need_refresh = anchor is None
                if lock_rg is not None and anchor is not None:
                    d_anchor = sqrt((anchor[0] - lock_rg.x) ** 2 + (anchor[1] - lock_rg.y) ** 2)
                    if is_primary and d_anchor > max(2.0, 1.35 * lock_rg.radius):
                        need_refresh = True
                    if (not is_primary) and d_anchor < max(2.0, 1.05 * lock_rg.radius):
                        need_refresh = True
                allow_inner_hold = False
                if (not is_primary) and lock_rg is not None:
                    d_now = sqrt((usv.state.x - lock_rg.x) ** 2 + (usv.state.y - lock_rg.y) ** 2)
                    allow_inner_hold = (
                        d_now <= 1.08 * lock_rg.radius
                        and not _position_conflict(
                            (usv.state.x, usv.state.y),
                            other_positions,
                            safe_distance=sim.safe_distance,
                        )
                    )
                if is_primary and lock_rg is not None:
                    d_center = sqrt((usv.state.x - lock_rg.x) ** 2 + (usv.state.y - lock_rg.y) ** 2)
                    if (
                        d_center <= 1.02 * lock_rg.radius
                        and not _position_conflict(
                            (usv.state.x, usv.state.y),
                            other_positions,
                            safe_distance=sim.safe_distance,
                        )
                    ):
                        usv.state = AgentState(
                            x=usv.state.x,
                            y=usv.state.y,
                            psi=usv.state.psi,
                            v=0.0,
                            energy=max(0.0, usv.state.energy - 0.12 * usv.params.base_power * sim.dt),
                        )
                        usv_prev_omega_cmd[usv.name] = 0.0
                        usv_turn_intent_sign[usv.name] = _update_turn_intent_sign(
                            usv_turn_intent_sign.get(usv.name, 0.0),
                            0.0,
                        )
                        usv_trajectories[usv.name].append(usv.state)
                        resolved_positions[usv.name] = (usv.state.x, usv.state.y)
                        q_det = usv_detection(stage1.grid_xy, usv.state, usv.params.sensor)
                        for i, val in enumerate(q_det):
                            usv_miss[i] *= 1.0 - val
                        usv_prev_assigned[usv.name] = currently_assigned
                        continue
                if anchor is not None and _position_conflict(anchor, other_positions, safe_distance=sim.safe_distance):
                    need_refresh = True
                if anchor is not None and lock_rg is not None:
                    # Hysteresis: don't reshuffle anchors unless change is meaningful.
                    preferred_probe = _pick_task_lock_anchor(
                        state=usv.state,
                        rg=lock_rg,
                        env=env,
                        other_positions=other_positions,
                        safe_distance=sim.safe_distance,
                        is_primary=is_primary,
                        queue_slot_idx=max(0, slot_idx - 1),
                        queue_slot_count=max(1, len(q) - 1),
                        allow_inner_hold=allow_inner_hold,
                    )
                    if sqrt((preferred_probe[0] - anchor[0]) ** 2 + (preferred_probe[1] - anchor[1]) ** 2) > LOCK_ANCHOR_HYSTERESIS_M:
                        need_refresh = True
                if need_refresh:
                    if lock_rg is not None:
                        anchor = _pick_task_lock_anchor(
                            state=usv.state,
                            rg=lock_rg,
                            env=env,
                            other_positions=other_positions,
                            safe_distance=sim.safe_distance,
                            is_primary=is_primary,
                            queue_slot_idx=max(0, slot_idx - 1),
                            queue_slot_count=max(1, len(q) - 1),
                            allow_inner_hold=allow_inner_hold,
                        )
                    else:
                        anchor = (usv.state.x, usv.state.y)
                    usv_hold_anchor[usv.name] = anchor
                d_anchor = sqrt((usv.state.x - anchor[0]) ** 2 + (usv.state.y - anchor[1]) ** 2)
                anchor_heading_error = 0.0
                if d_anchor > 1e-6:
                    anchor_heading_error = wrap_to_pi(atan2(anchor[1] - usv.state.y, anchor[0] - usv.state.x) - usv.state.psi)
                lock_ctrl = usv_control(
                    usv.state,
                    anchor,
                    usv.params,
                    heading_gain=max(profile.usv_heading_gain, 3.0),
                    speed_gain=max(profile.usv_speed_gain, 0.38),
                    turn_cap_ratio=max(profile.usv_turn_cap_ratio, 0.95),
                    allow_stop_near_target=True,
                )
                if d_anchor <= max(1.0, 0.18 * sim.safe_distance):
                    lock_ctrl = Control(
                        a=max(-0.50 * usv.params.a_max, min(lock_ctrl.a, 0.28 * usv.params.a_max)),
                        omega=lock_ctrl.omega,
                    )
                elif d_anchor <= max(2.2, 0.42 * sim.safe_distance):
                    lock_ctrl = Control(
                        a=max(-0.45 * usv.params.a_max, min(lock_ctrl.a, 0.32 * usv.params.a_max)),
                        omega=lock_ctrl.omega,
                    )
                prev_omega_cmd = usv_prev_omega_cmd.get(usv.name, 0.0)
                lock_omega = _rate_limit(lock_ctrl.omega, prev_omega_cmd, USV_OMEGA_RATE_LIMIT)
                if abs(lock_omega) < USV_OMEGA_DEADBAND:
                    lock_omega = 0.0
                lock_ctrl = Control(a=lock_ctrl.a, omega=lock_omega)
                prev_x, prev_y = usv.state.x, usv.state.y
                prev_psi = usv.state.psi
                usv.state = _safe_usv_step(
                    usv.state,
                    lock_ctrl,
                    usv.params,
                    env,
                    sim.dt,
                    t,
                    other_positions=other_positions,
                    safe_distance=sim.safe_distance,
                    preferred_turn_sign=usv_turn_intent_sign.get(usv.name, 0.0),
                    prev_omega=prev_omega_cmd,
                    target_heading_error=anchor_heading_error,
                    target_distance=d_anchor,
                )
                actual_omega = wrap_to_pi(usv.state.psi - prev_psi) / max(1e-6, sim.dt)
                usv_prev_omega_cmd[usv.name] = actual_omega
                usv_turn_intent_sign[usv.name] = _update_turn_intent_sign(
                    usv_turn_intent_sign.get(usv.name, 0.0),
                    actual_omega,
                )
                usv_travel_m[usv.name] += sqrt((usv.state.x - prev_x) ** 2 + (usv.state.y - prev_y) ** 2)
                usv_trajectories[usv.name].append(usv.state)
                resolved_positions[usv.name] = (usv.state.x, usv.state.y)
                q = usv_detection(stage1.grid_xy, usv.state, usv.params.sensor)
                for i, val in enumerate(q):
                    usv_miss[i] *= 1.0 - val
                usv_prev_assigned[usv.name] = currently_assigned
                continue

            detoured = _maybe_insert_minimal_detour(
                usv=usv,
                other_positions=other_positions,
                env=env,
                safe_distance=sim.safe_distance,
                regions=regions,
                processing_done_time=processing_done_time,
                cooldown_steps=usv_detour_cooldown.get(usv.name, 0),
            )
            if detoured:
                usv_detour_cooldown[usv.name] = USV_DETOUR_COOLDOWN_STEPS

            ahead_region = _nearest_pending_region_ahead(usv, regions, processing_done_time)
            if ahead_region is not None:
                cur_target = usv.current_target()
                if _dist2(cur_target, (ahead_region.x, ahead_region.y)) > 4.0 * 4.0:
                    remaining = usv.waypoints[usv.wp_index :] + usv.waypoints[: usv.wp_index]
                    remaining = [
                        p
                        for p in remaining
                        if _dist2(p, (ahead_region.x, ahead_region.y)) > 8.0 * 8.0
                    ]
                    usv.waypoints = [(ahead_region.x, ahead_region.y)] + remaining
                    usv.wp_index = 0

            waypoint_tol = max(profile.usv_waypoint_tol, IDLE_NAV_WAYPOINT_TOL)
            cur_target = usv.current_target()
            cur_region = _pending_region_for_waypoint(
                cur_target,
                regions,
                processing_done_time,
            )
            idle_mode = (usv.name not in active_assigned_usv) and (cur_region is None) and (lock_label is None)
            orbit_watch_mode = (cur_region is None) and (lock_label is None)
            if idle_mode:
                waypoint_tol = max(waypoint_tol, USV_IDLE_EXTRA_WAYPOINT_TOL)
            elif cur_region is None:
                waypoint_tol = max(waypoint_tol, IDLE_NAV_WAYPOINT_TOL + 1.5)
            if cur_region is not None:
                # For task waypoints, use tighter advance threshold to avoid skipping the task edge.
                waypoint_tol = min(waypoint_tol, max(1.8, 0.42 * cur_region.radius))

            usv.maybe_advance_waypoint(tol=waypoint_tol)
            cur_target = usv.current_target()
            cur_region = _pending_region_for_waypoint(
                cur_target,
                regions,
                processing_done_time,
            )
            idle_mode = (usv.name not in active_assigned_usv) and (cur_region is None) and (lock_label is None)
            orbit_watch_mode = (cur_region is None) and (lock_label is None)
            if orbit_watch_mode:
                prev_t = usv_idle_stuck_target.get(usv.name)
                if prev_t is None or _dist2(prev_t, cur_target) > 4.0 * 4.0:
                    usv_idle_stuck_target[usv.name] = cur_target
                    usv_idle_stuck_anchor[usv.name] = (usv.state.x, usv.state.y)
                    usv_idle_stuck_steps[usv.name] = 0
                anchor = usv_idle_stuck_anchor.get(usv.name)
                if anchor is None:
                    anchor = (usv.state.x, usv.state.y)
                    usv_idle_stuck_anchor[usv.name] = anchor
                moved = sqrt((usv.state.x - anchor[0]) ** 2 + (usv.state.y - anchor[1]) ** 2)
                if moved >= USV_IDLE_STUCK_PROGRESS_MIN:
                    usv_idle_stuck_anchor[usv.name] = (usv.state.x, usv.state.y)
                    usv_idle_stuck_steps[usv.name] = 0
                else:
                    usv_idle_stuck_steps[usv.name] = usv_idle_stuck_steps.get(usv.name, 0) + 1
                d_target = sqrt((usv.state.x - cur_target[0]) ** 2 + (usv.state.y - cur_target[1]) ** 2)
                if (
                    usv_idle_stuck_steps.get(usv.name, 0) >= USV_IDLE_STUCK_WINDOW_STEPS
                    and d_target <= USV_IDLE_STUCK_TARGET_RADIUS
                ):
                    if usv.waypoints and len(usv.waypoints) > 1:
                        usv.wp_index = (usv.wp_index + 1) % len(usv.waypoints)
                    else:
                        _assign_idle_waypoints(
                            usv,
                            env,
                            stage1.grid_xy,
                            latest_priority_map,
                            force_exploration=True,
                        )
                    usv_idle_stuck_target[usv.name] = usv.current_target()
                    usv_idle_stuck_anchor[usv.name] = (usv.state.x, usv.state.y)
                    usv_idle_stuck_steps[usv.name] = 0
                    cur_target = usv.current_target()
                    cur_region = _pending_region_for_waypoint(
                        cur_target,
                        regions,
                        processing_done_time,
                    )
                    idle_mode = (usv.name not in active_assigned_usv) and (cur_region is None) and (lock_label is None)
            else:
                usv_idle_stuck_steps[usv.name] = 0
                usv_idle_stuck_anchor[usv.name] = None
                usv_idle_stuck_target[usv.name] = None

            target_heading_error = wrap_to_pi(atan2(cur_target[1] - usv.state.y, cur_target[0] - usv.state.x) - usv.state.psi)
            dist_to_target = sqrt((usv.state.x - cur_target[0]) ** 2 + (usv.state.y - cur_target[1]) ** 2)
            ctrl = usv_control(
                usv.state,
                cur_target,
                usv.params,
                heading_gain=profile.usv_heading_gain,
                speed_gain=profile.usv_speed_gain,
                turn_cap_ratio=profile.usv_turn_cap_ratio,
                allow_stop_near_target=(cur_region is not None or idle_mode or dist_to_target <= 24.0),
            )
            cur_region = _pending_region_for_waypoint(
                cur_target,
                regions,
                processing_done_time,
            )
            if cur_region is not None:
                d = sqrt((usv.state.x - cur_region.x) ** 2 + (usv.state.y - cur_region.y) ** 2)
                if d <= 1.35 * cur_region.radius:
                    # Near task center, cap speed while preserving enough thrust to resist ocean current drift.
                    ctrl = Control(
                        a=max(-0.60 * usv.params.a_max, min(ctrl.a, 0.30 * usv.params.a_max)),
                        omega=ctrl.omega,
                    )
                if d <= 0.95 * cur_region.radius:
                    ctrl = Control(
                        a=max(-0.80 * usv.params.a_max, min(ctrl.a, 0.20 * usv.params.a_max)),
                        omega=ctrl.omega,
                    )
            prev_omega_cmd = usv_prev_omega_cmd.get(usv.name, 0.0)
            ctrl_omega = _rate_limit(ctrl.omega, prev_omega_cmd, USV_OMEGA_RATE_LIMIT)
            if abs(ctrl_omega) < USV_OMEGA_DEADBAND:
                ctrl_omega = 0.0
            ctrl = Control(a=ctrl.a, omega=ctrl_omega)
            prev_x, prev_y = usv.state.x, usv.state.y
            prev_psi = usv.state.psi
            usv.state = _safe_usv_step(
                usv.state,
                ctrl,
                usv.params,
                env,
                sim.dt,
                t,
                other_positions=other_positions,
                safe_distance=sim.safe_distance,
                preferred_turn_sign=usv_turn_intent_sign.get(usv.name, 0.0),
                prev_omega=prev_omega_cmd,
                target_heading_error=target_heading_error,
                target_distance=dist_to_target,
            )
            actual_omega = wrap_to_pi(usv.state.psi - prev_psi) / max(1e-6, sim.dt)
            usv_prev_omega_cmd[usv.name] = actual_omega
            usv_turn_intent_sign[usv.name] = _update_turn_intent_sign(
                usv_turn_intent_sign.get(usv.name, 0.0),
                actual_omega,
            )
            usv_travel_m[usv.name] += sqrt((usv.state.x - prev_x) ** 2 + (usv.state.y - prev_y) ** 2)
            usv_trajectories[usv.name].append(usv.state)
            resolved_positions[usv.name] = (usv.state.x, usv.state.y)

            q = usv_detection(stage1.grid_xy, usv.state, usv.params.sensor)
            for i, val in enumerate(q):
                usv_miss[i] *= 1.0 - val
            usv_prev_assigned[usv.name] = currently_assigned

        sim_time = (step_idx + 1) * sim.dt
        for usv in usvs:
            traj = usv_trajectories[usv.name]
            if len(traj) < 2:
                continue
            prev = traj[-2]
            cur = traj[-1]
            dx = cur.x - prev.x
            dy = cur.y - prev.y
            dist = sqrt(dx * dx + dy * dy)
            if dist < 1e-9:
                continue
            usv_dist_sum_cum += dist
            usv_turn_sum_cum += abs(wrap_to_pi(cur.psi - prev.psi))
        usv_discovery_changed = _update_region_processing(
            usvs=usvs,
            regions=regions,
            processing_progress=processing_progress,
            processing_done_time=processing_done_time,
            processing_required=processing_required,
            first_hit_time=first_hit_time,
            known_difficulty_by_task=known_difficulty_by_task,
            difficulty_discovered_by=difficulty_discovered_by,
            difficulty_discovered_time=difficulty_discovered_time,
            usv_work_s=usv_work_s,
            sim_time=sim_time,
            dt=sim.dt,
        )
        if usv_discovery_changed:
            do_replan()
            discovered_labels_now = {label for label, diff in known_difficulty_by_task.items() if diff is not None}
            _inject_missing_region_waypoints(
                usv_by_name,
                regions,
                processing_done_time,
                known_difficulty_by_task,
                eligible_labels=discovered_labels_now,
                assignment_by_task=task_assignment_active,
            )
        for rg in regions:
            if processing_done_time.get(rg.label) is not None:
                max_contributors_by_region[rg.label] = max(max_contributors_by_region[rg.label], 1)
            else:
                n = 0
                rr2 = (1.02 * rg.radius) ** 2
                for usv in usvs:
                    if (usv.state.x - rg.x) ** 2 + (usv.state.y - rg.y) ** 2 <= rr2:
                        n += 1
                max_contributors_by_region[rg.label] = max(max_contributors_by_region[rg.label], n)

        if (step_idx + 1) % profile.urgent_insert_interval == 0:
            discovered_labels_now = {label for label, diff in known_difficulty_by_task.items() if diff is not None}
            _inject_missing_region_waypoints(
                usv_by_name,
                regions,
                processing_done_time,
                known_difficulty_by_task,
                eligible_labels=discovered_labels_now,
                assignment_by_task=task_assignment_active,
            )

        t += sim.dt
        executed_steps = step_idx + 1
        record_frame(t)

        if executed_steps == stage1_snapshot_step and stage1_coverage is None:
            stage1_coverage = list(stage1.coverage_quality)
            stage1_uav_trajectories = {k: list(v) for k, v in uav_trajectories.items()}
            stage1_mean_scan_quality = sum(stage1_coverage) / len(stage1_coverage)

        if (step_idx + 1) % 10 == 0:
            usv_cov_now = [1.0 - m for m in usv_miss]
            uav_cov_now = stage1.coverage_quality
            mission_cov_now = [
                1.0 - (1.0 - uav_cov_now[i]) * (1.0 - usv_cov_now[i]) for i in range(grid_size)
            ]
            coverage_rate_now = sum(1 for c in mission_cov_now if c >= sim.coverage_threshold) / grid_size
            handled_now = sum(1 for tm in processing_done_time.values() if tm is not None)
            if coverage_rate_now >= targets.coverage_goal and handled_now == len(regions):
                mission_completion_time = executed_steps * sim.dt
                break

    if latest_partition is None:
        raise RuntimeError("replanning failed to initialize")
    if stage1_coverage is None:
        stage1_coverage = list(stage1.coverage_quality)
        stage1_uav_trajectories = {k: list(v) for k, v in uav_trajectories.items()}
        stage1_mean_scan_quality = sum(stage1_coverage) / len(stage1_coverage)
    if stage1_uav_trajectories is None:
        stage1_uav_trajectories = {k: list(v) for k, v in uav_trajectories.items()}

    usv_coverage = [1.0 - m for m in usv_miss]
    mean_usv_coverage = sum(usv_coverage) / grid_size
    uav_coverage_final = stage1.coverage_quality
    mission_coverage = [
        1.0 - (1.0 - uav_coverage_final[i]) * (1.0 - usv_coverage[i]) for i in range(grid_size)
    ]
    coverage_rate = sum(1 for c in mission_coverage if c >= sim.coverage_threshold) / grid_size
    usv_only_coverage_rate = sum(1 for c in usv_coverage if c >= sim.coverage_threshold) / grid_size

    handled_all = sum(1 for tm in processing_done_time.values() if tm is not None)
    all_problem_rate = handled_all / len(regions) if regions else 0.0
    handled_detected = sum(
        1 for rg in latest_detected_regions if processing_done_time.get(rg.label) is not None
    )
    detected_problem_rate = handled_detected / len(latest_detected_regions) if latest_detected_regions else 0.0

    hit_times = [tm for tm in first_hit_time.values() if tm is not None]
    avg_response_time = sum(hit_times) / len(hit_times) if hit_times else float("inf")
    completion_times = [tm for tm in processing_done_time.values() if tm is not None]
    avg_processing_completion_time = (
        sum(completion_times) / len(completion_times) if completion_times else float("inf")
    )
    known_difficulty_count = sum(1 for v in known_difficulty_by_task.values() if v is not None)
    if mission_completion_time is None:
        mission_completion_time = executed_steps * sim.dt

    turn_sum = 0.0
    dist_sum = 0.0
    events_sum = 0
    for usv in usvs:
        total_turn, total_dist, _, turn_events = compute_turning_metrics_from_states(usv_trajectories[usv.name])
        turn_sum += total_turn
        dist_sum += total_dist
        events_sum += turn_events
    turn_index_global = turn_sum / max(1e-6, dist_sum)
    usv_load_eq_m = {
        name: _usv_load_equivalent(usv_travel_m.get(name, 0.0), usv_work_s.get(name, 0.0))
        for name in usv_by_name.keys()
    }
    load_balance_cv = _coefficient_of_variation(list(usv_load_eq_m.values()))

    if generate_outputs:
        focus_regions = [(rg.x, rg.y, rg.radius, rg.label) for rg in regions]
        focus_detected = [(rg.x, rg.y, rg.radius, rg.label) for rg in latest_detected_regions]
        final_usv_states = {a.name: a.state for a in usvs}
        executed_paths = {name: [(st.x, st.y) for st in states] for name, states in usv_trajectories.items()}

        if args.visualize:
            svg1 = save_snapshot_svg(
                output_path=args.phase1_svg,
                env=env,
                sim=sim,
                trajectories=stage1_uav_trajectories,
                params_map=uav_params,
                grid_xy=stage1.grid_xy,
                coverage_quality=stage1_coverage,
                focus_regions=focus_regions,
                title=f"Stage 1 UAV Scan | seed={mission_seed} | algo={profile.name}",
            )
            svg2 = save_partition_svg(
                output_path=args.phase2_svg,
                env=env,
                sim=sim,
                grid_xy=stage1.grid_xy,
                owner_by_cell=latest_partition.owner_by_cell,
                priority_map=latest_priority_map,
                states=final_usv_states,
                params_map=usv_params,
                planned_paths=executed_paths,
                color_by_agent=agent_colors,
                focus_regions=focus_detected,
                title=f"Stage 2-3 USV Routes | seed={mission_seed} | algo={profile.name}",
            )
            print(f"SVG exported: {svg1}")
            print(f"SVG exported: {svg2}")

        if args.animate:
            obstacle_data = [(o.xmin, o.xmax, o.ymin, o.ymax) for o in env.obstacles]
            stage1_end_frame = min(stage1_snapshot_step, len(frame_times) - 1)
            region_done_time = {
                label: float(done_at)
                for label, done_at in processing_done_time.items()
                if done_at is not None
            }
            html_path = save_mission_animation_html(
                output_path=args.animation_html,
                env=env,
                obstacles=obstacle_data,
                focus_regions=focus_regions,
                region_done_time=region_done_time,
                frame_known_difficulty=frame_known_difficulty,
                frame_task_assignment=frame_task_assignment,
                frame_coverage_rate=frame_coverage_rate,
                frame_task_completion_rate=frame_task_completion_rate,
                frame_turn_rate=frame_turn_rate,
                frame_load_table=frame_load_table,
                frame_partition_snapshot_idx=frame_partition_snapshot_idx,
                partition_owner_snapshots=partition_owner_snapshots,
                partition_grid_xy=stage1.grid_xy,
                partition_grid_shape=(sim.nx, sim.ny),
                frame_times=frame_times,
                agent_types=agent_types,
                agent_states=frame_states,
                current_model=env.current.as_dict(),
                agent_colors=agent_colors,
                stage1_end_frame=stage1_end_frame,
                title=(
                    f"UAV-USV Mission Playback | seed={mission_seed} | "
                    f"algo={profile.name} | scenario={args.scenario_profile}"
                ),
            )
            print(f"Animation HTML exported: {html_path}")

    mean_scan_quality = sum(stage1.coverage_quality) / len(stage1.coverage_quality)
    latest_path_lengths = {name: len(path) for name, path in latest_final_paths.items()}
    log_path = _write_run_log(
        log_dir=Path(args.log_dir),
        mission_seed=mission_seed,
        profile_name=profile.name,
        scenario_profile=args.scenario_profile,
        env=env,
        regions=regions,
        stage1_snapshot_step=stage1_snapshot_step,
        replan_count=replan_count,
        executed_steps=executed_steps,
        coverage_rate=coverage_rate,
        all_problem_rate=all_problem_rate,
        turn_events=events_sum,
        turn_index_global=turn_index_global,
        mission_completion_time=mission_completion_time,
        avg_response_time=avg_response_time,
        avg_processing_completion_time=avg_processing_completion_time,
        processing_required=processing_required,
        processing_progress=processing_progress,
        processing_done_time=processing_done_time,
        first_hit_time=first_hit_time,
        known_difficulty_by_task=known_difficulty_by_task,
        difficulty_discovered_by=difficulty_discovered_by,
        difficulty_discovered_time=difficulty_discovered_time,
        task_assignment_latest=task_assignment_latest,
        task_assignment_history=task_assignment_history,
        max_contributors_by_region=max_contributors_by_region,
        usv_travel_m=usv_travel_m,
        usv_work_s=usv_work_s,
        usv_load_eq_m=usv_load_eq_m,
        load_balance_cv=load_balance_cv,
    )

    return MissionSummary(
        mission_seed=mission_seed,
        profile_name=profile.name,
        scenario_profile=args.scenario_profile,
        sea_width=env.xlim[1] - env.xlim[0],
        sea_height=env.ylim[1] - env.ylim[0],
        num_obstacles=len(env.obstacles),
        num_regions=len(regions),
        grid_cells=len(stage1.grid_xy),
        stage1_mean_scan_quality=stage1_mean_scan_quality,
        rolling_map_quality=mean_scan_quality,
        latest_detected_regions=len(latest_detected_regions),
        replan_count=replan_count,
        executed_steps=executed_steps,
        latest_partition_counts=dict(latest_partition.count_by_agent),
        assign_sum=assign_sum,
        latest_priority_counts=latest_priority_counts,
        latest_path_lengths=latest_path_lengths,
        coverage_rate=coverage_rate,
        usv_only_coverage_rate=usv_only_coverage_rate,
        mean_usv_coverage=mean_usv_coverage,
        all_problem_rate=all_problem_rate,
        handled_all=handled_all,
        detected_problem_rate=detected_problem_rate,
        handled_detected=handled_detected,
        turn_index_global=turn_index_global,
        turn_sum=turn_sum,
        turn_events=events_sum,
        dist_sum=dist_sum,
        usv_travel_m=usv_travel_m,
        usv_work_s=usv_work_s,
        usv_load_eq_m=usv_load_eq_m,
        load_balance_cv=load_balance_cv,
        mission_completion_time=mission_completion_time,
        avg_response_time=avg_response_time,
        avg_processing_completion_time=avg_processing_completion_time,
        known_difficulty_count=known_difficulty_count,
        log_path=log_path,
        meets_targets=(coverage_rate >= targets.coverage_goal and all_problem_rate >= targets.handling_goal),
    )


def print_summary(
    summary: MissionSummary,
    requested_algorithm: str,
    seed_mode: str,
    targets: MissionTargets,
) -> None:
    print("=== Step 3 Mission Workflow Demo ===")
    print(
        f"scenario_seed: {summary.mission_seed} (mode={seed_mode}), "
        f"algorithm_request={requested_algorithm}, selected_algorithm={summary.profile_name}, "
        f"scenario_profile={summary.scenario_profile}"
    )
    print(f"log_file: {summary.log_path}")

    print("Scenario")
    print(
        f"  sea_area: {summary.sea_width:.0f}m x {summary.sea_height:.0f}m, "
        f"obstacles={summary.num_obstacles}, target_regions={summary.num_regions}, "
        f"known_difficulties={summary.known_difficulty_count}/{summary.num_regions}, grid_cells={summary.grid_cells}"
    )

    print("Stage 1 | UAV global scan")
    print(
        f"  initial_map_quality={summary.stage1_mean_scan_quality:.3f}, "
        f"rolling_map_quality={summary.rolling_map_quality:.3f}, "
        f"detected_regions={summary.latest_detected_regions}/{summary.num_regions}"
    )

    print("Stage 2 | USV full coverage assignment")
    print(f"  replans={summary.replan_count}, executed_steps={summary.executed_steps}")
    for name in sorted(summary.latest_partition_counts.keys()):
        cells_latest = summary.latest_partition_counts[name]
        ratio_latest = cells_latest / summary.grid_cells if summary.grid_cells else 0.0
        cells_avg = summary.assign_sum[name] / max(1, summary.replan_count)
        ratio_avg = cells_avg / summary.grid_cells if summary.grid_cells else 0.0
        print(
            f"  {name}: latest={cells_latest} ({ratio_latest:.2%}), "
            f"rolling_avg={cells_avg:.1f} ({ratio_avg:.2%})"
        )

    print("Stage 3 | Priority inspection insertion")
    for name in sorted(summary.latest_path_lengths.keys()):
        n_priority = summary.latest_priority_counts.get(name, 0)
        print(f"  {name}: priority_stops={n_priority}, final_waypoints={summary.latest_path_lengths[name]}")

    print("Final Metrics")
    print(
        f"  1. 覆盖率: {summary.coverage_rate:.2%} "
        f"(target>={targets.coverage_goal:.0%}, "
        f"mission=UAV+USV, USV-only={summary.usv_only_coverage_rate:.2%}, USV mean={summary.mean_usv_coverage:.3f})"
    )
    print(
        f"  2. 问题区域处理率: {summary.all_problem_rate:.2%} "
        f"(target={targets.handling_goal:.0%}, handled={summary.handled_all}/{summary.num_regions}, "
        f"detected_rate={summary.detected_problem_rate:.2%}, detected-handled={summary.handled_detected}/{summary.latest_detected_regions})"
    )
    print(
        f"  3. 无人艇转弯指标: {summary.turn_index_global:.5f} rad/m "
        f"(total_turn={summary.turn_sum * 180.0 / pi:.1f} deg, turn_events={summary.turn_events}, "
        f"total_distance={summary.dist_sum:.1f} m)"
    )
    print(
        f"  4. 时间指标: 任务达标时间={summary.mission_completion_time:.1f}s, "
        f"问题区域平均响应时间={summary.avg_response_time:.1f}s, "
        f"问题区域平均处理完成时间={summary.avg_processing_completion_time:.1f}s"
    )
    print(
        f"  5. 负载均衡: CV={summary.load_balance_cv:.3f} "
        "(越低越均衡; 负载=航行距离+作业时间折算)"
    )
    for nm in sorted(summary.usv_load_eq_m.keys()):
        print(
            f"     {nm}: travel={summary.usv_travel_m.get(nm, 0.0):.1f}m, "
            f"work={summary.usv_work_s.get(nm, 0.0):.1f}s, "
            f"load_eq={summary.usv_load_eq_m.get(nm, 0.0):.1f}"
        )
    print(f"Target check: meets_targets={summary.meets_targets}")


def run_mission(args: argparse.Namespace) -> None:
    mission_seed = resolve_seed(args)
    targets = MissionTargets(
        coverage_goal=max(0.0, min(1.0, args.coverage_goal)),
        handling_goal=max(0.0, min(1.0, args.handling_goal)),
    )
    profile = make_algorithm_profile(args.algorithm)
    best = run_mission_once(
        args=args,
        mission_seed=mission_seed,
        targets=targets,
        profile=profile,
        generate_outputs=(args.visualize or args.animate),
    )

    print_summary(
        summary=best,
        requested_algorithm=args.algorithm,
        seed_mode=args.seed_mode,
        targets=targets,
    )


def parse_args() -> argparse.Namespace:
    base = Path("/Users/tom/Documents/Multi-Agent/outputs")
    parser = argparse.ArgumentParser(description="Large-area UAV-USV mission workflow demo")
    parser.add_argument(
        "--algorithm",
        choices=[CURRENT_ALGORITHM],
        default=CURRENT_ALGORITHM,
        help="Current mission planning algorithm. More algorithms can be added later for comparison.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["random", "fixed"],
        default="random",
        help="Use random seed each run, or fixed seed for reproducible map generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed used when --seed-mode fixed.",
    )
    parser.add_argument(
        "--scenario-profile",
        choices=list(SCENARIO_PROFILES),
        default="random",
        help="Structured map/task generator profile for highlighting heterogeneous collaboration.",
    )
    parser.add_argument(
        "--coverage-goal",
        type=float,
        default=0.90,
        help="Coverage target for selection and early stop, default 0.90.",
    )
    parser.add_argument(
        "--handling-goal",
        type=float,
        default=1.0,
        help="Problem-region handling target for selection and early stop, default 1.00.",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=420,
        help="Number of UAV warm-up scan steps in stage 1.",
    )
    parser.add_argument(
        "--track-interval",
        type=int,
        default=4,
        help="USV realtime tracking refresh interval (steps) while sharing UAV view.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export stage visualizations as SVG (default: enabled). Use --no-visualize to disable.",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export interactive animation HTML (default: enabled). Use --no-animate to disable.",
    )
    parser.add_argument(
        "--phase1-svg",
        default=str(base / "step3_phase1_uav_map.svg"),
        help="Stage-1 UAV map SVG output path",
    )
    parser.add_argument(
        "--phase2-svg",
        default=str(base / "step3_phase2_usv_routes.svg"),
        help="Stage-2/3 USV route SVG output path",
    )
    parser.add_argument(
        "--animation-html",
        default=str(base / "step3_mission_animation.html"),
        help="Interactive mission animation HTML output path",
    )
    parser.add_argument(
        "--log-dir",
        default=str(base / "logs"),
        help="Directory for run logs. A new log file is generated every run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_mission(parse_args())
