from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import atan2, pi, sqrt
from typing import Dict, List, Tuple

from maritime_coverage import (
    AgentParams,
    AgentState,
    AgentType,
    Control,
    CoverageProblem,
    Environment,
    ObstacleRect,
    OceanCurrent,
    SensorParams,
    SimulationConfig,
    WindField,
    fused_detection,
    save_snapshot_svg,
    step_agent,
    uav_detection,
    usv_detection,
)


def wrap_to_pi(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def waypoint_control(
    state: AgentState,
    target: Tuple[float, float],
    params: AgentParams,
) -> Control:
    tx, ty = target
    dx = tx - state.x
    dy = ty - state.y
    dist = sqrt(dx * dx + dy * dy)

    if params.agent_type == AgentType.UAV:
        # UAV uses holonomic velocity commands (ux, uy).
        gain = 0.55
        ux = gain * dx
        uy = gain * dy
        speed = sqrt(ux * ux + uy * uy)
        if speed > params.v_max and speed > 1e-9:
            s = params.v_max / speed
            ux *= s
            uy *= s
        if dist < 1.2:
            ux = 0.0
            uy = 0.0
        return Control(a=float(ux), omega=float(uy))

    desired_heading = atan2(dy, dx)
    heading_error = wrap_to_pi(desired_heading - state.psi)
    omega = max(-params.omega_max, min(1.2 * heading_error, params.omega_max))
    desired_speed = max(params.v_min, min(0.4 * dist, params.v_max))
    a = max(-params.a_max, min(desired_speed - state.v, params.a_max))
    return Control(a=float(a), omega=float(omega))


@dataclass
class AgentRunner:
    name: str
    params: AgentParams
    state: AgentState
    waypoints: List[Tuple[float, float]]
    wp_index: int = 0

    def current_target(self) -> Tuple[float, float]:
        return self.waypoints[self.wp_index]

    def maybe_advance_waypoint(self) -> None:
        tx, ty = self.current_target()
        if ((self.state.x - tx) ** 2 + (self.state.y - ty) ** 2) ** 0.5 < 6.0:
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)


@dataclass
class SimulationResult:
    env: Environment
    sim: SimulationConfig
    problem: CoverageProblem
    trajectories: Dict[str, List[AgentState]]
    params_map: Dict[str, AgentParams]
    final_states: Dict[str, AgentState]
    objective: float
    mean_quality: float
    coverage_ratio: float
    max_gap: float
    no_collision: bool
    no_obstacle_violation: bool
    energy_ok: bool
    coverage_ok: bool
    revisit_ok: bool
    collision_count: int
    obstacle_hits: int
    low_energy_agents: int
    uncovered_cells: int
    high_revisit_cells: int


def build_demo():
    env = Environment(
        xlim=(0.0, 220.0),
        ylim=(0.0, 160.0),
        obstacles=[ObstacleRect(95.0, 125.0, 60.0, 100.0)],
        current=OceanCurrent(c0x=0.35, c0y=0.12, gx=0.0, gy=0.0),
        wind=WindField(wx=0.4, wy=0.1),
    )

    sim = SimulationConfig(
        dt=0.5,
        horizon=220.0,
        nx=50,
        ny=36,
        safe_distance=8.0,
        coverage_threshold=0.75,
        revisit_max_gap=45.0,
        seen_prob_threshold=0.30,
        control_penalty=0.02,
    )

    uav_sensor = SensorParams(radius=36.0, kappa=0.95, sigma_r=17.0, fov=2.0 * pi)
    usv_sensor = SensorParams(radius=20.0, kappa=0.88, sigma_r=10.0, fov=pi / 2.0, sigma_phi=0.5)

    uav_params = AgentParams(
        agent_type=AgentType.UAV,
        v_min=0.0,
        v_max=20.0,
        omega_max=99.0,
        a_max=20.0,
        energy_min=1200.0,
        base_power=8.0,
        kv=0.08,
        kw=0.0,
        sensor=uav_sensor,
    )
    usv_params = AgentParams(
        agent_type=AgentType.USV,
        v_min=2.0,
        v_max=7.5,
        omega_max=0.55,
        a_max=1.2,
        energy_min=900.0,
        base_power=4.5,
        kv=0.22,
        kw=2.2,
        sensor=usv_sensor,
    )

    agents = [
        AgentRunner(
            name="uav_1",
            params=uav_params,
            state=AgentState(x=18.0, y=22.0, psi=0.0, v=11.0, energy=12000.0),
            waypoints=[(30, 30), (190, 30), (190, 140), (30, 140)],
        ),
        AgentRunner(
            name="uav_2",
            params=uav_params,
            state=AgentState(x=198.0, y=18.0, psi=pi, v=10.5, energy=12000.0),
            waypoints=[(190, 20), (30, 20), (30, 150), (190, 150)],
        ),
        AgentRunner(
            name="usv_1",
            params=usv_params,
            state=AgentState(x=24.0, y=112.0, psi=-0.6, v=3.5, energy=5000.0),
            waypoints=[(24, 120), (72, 36), (150, 36), (190, 118), (40, 126)],
        ),
        AgentRunner(
            name="usv_2",
            params=usv_params,
            state=AgentState(x=46.0, y=42.0, psi=0.2, v=3.3, energy=5000.0),
            waypoints=[(48, 34), (92, 24), (172, 26), (204, 70), (112, 48), (46, 42)],
        ),
        AgentRunner(
            name="usv_3",
            params=usv_params,
            state=AgentState(x=30.0, y=146.0, psi=-0.3, v=3.2, energy=5000.0),
            waypoints=[(36, 144), (88, 136), (170, 138), (205, 92), (118, 116), (32, 142)],
        ),
    ]

    return env, sim, agents


def run_simulation() -> SimulationResult:
    env, sim, agents = build_demo()
    problem = CoverageProblem(env, sim)
    steps = int(sim.horizon / sim.dt)

    trajectories: Dict[str, List[AgentState]] = {a.name: [a.state] for a in agents}
    control_history: Dict[str, List[Tuple[float, float]]] = {a.name: [] for a in agents}
    params_map: Dict[str, AgentParams] = {a.name: a.params for a in agents}

    t = 0.0
    for _ in range(steps):
        detections = []
        for agent in agents:
            agent.maybe_advance_waypoint()
            target = agent.current_target()
            control = waypoint_control(agent.state, target, agent.params)
            agent.state = step_agent(agent.state, control, agent.params, env, sim.dt, t)

            trajectories[agent.name].append(agent.state)
            control_history[agent.name].append((control.a, control.omega))

            if agent.params.agent_type == AgentType.UAV:
                q = uav_detection(problem.grid_xy, agent.state, agent.params.sensor)
            else:
                q = usv_detection(problem.grid_xy, agent.state, agent.params.sensor)
            detections.append(q)

        fused = fused_detection(detections)
        problem.update(fused, sim.dt)
        t += sim.dt

    final_states = {a.name: a.state for a in agents}
    report = problem.evaluate_constraints(trajectories, final_states, params_map)
    objective = problem.objective(control_history)

    quality = problem.coverage_quality
    covered_cells = sum(1 for q in quality if q >= sim.coverage_threshold)
    coverage_ratio = covered_cells / len(quality)
    mean_quality = sum(quality) / len(quality)
    max_gap = max(problem.max_gap) if problem.max_gap else 0.0

    return SimulationResult(
        env=env,
        sim=sim,
        problem=problem,
        trajectories=trajectories,
        params_map=params_map,
        final_states=final_states,
        objective=objective,
        mean_quality=mean_quality,
        coverage_ratio=coverage_ratio,
        max_gap=max_gap,
        no_collision=report.no_collision,
        no_obstacle_violation=report.no_obstacle_violation,
        energy_ok=report.energy_ok,
        coverage_ok=report.coverage_ok,
        revisit_ok=report.revisit_ok,
        collision_count=report.collision_count,
        obstacle_hits=report.obstacle_hits,
        low_energy_agents=report.low_energy_agents,
        uncovered_cells=report.uncovered_cells,
        high_revisit_cells=report.high_revisit_cells,
    )


def print_summary(result: SimulationResult) -> None:
    sim = result.sim
    quality = result.problem.coverage_quality

    print("=== Step 1: Heterogeneous Maritime Coverage Modeling Demo ===")
    print(f"Grid cells: {len(quality)}")
    print(f"Mean coverage quality: {result.mean_quality:.4f}")
    print(f"Coverage ratio (>= {sim.coverage_threshold:.2f}): {result.coverage_ratio:.4f}")
    print(f"Max revisit gap: {result.max_gap:.2f} s")
    print(f"Objective value: {result.objective:.2f}")
    print("Constraint flags:")
    print(f"  no_collision: {result.no_collision} (count={result.collision_count})")
    print(f"  no_obstacle_violation: {result.no_obstacle_violation} (hits={result.obstacle_hits})")
    print(f"  energy_ok: {result.energy_ok} (low_energy_agents={result.low_energy_agents})")
    print(f"  coverage_ok: {result.coverage_ok} (uncovered_cells={result.uncovered_cells})")
    print(f"  revisit_ok: {result.revisit_ok} (high_revisit_cells={result.high_revisit_cells})")
    for name, st in result.final_states.items():
        print(
            f"  final_state[{name}] = (x={st.x:.1f}, y={st.y:.1f}, psi={st.psi:.2f}, v={st.v:.2f}, E={st.energy:.1f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 heterogeneous UAV-USV modeling demo")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export an SVG snapshot with UAV/USV shapes, trajectories, and coverage heatmap",
    )
    parser.add_argument(
        "--svg",
        default="outputs/step1_snapshot.svg",
        help="Output SVG path used with --visualize",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_simulation()
    print_summary(result)
    if args.visualize:
        svg_path = save_snapshot_svg(
            output_path=args.svg,
            env=result.env,
            sim=result.sim,
            trajectories=result.trajectories,
            params_map=result.params_map,
            grid_xy=result.problem.grid_xy,
            coverage_quality=result.problem.coverage_quality,
            title="Step 1 UAV-USV Cooperative Coverage (Snapshot)",
        )
        print(f"SVG exported: {svg_path}")
